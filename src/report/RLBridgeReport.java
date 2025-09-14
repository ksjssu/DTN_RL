/*
 * RL Bridge report: synchronous per-step update with DRL module.
 * At each sample interval: send prev transitions + current state; receive actions (deltas) and apply.
 */
package report;

import core.Connection;
import core.DTNHost;
import core.Message;
import core.Settings;
import core.SimClock;
import core.UpdateListener;
import routing.MessageRouter;
import routing.ProphetRouter;
import routing.ProphetV2Router;
import routing.ProphetRouterWithEstimation;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.*;

public class RLBridgeReport extends SamplingReport implements UpdateListener, core.MessageListener {

    // Settings
    public static final String URL_S = "url";
    public static final String WINDOW_SIZE_S = "windowSize";
    public static final String CONTACTS_NORM_MODE_S = "contactsNormMode"; // cmax|saturate
    public static final String CONTACTS_CMAX_S = "contactsCmax";
    public static final String CONTACTS_TAU_S = "contactsTau";
    public static final String MAX_MSG_PER_NODE_S = "maxMessagesPerNode"; // 0=unlimited
    public static final String DELTA_LIMIT_S = "deltaLimit";
    public static final String TIMEOUT_MS_S = "timeoutMs";
    public static final String UNBOUNDED_DELTA_S = "unboundedDelta"; // true|false
    public static final String LOG_ACTIONS_S = "logActions"; // true|false
    public static final String LOG_ACTIONS_MAX_S = "logActionsMax"; // max lines per step
    public static final String DELIVERY_RELAY_BONUS_S = "deliveryRelayBonus"; // delivered counted as extra relayed units

    private final String endpoint;
    private final int windowSizeSeconds;
    private final String contactsNormMode;
    private final double contactsCmax;
    private final double contactsTau;
    private final int maxMsgsPerNode;
    private final double deltaLimit;
    private final int timeoutMs;
    private String lastPolicyId = "";
    private final boolean logActions;
    private final int logActionsMax;
    private final boolean unboundedDelta;
    private final double deliveryRelayBonus;

    // Histories for windowed features
    private final Map<Integer, Deque<Double>> contactsHistory = new HashMap<Integer, Deque<Double>>();
    private final Map<Integer, Deque<Double>> freeFracHistory = new HashMap<Integer, Deque<Double>>();

    // Host-dest updated keys since last sample: hostStr -> set of destStr
    private final Map<String, Set<String>> updatedKeysByHost = new HashMap<String, Set<String>>();

    // Reward counters since last sample (host-dest granularity), key: "hostStr#destStr"
    private final Map<String, Integer> relayedByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> droppedByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> abortedByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> deliveredByKey = new HashMap<String, Integer>();
    private final Map<String, Double> totalDelayByKey = new HashMap<String, Double>();  // total delay for delivered messages

    public RLBridgeReport() {
        super();
        final Settings s = getSettings();
        this.endpoint = s.getSetting(URL_S, "");
        int w = (int)Math.round(s.getDouble(WINDOW_SIZE_S, 600.0));
        if (w <= 0) { w = 600; }
        this.windowSizeSeconds = w;
        this.contactsNormMode = s.getSetting(CONTACTS_NORM_MODE_S, "cmax").toLowerCase();
        this.contactsCmax = s.getDouble(CONTACTS_CMAX_S, 10.0);
        this.contactsTau = s.getDouble(CONTACTS_TAU_S, 3.0);
        this.maxMsgsPerNode = (int)Math.round(s.getDouble(MAX_MSG_PER_NODE_S, 20.0));
        this.deltaLimit = s.getDouble(DELTA_LIMIT_S, 0.05);
        this.timeoutMs = (int)Math.round(s.getDouble(TIMEOUT_MS_S, 1000.0));
        String logA = s.getSetting(LOG_ACTIONS_S, "false").toLowerCase();
        this.logActions = ("true".equals(logA) || "1".equals(logA) || "yes".equals(logA));
        this.logActionsMax = (int)Math.round(s.getDouble(LOG_ACTIONS_MAX_S, 10.0));
        String ub = s.getSetting(UNBOUNDED_DELTA_S, "false").toLowerCase();
        this.unboundedDelta = ("true".equals(ub) || "1".equals(ub) || "yes".equals(ub));
        this.deliveryRelayBonus = s.getDouble(DELIVERY_RELAY_BONUS_S, 3.0);

        write("# RLBridge active. endpoint=" + (endpoint.length()>0?endpoint:"(none)") +
                " sampleInterval=" + format(super.interval) + " windowSize=" + windowSizeSeconds);
    }

    @Override
    protected void sample(List<DTNHost> hosts) {
        final int windowSamples = Math.max(1, (int)Math.round(this.windowSizeSeconds / super.interval));
        final int now = (int) SimClock.getTime();

        // Build request payload
        StringBuilder req = new StringBuilder();
        req.append("{\"sim_id\":\"").append(escape(getScenarioName())).append("\",");
        req.append("\"time\":").append(now).append(",");
        // If unbounded, signal with negative delta_limit
        double effDeltaLimit = this.unboundedDelta ? -1.0 : this.deltaLimit;
        req.append("\"delta_limit\":").append(format(effDeltaLimit)).append(",");
        req.append("\"prev_transition\":[");
        boolean firstPrev = true;

        // Current state batch
        StringBuilder stateBatch = new StringBuilder();
        stateBatch.append("\"state_batch\":[");
        boolean firstState = true;
        int stateCount = 0;

        // Clear previous external offsets before applying new ones
        for (DTNHost h : hosts) {
            MessageRouter r = h.getRouter();
            if (r instanceof ProphetRouter) {
                ((ProphetRouter) r).clearExternalOffsets();
            }
        }

        for (DTNHost h : hosts) {
            final int addr = h.getAddress();
            final String hostStr = h.toString();

            // Build peers set
            final Set<Integer> peers = new HashSet<Integer>();
            for (Connection c : h.getConnections()) {
                peers.add(c.getOtherNode(h).getAddress());
            }

            // contacts_now
            final double contactsNow = peers.size();
            // free fraction now across {self+peers}
            double sumFrac = 0.0; int fracCount = 0;
            // self
            final MessageRouter selfRouter = h.getRouter();
            if (selfRouter != null) {
                long size = selfRouter.getBufferSize();
                long free = selfRouter.getFreeBufferSize();
                if (size > 0 && size < Integer.MAX_VALUE) {
                    double frac = (free * 1.0) / size; if (frac<0) frac=0; if (frac>1) frac=1;
                    sumFrac += frac; fracCount++;
                }
            }
            // peers
            for (Integer pid : peers) {
                DTNHost ph = null;
                for (DTNHost cand : hosts) { if (cand.getAddress()==pid) { ph=cand; break; } }
                if (ph != null && ph.getRouter()!=null) {
                    long size = ph.getRouter().getBufferSize();
                    long free = ph.getRouter().getFreeBufferSize();
                    if (size > 0 && size < Integer.MAX_VALUE) {
                        double frac = (free * 1.0) / size; if (frac<0) frac=0; if (frac>1) frac=1;
                        sumFrac += frac; fracCount++;
                    }
                }
            }
            final double freeFracNow = (fracCount>0)?(sumFrac/fracCount):Double.NaN;

            // Update histories
            Deque<Double> ch = contactsHistory.get(addr);
            if (ch == null) { ch = new ArrayDeque<Double>(windowSamples); contactsHistory.put(addr, ch); }
            Deque<Double> fh = freeFracHistory.get(addr);
            if (fh == null) { fh = new ArrayDeque<Double>(windowSamples); freeFracHistory.put(addr, fh); }
            if (ch.size()==windowSamples) ch.removeFirst();
            if (fh.size()==windowSamples) fh.removeFirst();
            ch.addLast(contactsNow); fh.addLast(freeFracNow);
            if (ch.size()<windowSamples || fh.size()<windowSamples) { continue; }

            double sumC=0.0, sumF=0.0; for(double v:ch) sumC+=v; for(double v:fh) sumF+=v;
            double avgC = sumC / windowSamples; double freebufNorm = sumF / windowSamples;
            double contactsNorm;
            if ("saturate".equals(this.contactsNormMode)) {
                double tau = (this.contactsTau>0?this.contactsTau:3.0);
                contactsNorm = 1.0 - Math.exp(-avgC/tau);
            } else {
                double cmax = (this.contactsCmax>0?this.contactsCmax:10.0);
                contactsNorm = avgC / cmax; if (contactsNorm<0) contactsNorm=0; if (contactsNorm>1) contactsNorm=1;
            }

            // Prev transition: drain counters for this host at host-dest granularity
            Set<String> destsUpdated = updatedKeysByHost.get(hostStr);
            if (destsUpdated != null && !destsUpdated.isEmpty()) {
                // copy to avoid concurrent modification
                java.util.List<String> copy = new java.util.ArrayList<String>(destsUpdated);
                for (String destStr : copy) {
                    String k = hostStr + "#" + destStr;
                    int rel = getAndResetStr(relayedByKey, k);
                    int dr = getAndResetStr(droppedByKey, k);
                    int ab = getAndResetStr(abortedByKey, k);
                    int de = getAndResetStr(deliveredByKey, k);
                    double totalDelay = getAndResetDouble(totalDelayByKey, k);
                    double avgDelay = (de > 0) ? (totalDelay / de) : 0.0;
                    int relAug = rel + (int)Math.round(this.deliveryRelayBonus * de);
                    if (rel != 0 || dr != 0 || ab != 0 || de != 0) {
                        if (!firstPrev) req.append(","); firstPrev=false;
                        req.append("{\"host\":\"").append(escape(hostStr)).append("\",")
                           .append("\"dest\":\"").append(escape(destStr)).append("\",")
                           .append("\"relayed\":").append(relAug).append(",\"drops\":").append(dr)
                           .append(",\"aborted\":").append(ab).append(",\"delivered\":").append(de)
                           .append(",\"avg_delay\":").append(String.format("%.3f", avgDelay)).append("}");
                    }
                    destsUpdated.remove(destStr);
                }
                if (destsUpdated.isEmpty()) { updatedKeysByHost.remove(hostStr); }
            }

            // State batch per destination (unique dests among buffered messages)
            int emitted=0;
            Set<String> uniqueDests = new HashSet<String>();
            for (Message m : h.getMessageCollection()) { uniqueDests.add(m.getTo().toString()); }
            for (String destStr : uniqueDests) {
                if (this.maxMsgsPerNode>0 && emitted>=this.maxMsgsPerNode) break;
                // resolve dest host
                DTNHost destHost = null; for (DTNHost cand : hosts) { if (cand.toString().equals(destStr)) { destHost = cand; break; } }
                if (destHost == null) { continue; }
                double pred = getPredFor(h, destHost);
                if (!firstState) stateBatch.append(","); firstState=false;
                stateBatch.append("{\"host\":\"").append(escape(hostStr)).append("\",")
                        .append("\"dest\":\"").append(escape(destStr)).append("\",")
                        .append("\"contacts_norm\":").append(format(contactsNorm)).append(",")
                        .append("\"freebuf_norm\":").append(format(freebufNorm)).append(",")
                        .append("\"pred\":").append(format(pred)).append("}");
                emitted++; stateCount++;
            }
        }

        req.append("],");
        req.append(stateBatch.toString()).append("]}");

        // Call DRL module and apply actions
        Map<String, Double> actions = callDrl(endpoint, req.toString());
        if (actions != null) {
            // actions map key: host#dest, value: delta
            int applied = 0;
            int recvActions = actions.size();
            int logged = 0;
            for (DTNHost h : hosts) {
                final String hostStr = h.toString();
                // Aggregate unique destinations
                Set<String> uniqueDests = new HashSet<String>();
                Map<String, DTNHost> destMap = new HashMap<String, DTNHost>();
                for (Message m : h.getMessageCollection()) { String d = m.getTo().toString(); uniqueDests.add(d); if (!destMap.containsKey(d)) destMap.put(d, m.getTo()); }
                for (String destStr : uniqueDests) {
                    String key = hostStr + "#" + destStr;
                    if (!actions.containsKey(key)) continue;
                    double delta = actions.get(key).doubleValue();
                    if (!this.unboundedDelta) { if (delta > deltaLimit) delta = deltaLimit; if (delta < -deltaLimit) delta = -deltaLimit; }
                    MessageRouter r = h.getRouter();
                    if (r instanceof ProphetRouter) {
                        DTNHost destHost = destMap.get(destStr);
                        if (destHost != null) {
                            ((ProphetRouter) r).setExternalOffset(destHost, delta);
                            applied++;
                            if (this.logActions && logged < this.logActionsMax) { write(now + " A " + hostStr + " - " + destStr + " " + format(delta)); logged++; }
                        }
                    }
                }
            }
            write("# RLBridge OK t=" + now + " policy=" + (lastPolicyId==null?"":lastPolicyId) +
                    " states=" + stateCount + " recv_actions=" + recvActions + " applied_actions=" + applied);
        }
        else {
            write("# RLBridge: no endpoint/failed request; policy=(none) states=" + stateCount + " recv_actions=0 applied_actions=0");
        }
    }

    private String escape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private int getAndReset(Map<Integer, Integer> map, int key) {
        Integer v = map.get(key); if (v == null) v = 0; map.put(key, 0); return v;
    }
    private int getAndResetStr(Map<String, Integer> map, String key) {
        Integer v = map.get(key); if (v == null) v = 0; map.put(key, 0); return v;
    }

    private double getAndResetDouble(Map<String, Double> map, String key) {
        Double v = map.get(key); if (v == null) v = 0.0; map.put(key, 0.0); return v;
    }

    private double getPredFor(DTNHost self, DTNHost dest) {
        MessageRouter r = self.getRouter();
        if (r instanceof ProphetRouter) return ((ProphetRouter) r).getPredFor(dest);
        if (r instanceof ProphetV2Router) return ((ProphetV2Router) r).getPredFor(dest);
        if (r instanceof ProphetRouterWithEstimation) return ((ProphetRouterWithEstimation) r).getPredFor(dest);
        return Double.NaN;
    }

    private Map<String, Double> callDrl(String urlStr, String payload) {
        if (urlStr == null || urlStr.trim().length() == 0) { return null; }
        HttpURLConnection con = null;
        try {
            URL url = new URL(urlStr);
            con = (HttpURLConnection) url.openConnection();
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type", "application/json");
            con.setConnectTimeout(timeoutMs);
            con.setReadTimeout(timeoutMs);
            con.setDoOutput(true);
            DataOutputStream out = new DataOutputStream(con.getOutputStream());
            out.write(payload.getBytes("UTF-8"));
            out.flush(); out.close();
            int code = con.getResponseCode();
            if (code != 200) { return null; }
            BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
            StringBuilder resp = new StringBuilder();
            String line; while ((line = in.readLine()) != null) { resp.append(line); }
            in.close();
            String body = resp.toString();
            try { this.lastPolicyId = extractString(body, "policy_id"); } catch (Exception ignore) {}
            Map<String, Double> parsed = parseActionsKV(body);
            if (parsed.isEmpty()) {
                parsed = parseActions(body);
            }
            if (parsed.isEmpty()) {
                // Emit a short debug snippet to help diagnose parsing issues
                String snippet;
                if (body.length() > 240) { snippet = body.substring(0, 240) + "..."; } else { snippet = body; }
                write("# RLBridge WARN: parsed zero actions; body_snippet=" + snippet.replace('\n',' ').replace('\r',' '));
            }
            return parsed;
        } catch (Exception e) {
            write("# RLBridge HTTP error: " + e.getMessage());
            return null;
        } finally {
            if (con != null) try { con.disconnect(); } catch (Exception ignore) {}
        }
    }

    // Simpler parsing for flat map: {"actions_kv": {"p0#H1": 0.01, "p1#H2": -0.003}}
    private Map<String, Double> parseActionsKV(String json) {
        Map<String, Double> result = new HashMap<String, Double>();
        if (json == null) return result;
        int idx = json.indexOf("\"actions_kv\""); if (idx < 0) return result;
        int objStart = json.indexOf('{', idx); if (objStart < 0) return result;
        int objEnd = findMatchingBracket(json, objStart, '{', '}'); if (objEnd < 0) return result;
        String body = json.substring(objStart + 1, objEnd);
        int i = 0;
        while (i < body.length()) {
            // find key
            int k1 = body.indexOf('"', i); if (k1 < 0) break;
            int k2 = body.indexOf('"', k1+1); if (k2 < 0) break;
            String key = body.substring(k1+1, k2);
            int colon = body.indexOf(':', k2); if (colon < 0) break;
            // parse value number
            int e = colon+1;
            while (e < body.length() && (Character.isWhitespace(body.charAt(e)) || body.charAt(e)==',')) e++;
            int j = e;
            while (j < body.length()) {
                char ch = body.charAt(j);
                if ((ch>='0' && ch<='9') || ch=='-' || ch=='.' || ch=='e' || ch=='E' || ch=='+') { j++; }
                else { break; }
            }
            try {
                Double val = Double.parseDouble(body.substring(e, j));
                result.put(key, val);
            } catch (Exception ignore) {}
            i = j+1;
        }
        return result;
    }

    // Minimal JSON parser for nested structure: {"actions":[{"host":"p1","per_message":[{"msg_id":"H1","delta":0.01}, ...]} ...]}
    private Map<String, Double> parseActions(String json) {
        Map<String, Double> result = new HashMap<String, Double>();
        if (json == null) return result;
        int idx = json.indexOf("\"actions\""); if (idx < 0) return result;
        int arrStart = json.indexOf('[', idx); if (arrStart < 0) return result;
        int arrEnd = findMatchingBracket(json, arrStart, '[', ']'); if (arrEnd < 0) return result;
        String arr = json.substring(arrStart + 1, arrEnd);
        // Extract top-level objects inside actions array
        java.util.List<String> hostObjs = extractTopLevelObjects(arr);
        for (String hb : hostObjs) {
            String host = extractString(hb, "host"); if (host == null) continue;
            int pmIdx = hb.indexOf("per_message"); if (pmIdx < 0) continue;
            int pmStart = hb.indexOf('[', pmIdx); if (pmStart < 0) continue;
            int pmEnd = findMatchingBracket(hb, pmStart, '[', ']'); if (pmEnd < 0) continue;
            String pmArr = hb.substring(pmStart + 1, pmEnd);
            for (String ent : extractTopLevelObjects(pmArr)) {
                String msgId = extractString(ent, "msg_id");
                Double delta = extractDouble(ent, "delta");
                if (msgId != null && delta != null) {
                    result.put(host + "#" + msgId, delta);
                }
            }
        }
        return result;
    }

    private int findMatchingBracket(String s, int start, char open, char close) {
        int depth = 0;
        for (int i = start; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == open) depth++;
            else if (ch == close) {
                depth--;
                if (depth == 0) return i;
            }
        }
        return -1;
    }

    private java.util.List<String> extractTopLevelObjects(String s) {
        java.util.List<String> out = new java.util.ArrayList<String>();
        int i = 0;
        while (i < s.length()) {
            // skip whitespace and commas
            while (i < s.length()) {
                char ch = s.charAt(i);
                if (Character.isWhitespace(ch) || ch == ',') i++; else break;
            }
            if (i >= s.length()) break;
            if (s.charAt(i) != '{') { i++; continue; }
            int end = findMatchingBracket(s, i, '{', '}');
            if (end < 0) break;
            out.add(s.substring(i, end + 1));
            i = end + 1;
        }
        return out;
    }

    private String extractString(String block, String key) {
        String k = "\"" + key + "\"";
        int i = block.indexOf(k); if (i < 0) return null;
        int q1 = block.indexOf('"', i + k.length()); if (q1 < 0) return null;
        int q2 = block.indexOf('"', q1+1); if (q2 < 0) return null;
        return block.substring(q1+1, q2);
    }

    private Double extractDouble(String block, String key) {
        String k = "\"" + key + "\"";
        int i = block.indexOf(k); if (i < 0) return null;
        int c = block.indexOf(':', i + k.length()); if (c < 0) return null;
        int e = c+1;
        while (e < block.length() && (Character.isDigit(block.charAt(e)) || block.charAt(e)=='.' || block.charAt(e)=='-' )) e++;
        try { return Double.parseDouble(block.substring(c+1, e)); } catch (Exception ex) { return null; }
    }

    // MessageListener-like hooks to collect rewards at host-dest granularity
    public void newMessage(Message m) {}
    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) {}
    public void messageDeleted(Message m, DTNHost where, boolean dropped) {
        if (dropped) {
            String hostStr = where.toString();
            String destStr = m.getTo().toString();
            String key = hostStr + "#" + destStr;
            Integer v = droppedByKey.get(key); droppedByKey.put(key, (v==null?1:v+1));
            markUpdated(hostStr, destStr);
        }
    }
    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) {
        String hostStr = from.toString();
        String destStr = m.getTo().toString();
        String key = hostStr + "#" + destStr;
        Integer v = abortedByKey.get(key); abortedByKey.put(key, (v==null?1:v+1));
        markUpdated(hostStr, destStr);
    }
    public void messageTransferred(Message m, DTNHost from, DTNHost to, boolean firstDelivery) {
        String hostStr = from.toString();
        String destStr = m.getTo().toString();
        String key = hostStr + "#" + destStr;
        Integer vr = relayedByKey.get(key); relayedByKey.put(key, (vr==null?1:vr+1));
        if (firstDelivery) { 
            Integer vd = deliveredByKey.get(key); deliveredByKey.put(key, (vd==null?1:vd+1)); 
            // Calculate delivery delay: current time - message creation time
            double deliveryDelay = SimClock.getTime() - m.getCreationTime();
            Double totalDelay = totalDelayByKey.get(key); 
            totalDelayByKey.put(key, (totalDelay == null ? deliveryDelay : totalDelay + deliveryDelay));
        }
        markUpdated(hostStr, destStr);
    }

    private void markUpdated(String hostStr, String destStr) {
        Set<String> s = updatedKeysByHost.get(hostStr);
        if (s == null) { s = new HashSet<String>(); updatedKeysByHost.put(hostStr, s); }
        s.add(destStr);
    }

}
