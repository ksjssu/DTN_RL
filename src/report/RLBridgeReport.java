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
    public static final String LOCAL_POLICY_PATH_S = "localPolicyPath";
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
    public static final String BUF_OCC_MAX_AGE_S = "bufOccMaxAge"; // seconds to keep shared occupancy samples

    private final String endpoint;
    private final String localPolicyPath;
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
    private LocalPpoPolicy localPolicy = null; // when non-null, perform local inference (CTDE execution)
    private final BufferOccupancyTracker bufOccTracker = new BufferOccupancyTracker();
    private final int bufOccMaxAge;

    // Histories for windowed features
    private final Map<Integer, Deque<Double>> contactsHistory = new HashMap<Integer, Deque<Double>>();

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
        this.localPolicyPath = s.getSetting(LOCAL_POLICY_PATH_S, "").trim();
        int w = (int)Math.round(s.getDouble(WINDOW_SIZE_S, 600.0));
        if (w <= 0) { w = 600; }
        this.windowSizeSeconds = w;
        this.contactsNormMode = s.getSetting(CONTACTS_NORM_MODE_S, "cmax").toLowerCase();
        this.contactsCmax = s.getDouble(CONTACTS_CMAX_S, 10.0);
        this.contactsTau = s.getDouble(CONTACTS_TAU_S, 3.0);
        this.maxMsgsPerNode = (int)Math.round(s.getDouble(MAX_MSG_PER_NODE_S, 20.0));
        this.deltaLimit = s.getDouble(DELTA_LIMIT_S, 1.0);
        this.timeoutMs = (int)Math.round(s.getDouble(TIMEOUT_MS_S, 1000.0));
        String logA = s.getSetting(LOG_ACTIONS_S, "false").toLowerCase();
        this.logActions = ("true".equals(logA) || "1".equals(logA) || "yes".equals(logA));
        this.logActionsMax = (int)Math.round(s.getDouble(LOG_ACTIONS_MAX_S, 10.0));
        String ub = s.getSetting(UNBOUNDED_DELTA_S, "false").toLowerCase();
        this.unboundedDelta = ("true".equals(ub) || "1".equals(ub) || "yes".equals(ub));
        this.deliveryRelayBonus = s.getDouble(DELIVERY_RELAY_BONUS_S, 3.0);
        int age = (int)Math.round(s.getDouble(BUF_OCC_MAX_AGE_S, this.windowSizeSeconds));
        if (age < 0) { age = this.windowSizeSeconds; }
        this.bufOccMaxAge = age;

        write("# RLBridge active. endpoint=" + (endpoint.length()>0?endpoint:"(none)") +
                " sampleInterval=" + format(super.interval) + " windowSize=" + windowSizeSeconds);

        // Try load local policy for CTDE execution
        if (this.localPolicyPath.length() > 0) {
            try {
                this.localPolicy = new LocalPpoPolicy(this.localPolicyPath);
                write("# RLBridge local policy loaded from " + this.localPolicyPath);
            } catch (Exception e) {
                this.localPolicy = null;
                write("# RLBridge WARN: failed to load local policy '" + this.localPolicyPath + "': " + e.getMessage());
            }
        }
    }

    @Override
    protected void sample(List<DTNHost> hosts) {
        final int windowSamples = Math.max(1, (int)Math.round(this.windowSizeSeconds / super.interval));
        final int now = (int) SimClock.getTime();

        final boolean localMode = (this.localPolicy != null);

        // Initialize variables for state and transition building
        StringBuilder req = new StringBuilder();
        StringBuilder stateBatch = new StringBuilder();
        boolean firstPrev = true;
        boolean firstState = true;
        int stateCount = 0;
        int loggedLocal = 0;
        double effDeltaLimit = this.unboundedDelta ? -1.0 : this.deltaLimit;

        double maxBufferSize = 0.0;
        // First pass: calculate maxBufferSize
        if (hosts != null) {
            for (DTNHost h : hosts) {
                MessageRouter router = h.getRouter();
                if (router != null) {
                    maxBufferSize = Math.max(maxBufferSize, router.getBufferSize());
                }
            }
        }
        if (maxBufferSize <= 0.0) {
            maxBufferSize = 1.0;
        }

        // Update buffer occupancy tracker
        try {
            bufOccTracker.update(hosts, now, this.bufOccMaxAge);
        } catch (Exception ignore) { /* best effort */ }

        // Start building request JSON
        if (!localMode) {
            String simId = escape(getScenarioName());
            req.append("{\"sim_id\":\"").append(simId).append("\",\"time\":").append(now)
               .append(",\"delta_limit\":").append(effDeltaLimit).append(",\"prev_transition\":[");
            stateBatch.append("\"state_batch\":[");
        }

        // Clear previous external offsets before applying new ones
        if (hosts != null) {
            for (DTNHost h : hosts) {
                MessageRouter router = h.getRouter();
                if (router instanceof ProphetRouter) {
                    ((ProphetRouter) router).clearExternalOffsets();
                }
            }
        }

        if (hosts != null) {
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

            final MessageRouter selfRouter = h.getRouter();
            final long bufferCapacity = (selfRouter != null) ? selfRouter.getBufferSize() : 0;
            double capacityNorm = 0.0;
            double selfBufUtil = 0.0;
            if (selfRouter != null) {
                capacityNorm = ((double) selfRouter.getBufferSize()) / maxBufferSize;
                if (capacityNorm < 0) { capacityNorm = 0; }
                if (capacityNorm > 1) { capacityNorm = 1; }
                long size = selfRouter.getBufferSize();
                long free = selfRouter.getFreeBufferSize();
                if (size > 0 && size < Integer.MAX_VALUE) {
                    double util = 1.0 - ((double) free / size);
                    if (util < 0) { util = 0; }
                    if (util > 1) { util = 1; }
                    selfBufUtil = util;
                }
            }

            Deque<Double> ch = contactsHistory.get(addr);

            if (ch == null) { ch = new ArrayDeque<Double>(windowSamples); contactsHistory.put(addr, ch); }

            if (ch.size()==windowSamples) { ch.removeFirst(); }

            ch.addLast(contactsNow);

            if (ch.size()<windowSamples) { continue; }

            double sumC=0.0; for(double v:ch) sumC+=v;

            double avgC = sumC / windowSamples;

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
                    if (!localMode) {
                        if (rel != 0 || dr != 0 || ab != 0 || de != 0) {
                            if (!firstPrev) req.append(","); firstPrev=false;
                            req.append("{\"host\":\"").append(escape(hostStr)).append("\",")
                               .append("\"dest\":\"").append(escape(destStr)).append("\",")
                               .append("\"relayed\":").append(relAug).append(",\"drops\":").append(dr)
                               .append(",\"aborted\":").append(ab).append(",\"delivered\":").append(de)
                                .append(",\"avg_delay\":").append(String.format("%.3f", avgDelay))
                                .append(",\"buffer_size\":").append(bufferCapacity).append("}");
                        }
                    } // if localMode, just drain counters without building JSON
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
                double bufOccMean = this.bufOccTracker.getMeanOccupancy(addr);

                // Skip this host-dest pair if information is incomplete (NaN values)
                if (Double.isNaN(pred) || Double.isNaN(bufOccMean)) {
                    continue; // Only send complete state tuples to DRL
                }

                if (localMode) {
                    // Local inference (CTDE execution)
                    double[] obs = new double[] { contactsNorm, pred, bufOccMean, capacityNorm, selfBufUtil };
                    double delta = 0.0;
                    try {
                        delta = this.localPolicy.infer(obs, effDeltaLimit);
                    } catch (Exception e) {
                        delta = 0.0; // safe fallback
                    }
                    if (!this.unboundedDelta) {
                        if (delta > deltaLimit) delta = deltaLimit; if (delta < -deltaLimit) delta = -deltaLimit;
                    }
                    MessageRouter r = h.getRouter();
                    if (r instanceof ProphetRouter) {
                        ((ProphetRouter) r).setExternalOffset(destHost, delta);
                        if (this.logActions && loggedLocal < this.logActionsMax) { write(now + " A " + hostStr + " - " + destStr + " " + format(delta)); loggedLocal++; }
                    }
                    emitted++; stateCount++;
                } else {
                    if (!firstState) stateBatch.append(","); firstState=false;
                    stateBatch.append("{\"host\":\"").append(escape(hostStr)).append("\",")
                            .append("\"dest\":\"").append(escape(destStr)).append("\",")
                            .append("\"contacts_norm\":").append(format(contactsNorm)).append(",")
                                                        .append("\"pred\":").append(format(pred)).append(",")
                            .append("\"bufocc_mean\":").append(format(bufOccMean)).append(",")
                            .append("\"capacity_norm\":").append(format(capacityNorm)).append(",")
                            .append("\"self_buf_util\":").append(format(selfBufUtil)).append("}");
                    emitted++; stateCount++;
                }
            }
        }
        }
        if (localMode) {
            // Local mode already applied actions inside the loop
            write("# RLBridge LOCAL t=" + now + " policy=(local) states=" + stateCount + " recv_actions=" + stateCount + " applied_actions=" + stateCount);
        } else {
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
        // Skip leading whitespace
        while (e < block.length() && Character.isWhitespace(block.charAt(e))) e++;
        int j = e;
        // Allow digits, dot, minus, plus, and exponential notation (e/E)
        while (j < block.length()) {
            char ch = block.charAt(j);
            if ((ch>='0' && ch<='9') || ch=='-' || ch=='.' || ch=='e' || ch=='E' || ch=='+') { j++; }
            else { break; }
        }
        try { return Double.parseDouble(block.substring(e, j).trim()); } catch (Exception ex) { return null; }
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
