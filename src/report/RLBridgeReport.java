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

    private final String endpoint;
    private final int windowSizeSeconds;
    private final String contactsNormMode;
    private final double contactsCmax;
    private final double contactsTau;
    private final int maxMsgsPerNode;
    private final double deltaLimit;
    private final int timeoutMs;

    // Histories for windowed features
    private final Map<Integer, Deque<Double>> contactsHistory = new HashMap<Integer, Deque<Double>>();
    private final Map<Integer, Deque<Double>> freeFracHistory = new HashMap<Integer, Deque<Double>>();

    // Last state/action per message for prev_transition
    private static class MsgState {
        double cNorm; double fNorm; double pred; double action;
        MsgState(double c, double f, double p, double a){cNorm=c;fNorm=f;pred=p;action=a;}
    }
    private final Map<String, MsgState> lastMsgState = new HashMap<String, MsgState>(); // key: host#msgId

    // Reward counters since last sample (node-level)
    private final Map<Integer, Integer> relayedCnt = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> droppedCnt = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> abortedCnt = new HashMap<Integer, Integer>();

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
        req.append("\"delta_limit\":").append(format(this.deltaLimit)).append(",");
        req.append("\"prev_transition\":[");
        boolean firstPrev = true;

        // Current state batch
        StringBuilder stateBatch = new StringBuilder();
        stateBatch.append("\"state_batch\":[");
        boolean firstState = true;

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

            // Prev transition (node-level reward)
            int rel = getAndReset(relayedCnt, addr);
            int dr = getAndReset(droppedCnt, addr);
            int ab = getAndReset(abortedCnt, addr);
            if (!firstPrev) req.append(","); firstPrev=false;
            req.append("{\"host\":\"").append(escape(hostStr)).append("\",")
               .append("\"relayed\":").append(rel).append(",\"drops\":").append(dr)
               .append(",\"aborted\":").append(ab).append("}");

            // State batch per message
            int emitted=0;
            for (Message m : h.getMessageCollection()) {
                if (this.maxMsgsPerNode>0 && emitted>=this.maxMsgsPerNode) break;
                double pred = getPredFor(h, m.getTo());
                if (!firstState) stateBatch.append(","); firstState=false;
                stateBatch.append("{\"host\":\"").append(escape(hostStr)).append("\",")
                        .append("\"msg_id\":\"").append(escape(m.getId())).append("\",")
                        .append("\"dest\":\"").append(escape(m.getTo().toString())).append("\",")
                        .append("\"contacts_norm\":").append(format(contactsNorm)).append(",")
                        .append("\"freebuf_norm\":").append(format(freebufNorm)).append(",")
                        .append("\"pred\":").append(format(pred)).append("}");
                // Save last state for this message with action placeholder (filled after response)
                String key = hostStr + "#" + m.getId();
                MsgState prev = lastMsgState.get(key);
                if (prev == null) { lastMsgState.put(key, new MsgState(contactsNorm, freebufNorm, pred, 0.0)); }
                emitted++;
            }
        }

        req.append("],");
        req.append(stateBatch.toString()).append("]}");

        // Call DRL module and apply actions
        Map<String, Double> actions = callDrl(endpoint, req.toString());
        if (actions != null) {
            // actions map key: host#msg_id, value: delta
            int applied = 0;
            for (DTNHost h : hosts) {
                final String hostStr = h.toString();
                // apply per message
                for (Message m : h.getMessageCollection()) {
                    String key = hostStr + "#" + m.getId();
                    if (!actions.containsKey(key)) continue;
                    double delta = actions.get(key).doubleValue();
                    if (delta > deltaLimit) delta = deltaLimit;
                    if (delta < -deltaLimit) delta = -deltaLimit;
                    // Apply as external offset to destination on Prophet routers
                    MessageRouter r = h.getRouter();
                    if (r instanceof ProphetRouter) {
                        // We set offset equal to delta (additive). If multiple messages to same dest, last one wins.
                        ((ProphetRouter) r).setExternalOffset(m.getTo(), delta);
                        applied++;
                    }
                    // store last action
                    lastMsgState.put(key, new MsgState(0,0,0, delta));
                }
            }
            write("# RLBridge OK t=" + now + " applied_actions=" + applied);
        }
        else {
            write("# RLBridge: no endpoint/failed request; applied zero deltas");
        }
    }

    private String escape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private int getAndReset(Map<Integer, Integer> map, int key) {
        Integer v = map.get(key); if (v == null) v = 0; map.put(key, 0); return v;
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
            return parseActions(resp.toString());
        } catch (Exception e) {
            write("# RLBridge HTTP error: " + e.getMessage());
            return null;
        } finally {
            if (con != null) try { con.disconnect(); } catch (Exception ignore) {}
        }
    }

    // Minimal JSON parser for {"actions":[{"host":"p1","per_message":[{"msg_id":"H1","delta":0.01}, ...]} ...]}
    private Map<String, Double> parseActions(String json) {
        Map<String, Double> result = new HashMap<String, Double>();
        if (json == null) return result;
        // naive parsing: find "actions":[ ... ] and then extract host/msg_id/delta triples
        int idx = json.indexOf("\"actions\""); if (idx < 0) return result;
        int arrStart = json.indexOf('[', idx); if (arrStart < 0) return result;
        int arrEnd = json.indexOf(']', arrStart); if (arrEnd < 0) return result;
        String arr = json.substring(arrStart+1, arrEnd);
        String[] hostBlocks = arr.split("\\},\\{");
        for (String hb : hostBlocks) {
            String host = extractString(hb, "host"); if (host == null) continue;
            // per_message array
            int pmIdx = hb.indexOf("per_message"); if (pmIdx < 0) continue;
            int pmStart = hb.indexOf('[', pmIdx); if (pmStart < 0) continue;
            int pmEnd = hb.indexOf(']', pmStart); if (pmEnd < 0) continue;
            String pmArr = hb.substring(pmStart+1, pmEnd);
            String[] entries = pmArr.split("\\},\\{");
            for (String ent : entries) {
                String msgId = extractString(ent, "msg_id");
                Double delta = extractDouble(ent, "delta");
                if (msgId != null && delta != null) {
                    result.put(host + "#" + msgId, delta);
                }
            }
        }
        return result;
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

    // MessageListener-like hooks to collect rewards; wire via Settings by adding as Report and Simulation listeners already attach UpdateListener
    public void newMessage(Message m) {}
    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) {}
    public void messageDeleted(Message m, DTNHost where, boolean dropped) {
        if (dropped) {
            int k = where.getAddress();
            Integer v = droppedCnt.get(k); droppedCnt.put(k, (v==null?1:v+1));
        }
    }
    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) {
        int k = from.getAddress();
        Integer v = abortedCnt.get(k); abortedCnt.put(k, (v==null?1:v+1));
    }
    public void messageTransferred(Message m, DTNHost from, DTNHost to, boolean firstDelivery) {
        int k = from.getAddress();
        Integer v = relayedCnt.get(k); relayedCnt.put(k, (v==null?1:v+1));
    }

}
