/*
 * RL State and Reward sampling report.
 * Emits, at fixed sampling intervals, per-node reward summary and per-message state tuples.
 * State per message m: [contacts_norm, freebuf_norm, pred(self->dest(m))].
 * Reward per node over the last interval: r = wDeliver*delivered - wDrop*dropped - wAbort*aborted.
 */
package report;

import core.DTNHost;
import core.Message;
import core.MessageListener;
import core.Settings;
import core.SimClock;
import core.Connection;
import routing.MessageRouter;
import routing.ProphetRouter;
import routing.ProphetRouterWithEstimation;
import routing.ProphetV2Router;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class RLStateReport extends SamplingReport implements MessageListener {

    // Settings
    public static final String WINDOW_SIZE_S = "windowSize"; // seconds
    public static final String CONTACTS_NORM_MODE_S = "contactsNormMode"; // cmax|saturate
    public static final String CONTACTS_CMAX_S = "contactsCmax"; // for cmax mode
    public static final String CONTACTS_TAU_S = "contactsTau";   // for saturate mode
    public static final String MAX_MSG_PER_NODE_S = "maxMessagesPerNode"; // 0 = unlimited

    public static final String W_RELAY_S = "wRelay";
    public static final String W_DROP_S = "wDrop";
    public static final String W_ABORT_S = "wAbort";
    public static final String REPORT_URL_S = "reportUrl"; // optional: POST episode-end summary to DRL server

    // Windowed stats
    private final int windowSizeSeconds;
    private final String contactsNormMode;
    private final double contactsCmax;
    private final double contactsTau;
    private final int maxMsgsPerNode;

    private final Map<Integer, Deque<Double>> contactsHistory = new HashMap<Integer, Deque<Double>>();
    private final Map<Integer, Deque<Double>> freeFracHistory = new HashMap<Integer, Deque<Double>>();

    // Reward counters since last sample
    private final Map<Integer, Integer> relayedCnt = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> droppedCnt = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> abortedCnt = new HashMap<Integer, Integer>();

    // For episode-end delivery rate bonus summary
    private int createdTotal = 0;
    private int deliveredFinalTotal = 0;

    private final double wRelay;
    private final double wDrop;
    private final double wAbort;
    private final String reportUrl;

    public RLStateReport() {
        super();
        final Settings s = getSettings();
        int w = (int)Math.round(s.getDouble(WINDOW_SIZE_S, 600.0));
        if (w <= 0) { w = 600; }
        this.windowSizeSeconds = w;
        this.contactsNormMode = s.getSetting(CONTACTS_NORM_MODE_S, "cmax").toLowerCase();
        this.contactsCmax = s.getDouble(CONTACTS_CMAX_S, 10.0);
        this.contactsTau = s.getDouble(CONTACTS_TAU_S, 3.0);
        this.maxMsgsPerNode = (int)Math.round(s.getDouble(MAX_MSG_PER_NODE_S, 20.0));

        this.wRelay = s.getDouble(W_RELAY_S, 1.0);
        this.wDrop = s.getDouble(W_DROP_S, 1.0);
        this.wAbort = s.getDouble(W_ABORT_S, 0.5);
        this.reportUrl = s.getSetting(REPORT_URL_S, "");

        write("# time type host msgId dest contacts_norm freebuf_norm pred relayed drops aborted reward");
    }

    @Override
    protected void sample(final List<DTNHost> hosts) {
        final int windowSamples = Math.max(1, (int)Math.round(this.windowSizeSeconds / super.interval));
        final int now = (int) SimClock.getTime();

        for (DTNHost h : hosts) {
            final String hostStr = h.toString();
            final int addr = h.getAddress();

            // Build peers set
            final Set<Integer> peers = new HashSet<Integer>();
            for (Connection c : h.getConnections()) {
                peers.add(c.getOtherNode(h).getAddress());
            }

            // contacts_now
            final double contactsNow = peers.size();

            // free fraction now across {self+peers}
            double sumFrac = 0.0;
            int fracCount = 0;

            // self
            final MessageRouter selfRouter = h.getRouter();
            if (selfRouter != null) {
                long size = selfRouter.getBufferSize();
                long free = selfRouter.getFreeBufferSize();
                if (size > 0 && size < Integer.MAX_VALUE) {
                    double frac = (free * 1.0) / size;
                    if (frac < 0.0) frac = 0.0;
                    if (frac > 1.0) frac = 1.0;
                    sumFrac += frac;
                    fracCount += 1;
                }
            }

            // peers
            for (Integer pid : peers) {
                DTNHost ph = null;
                for (DTNHost cand : hosts) {
                    if (cand.getAddress() == pid) { ph = cand; break; }
                }
                if (ph != null && ph.getRouter() != null) {
                    MessageRouter r = ph.getRouter();
                    long size = r.getBufferSize();
                    long free = r.getFreeBufferSize();
                    if (size > 0 && size < Integer.MAX_VALUE) {
                        double frac = (free * 1.0) / size;
                        if (frac < 0.0) frac = 0.0;
                        if (frac > 1.0) frac = 1.0;
                        sumFrac += frac;
                        fracCount += 1;
                    }
                }
            }

            final double freeFracNow = (fracCount > 0) ? (sumFrac / fracCount) : Double.NaN;

            // Update histories
            Deque<Double> ch = contactsHistory.get(addr);
            if (ch == null) { ch = new ArrayDeque<Double>(windowSamples); contactsHistory.put(addr, ch); }
            Deque<Double> fh = freeFracHistory.get(addr);
            if (fh == null) { fh = new ArrayDeque<Double>(windowSamples); freeFracHistory.put(addr, fh); }

            if (ch.size() == windowSamples) ch.removeFirst();
            if (fh.size() == windowSamples) fh.removeFirst();
            ch.addLast(contactsNow);
            fh.addLast(freeFracNow);

            // Need full window for normalized outputs
            if (ch.size() < windowSamples || fh.size() < windowSamples) {
                continue;
            }

            double sumC = 0.0;
            double sumFfrac = 0.0;
            for (double v : ch) sumC += v;
            for (double v : fh) sumFfrac += v;
            final double avgC = sumC / windowSamples;
            final double freebufNorm = sumFfrac / windowSamples; // [0,1]

            // contacts normalization
            double contactsNorm;
            if ("saturate".equals(this.contactsNormMode)) {
                double tau = (this.contactsTau > 0 ? this.contactsTau : 3.0);
                contactsNorm = 1.0 - Math.exp(-avgC / tau);
            } else {
                double cmax = (this.contactsCmax > 0 ? this.contactsCmax : 10.0);
                contactsNorm = avgC / cmax;
                if (contactsNorm > 1.0) contactsNorm = 1.0;
                if (contactsNorm < 0.0) contactsNorm = 0.0;
            }

            // Reward summary line (type R)
            int rel = getAndReset(relayedCnt, addr);
            int dr = getAndReset(droppedCnt, addr);
            int ab = getAndReset(abortedCnt, addr);
            double reward = this.wRelay * rel - this.wDrop * dr - this.wAbort * ab;
            write(now + " R " + hostStr + " - - " + format(contactsNorm) + " " + format(freebufNorm) + " NaN " + rel + " " + dr + " " + ab + " " + format(reward));

            // Per-message state lines (type S)
            int emitted = 0;
            for (Message m : h.getMessageCollection()) {
                if (this.maxMsgsPerNode > 0 && emitted >= this.maxMsgsPerNode) {
                    break;
                }
                // predictability
                double pred = getPredFor(h, m.getTo());
                String destStr = m.getTo().toString();
                write(now + " S " + hostStr + " " + m.getId() + " " + destStr + " " + format(contactsNorm) + " " + format(freebufNorm) + " " + format(pred) + " 0 0 0 0.0000");
                emitted++;
            }
        }

        // Step-level cumulative delivery rate logging (no episodes)
        double rateNow = (createdTotal > 0) ? ((double) deliveredFinalTotal) / createdTotal : Double.NaN;
        write("# step_cum_rate t=" + now + " rate=" + format(rateNow) +
                " delivered=" + deliveredFinalTotal + " created=" + createdTotal);
    }

    private int getAndReset(Map<Integer, Integer> map, int key) {
        Integer v = map.get(key);
        if (v == null) v = 0;
        map.put(key, 0);
        return v;
    }

    private double getPredFor(DTNHost self, DTNHost dest) {
        MessageRouter r = self.getRouter();
        if (r instanceof ProphetRouter) {
            return ((ProphetRouter) r).getPredFor(dest);
        } else if (r instanceof ProphetV2Router) {
            return ((ProphetV2Router) r).getPredFor(dest);
        } else if (r instanceof ProphetRouterWithEstimation) {
            return ((ProphetRouterWithEstimation) r).getPredFor(dest);
        } else {
            return Double.NaN;
        }
    }

    // MessageListener implementation for reward counters
    public void newMessage(Message m) { createdTotal++; }

    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) { /* not used */ }

    public void messageDeleted(Message m, DTNHost where, boolean dropped) {
        if (dropped) {
            int k = where.getAddress();
            Integer v = droppedCnt.get(k);
            droppedCnt.put(k, (v == null ? 1 : v + 1));
        }
    }

    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) {
        int k = from.getAddress();
        Integer v = abortedCnt.get(k);
        abortedCnt.put(k, (v == null ? 1 : v + 1));
    }

    public void messageTransferred(Message m, DTNHost from, DTNHost to, boolean firstDelivery) {
        // Count all successful transfers (hop-level relays) for reward
        int kFrom = from.getAddress();
        Integer vr = relayedCnt.get(kFrom);
        relayedCnt.put(kFrom, (vr == null ? 1 : vr + 1));

        // Track final deliveries for episode-end bonus summary
        if (firstDelivery) {
            deliveredFinalTotal++;
        }
    }

    @Override
    public void done() {
        // Emit episode-end delivery rate summary line
        double rate = (createdTotal > 0) ? ((double) deliveredFinalTotal) / createdTotal : Double.NaN;
        write("# episode_end average_delivery_rate " + format(rate) + " (delivered=" + deliveredFinalTotal + "/created=" + createdTotal + ")");
        // Also write a concise episode reward file for external consumers (e.g., DRL server dashboards)
        try {
            Settings sAll = new Settings();
            String dir = sAll.getSetting(Report.REPORTDIR_SETTING);
            if (dir == null || dir.length() == 0) { dir = "reports"; }
            java.io.File out = new java.io.File(dir, getScenarioName() + "_EpisodeReward.txt");
            java.io.PrintWriter pw = new java.io.PrintWriter(new java.io.OutputStreamWriter(new java.io.FileOutputStream(out, false), "UTF-8"));
            pw.println("average_delivery_rate " + format(rate) + " delivered " + deliveredFinalTotal + " created " + createdTotal);
            pw.close();
        } catch (Exception ignore) {}
        // Optional: notify DRL server
        if (this.reportUrl != null && this.reportUrl.trim().length() > 0) {
            try {
                java.net.URL url = new java.net.URL(this.reportUrl);
                java.net.HttpURLConnection con = (java.net.HttpURLConnection) url.openConnection();
                con.setRequestMethod("POST");
                con.setRequestProperty("Content-Type", "application/json");
                con.setConnectTimeout(1000);
                con.setReadTimeout(1000);
                con.setDoOutput(true);
                String payload = "{\"sim_id\":\"" + escape(getScenarioName()) + "\",\"average_delivery_rate\":" + format(rate) +
                        ",\"delivered\":" + deliveredFinalTotal + ",\"created\":" + createdTotal + "}";
                java.io.DataOutputStream out = new java.io.DataOutputStream(con.getOutputStream());
                out.write(payload.getBytes("UTF-8")); out.flush(); out.close();
                int code = con.getResponseCode();
                if (code != 200) {
                    write("# RLStateReport WARN: episode_end POST failed: HTTP " + code);
                }
                try { con.disconnect(); } catch (Exception ignore) {}
            } catch (Exception e) {
                write("# RLStateReport WARN: episode_end POST error: " + e.getMessage());
            }
        }
        super.done();
    }

    private String escape(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
