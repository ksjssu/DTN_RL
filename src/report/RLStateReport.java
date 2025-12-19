/*
 * RL State and Reward sampling report.
 * Emits, at fixed sampling intervals, per-node reward summary and per-message state tuples.
 * State per message m: [contacts_norm, pred(self->dest(m)), bufocc_mean, capacity_norm, self_buf_util].
 * Reward per node over the last interval (matching Python DRL server calculation):
 *   r = w_del(B)*delivered + [w_relay(B) + pressure_coeff(B,Δp)]*relayed + w_abort(B)*aborted
 * where B = buffer size (MB), Δp = self_buf_util - network_avg_buf (pressure difference).
 * Buffer-specific weights use sigmoid functions to adapt relay/abort penalties based on buffer capacity.
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

    // Reward counters since last sample
    private final Map<Integer, Integer> relayedCnt = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> droppedCnt = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> abortedCnt = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> deliveredCnt = new HashMap<Integer, Integer>(); // Track delivered per node

    // For episode-end delivery rate bonus summary
    private int createdTotal = 0;
    private int deliveredFinalTotal = 0;

    // Python DRL server reward constants (matching toolkit/drl_server.py)
    private static final double W_DELIVER_FALLBACK = 1.0;
    private static final double RELAY_MIN = -0.4;
    private static final double RELAY_MAX = 3.0;
    private static final double RELAY_SLOPE = 0.12;
    private static final double RELAY_MID = 40.0;
    private static final double ABORT_GAIN = 0.6;
    private static final double ABORT_SLOPE = 0.18;
    private static final double ABORT_MID = 25.0;
    private static final double NETWORK_AVG_LOW_THRESHOLD = 0.35;
    private static final double NETWORK_AVG_HIGH_THRESHOLD = 0.75;
    private static final double RELAY_CRIT_STATIC_PENALTY = 1.0;
    private static final double RELAY_CRIT_MULT = 2.0;
    private static final double H_WARNING_GAIN = 0.4;
    private static final double FLUSH_UTIL_THRESHOLD = 0.30;
    private static final double FLUSH_GAIN = 0.8;
    private static final double RATE_DELTA_GAIN = 10.0;

    private final double wRelay;
    private final double wDrop;
    private final double wAbort;
    private final String reportUrl;

    // Buffer occupancy knowledge (contact-based exchange)
    private final BufferOccupancyTracker bufOccTracker = new BufferOccupancyTracker();

    // CTDE: Track previous values for change rate calculation
    private double prevTotalMessages = 0.0;
    private double prevAvgBufferUtil = 0.0;
    private double prevTotalAborted = 0.0;
    private double prevTimestamp = 0.0;
    private double prevMsgChangeRate = 0.0;
    private double prevBufUtilChangeRate = 0.0;

    private double prevStepRate = Double.NaN;
    private double lastRateDelta = 0.0;

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

        write("# time type host msgId dest contacts_norm pred bufocc_mean capacity_norm self_buf_util relayed drops aborted reward legacy_reward global_active_conns global_total_msgs global_avg_buf global_avg_contacts global_msg_change_rate global_msg_change_momentum global_buf_util_change_rate global_buf_util_momentum global_avg_free_buf global_high_util_frac");
    }

    @Override
    protected void sample(final List<DTNHost> hosts) {
        final int windowSamples = Math.max(1, (int)Math.round(this.windowSizeSeconds / super.interval));
        final int now = (int) SimClock.getTime();

        double maxBufferSize = 0.0;
        if (hosts != null) {
            for (DTNHost h : hosts) {
                if (h == null) { continue; }
                MessageRouter router = h.getRouter();
                if (router != null) {
                    long bufSize = router.getBufferSize();
                    if (bufSize > maxBufferSize) {
                        maxBufferSize = bufSize;
                    }
                }
            }
        }
        if (maxBufferSize <= 0) {
            maxBufferSize = 1.0;
        }

        // CTDE: Collect global network features (10D)
        double[] globalFeatures = collectGlobalFeatures(hosts, maxBufferSize);

        // Update buffer occupancy tables (knowledge exchange) before sampling
        try { this.bufOccTracker.update(hosts, now, this.windowSizeSeconds); } catch (Exception ignore) {}

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

            final MessageRouter selfRouter = h.getRouter();
            double capacityNorm = 0.0;
            double selfBufUtil = Double.NaN;
            if (selfRouter != null) {
                if (maxBufferSize > 0) {
                    capacityNorm = ((double) selfRouter.getBufferSize()) / maxBufferSize;
                    if (capacityNorm < 0.0) { capacityNorm = 0.0; }
                    if (capacityNorm > 1.0) { capacityNorm = 1.0; }
                }
                long size = selfRouter.getBufferSize();
                long free = selfRouter.getFreeBufferSize();
                if (size > 0 && size < Integer.MAX_VALUE) {
                    double util = 1.0 - ((double) free / size);
                    if (util < 0.0) { util = 0.0; }
                    if (util > 1.0) { util = 1.0; }
                    selfBufUtil = util;
                }
            }

            Deque<Double> ch = contactsHistory.get(addr);
            if (ch == null) { ch = new ArrayDeque<Double>(windowSamples); contactsHistory.put(addr, ch); }
            if (ch.size() == windowSamples) { ch.removeFirst(); }
            ch.addLast(contactsNow);

            if (ch.size() < windowSamples) {
                continue;
            }

            double sumC = 0.0;
            for (double v : ch) { sumC += v; }
            final double avgC = sumC / windowSamples;

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
            int del = getAndReset(deliveredCnt, addr); // Get delivered count for this node

            // Calculate reward using same formula as Python DRL server (toolkit/drl_server.py)
            double reward = 0.0;

            // Get buffer size for this host (in MB)
            double bufferSizeMB = 60.0; // Default to 60M, will extract from router
            if (selfRouter != null) {
                long bufferSizeBytes = selfRouter.getBufferSize();
                bufferSizeMB = bufferSizeBytes / (1024.0 * 1024.0);
            }

            // Get buffer-specific weights
            double[] weights = bufferWeightProfile(bufferSizeMB);
            double wDel = weights[0];
            double wRelayWeight = weights[1];
            double wAbortWeight = weights[2];

            // 1. Delivered reward
            if (del > 0) {
                reward += wDel * del;
            }

            // 2. Aborted penalty
            if (ab > 0) {
                double abortPenalty = wAbortWeight * ab;
                reward += abortPenalty;
            }

            // 3. Relay reward with buffer-utilization-based threshold
            double bufOccMean = this.bufOccTracker.getMeanOccupancy(addr);

            if (rel > 0) {
                double relayReward = 0.0;
                double meanOcc = Double.isNaN(bufOccMean) ? 0.5 : bufOccMean;
                if (meanOcc <= NETWORK_AVG_LOW_THRESHOLD) {
                    relayReward = wRelayWeight * rel;
                } else if (meanOcc >= NETWORK_AVG_HIGH_THRESHOLD) {
                    relayReward = -RELAY_CRIT_MULT * wRelayWeight * rel;
                } else {
                    relayReward = 0.0;
                }

                double msgChangeRate = globalFeatures[4];
                if (msgChangeRate > 0.3) {
                    double warningScale = Math.min(1.0, Math.max(0.0, msgChangeRate));
                    relayReward -= H_WARNING_GAIN * warningScale * rel;
                }

                reward += relayReward;
            }

            if (!Double.isNaN(bufOccMean) && bufOccMean >= NETWORK_AVG_HIGH_THRESHOLD) {
                reward -= RELAY_CRIT_STATIC_PENALTY;
            }

            reward += RATE_DELTA_GAIN * this.lastRateDelta;

            double legacyReward = reward;
            double drlReward = RLBridgeReport.getLatestHostReward(hostStr, now);
            double finalReward = Double.isNaN(drlReward) ? legacyReward : drlReward;

            double g0 = safe(globalFeatures[0]);
            double g1 = safe(globalFeatures[1]);
            double g2 = safe(globalFeatures[2]);
            double g3 = safe(globalFeatures[3]);
            double g4 = safe(globalFeatures[4]);
            double g5 = safe(globalFeatures[5]);
            double g6 = safe(globalFeatures[6]);
            double g7 = safe(globalFeatures[7]);
            double g8 = safe(globalFeatures[8]);
            double g9 = safe(globalFeatures[9]);
            write(now + " R " + hostStr + " - - " + format(safe(contactsNorm)) + " NaN " + format(safe(bufOccMean)) + " " + format(safe(capacityNorm)) + " " + format(safe(selfBufUtil)) + " " + rel + " " + dr + " " + ab + " " + format(finalReward) + " " + format(legacyReward) + " " + format(g0) + " " + format(g1) + " " + format(g2) + " " + format(g3) + " " + format(g4) + " " + format(g5) + " " + format(g6) + " " + format(g7) + " " + format(g8) + " " + format(g9));

            // Per-message state lines (type S)
            int emitted = 0;
            for (Message m : h.getMessageCollection()) {
                if (this.maxMsgsPerNode > 0 && emitted >= this.maxMsgsPerNode) {
                    break;
                }
                // predictability
                double pred = getPredFor(h, m.getTo());
                double bufOccMean2 = this.bufOccTracker.getMeanOccupancy(addr);
                String destStr = m.getTo().toString();
                write(now + " S " + hostStr + " " + m.getId() + " " + destStr + " " + format(safe(contactsNorm)) + " " + format(safe(pred)) + " " + format(safe(bufOccMean2)) + " " + format(safe(capacityNorm)) + " " + format(safe(selfBufUtil)) + " 0 0 0 0.0000 0.0000 " + format(g0) + " " + format(g1) + " " + format(g2) + " " + format(g3) + " " + format(g4) + " " + format(g5) + " " + format(g6) + " " + format(g7) + " " + format(g8) + " " + format(g9));
                emitted++;
            }
        }

        double rateNow = (createdTotal > 0) ? ((double) deliveredFinalTotal) / createdTotal : Double.NaN;
        double rateDelta = 0.0;
        if (!Double.isNaN(rateNow) && !Double.isNaN(this.prevStepRate)) {
            rateDelta = rateNow - this.prevStepRate;
        }
        this.prevStepRate = rateNow;
        this.lastRateDelta = rateDelta;
        write("# step_cum_rate t=" + now + " rate=" + format(rateNow) +
                " delivered=" + deliveredFinalTotal + " created=" + createdTotal +
                " delta=" + format(rateDelta));
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
            // Track delivered per node (assign to sender of final hop)
            Integer vd = deliveredCnt.get(kFrom);
            deliveredCnt.put(kFrom, (vd == null ? 1 : vd + 1));
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

    /**
     * Calculate buffer-specific weight profile (matching Python buffer_weight_profile).
     * Returns [delivery_weight, relay_weight, abort_weight].
     */
    private double[] bufferWeightProfile(double bufferSizeMB) {
        // Sigmoid function for relay weight: RELAY_MIN + (RELAY_MAX - RELAY_MIN) * sigmoid(buffer_mb)
        double sigmoid = 1.0 / (1.0 + Math.exp(-RELAY_SLOPE * (bufferSizeMB - RELAY_MID)));
        double relayWeight = RELAY_MIN + (RELAY_MAX - RELAY_MIN) * sigmoid;
        relayWeight *= 2.0;
        if (relayWeight < 0.0) {
            relayWeight = Math.min(0.15, Math.abs(relayWeight)); // small positive reward for tiny buffers
        }

        // Abort weight calculation
        double abortWeight = -ABORT_GAIN / (1.0 + Math.exp(ABORT_SLOPE * (bufferSizeMB - ABORT_MID)));

        return new double[] { W_DELIVER_FALLBACK, relayWeight, abortWeight };
    }

    /**
     * CTDE: Collect global network state features (10 dimensions).
     * Same method as in RLBridgeReport for consistency.
     */
    private double[] collectGlobalFeatures(List<DTNHost> hosts, double maxBufferSize) {
        if (hosts == null || hosts.isEmpty()) {
            return new double[10]; // Return zeros if no hosts
        }

        double now = SimClock.getTime();
        // Statistics accumulators
        int totalNodes = hosts.size();
        int activeConnections = 0;
        int totalMessages = 0;
        double sumBufferUtil = 0.0;
        int validBufferCount = 0;
        double sumContacts = 0.0;
        int totalAborted = 0;
        int highUtilNodes = 0;

        // Collect statistics from all hosts
        for (DTNHost h : hosts) {
            // Count active connections
            if (h.getConnections() != null) {
                activeConnections += h.getConnections().size();
                sumContacts += h.getConnections().size();
            }

            // Count messages and buffer utilization
            MessageRouter router = h.getRouter();
            if (router != null) {
                java.util.Collection<Message> messages = router.getMessageCollection();
                if (messages != null) {
                    totalMessages += messages.size();
                }

                // Buffer utilization
                long size = router.getBufferSize();
                long free = router.getFreeBufferSize();
                if (size > 0 && size < Integer.MAX_VALUE) {
                    double util = 1.0 - ((double) free / size);
                    if (util < 0) util = 0;
                    if (util > 1) util = 1;
                    sumBufferUtil += util;
                    validBufferCount++;
                    if (util >= NETWORK_AVG_HIGH_THRESHOLD) {
                        highUtilNodes++;
                    }
                }
            }
        }

        // Sum aborted messages across all hosts
        for (Integer count : this.abortedCnt.values()) {
            totalAborted += count;
        }

        // Calculate statistics
        double avgBufferUtil = validBufferCount > 0 ? sumBufferUtil / validBufferCount : 0.0;

        double avgContacts = totalNodes > 0 ? sumContacts / totalNodes : 0.0;
        double highUtilFraction = totalNodes > 0 ? (double) highUtilNodes / totalNodes : 0.0;

        // Calculate change rates (trends)
        double timeDelta = (prevTimestamp > 0) ? (now - prevTimestamp) : 1.0;
        if (timeDelta <= 0) timeDelta = 1.0;

        double msgChangeRate = (totalMessages - prevTotalMessages) / timeDelta;
        double bufUtilChangeRate = (avgBufferUtil - prevAvgBufferUtil) / timeDelta;
        if (!Double.isFinite(msgChangeRate)) { msgChangeRate = 0.0; }
        if (!Double.isFinite(bufUtilChangeRate)) { bufUtilChangeRate = 0.0; }

        double msgChangeRateNorm = Math.max(-1.0, Math.min(1.0, msgChangeRate / 100.0));
        double bufUtilChangeRateNorm = Math.max(-1.0, Math.min(1.0, bufUtilChangeRate / 0.1));

        double msgChangeMomentum = 0.0;
        double denom = Math.abs(prevMsgChangeRate) > 1e-6 ? Math.abs(prevMsgChangeRate) : 1.0;
        msgChangeMomentum = Math.max(-1.0, Math.min(1.0, (msgChangeRate - prevMsgChangeRate) / denom));
        if (!Double.isFinite(msgChangeMomentum)) { msgChangeMomentum = 0.0; }

        double bufUtilMomentum = 0.0;
        double bufDenom = Math.abs(prevBufUtilChangeRate) > 1e-6 ? Math.abs(prevBufUtilChangeRate) : 1.0;
        bufUtilMomentum = Math.max(-1.0, Math.min(1.0, (bufUtilChangeRate - prevBufUtilChangeRate) / bufDenom));
        if (!Double.isFinite(bufUtilMomentum)) { bufUtilMomentum = 0.0; }

        prevTotalMessages = totalMessages;
        prevAvgBufferUtil = avgBufferUtil;
        prevTotalAborted = totalAborted;
        prevTimestamp = now;
        prevMsgChangeRate = msgChangeRate;
        prevBufUtilChangeRate = bufUtilChangeRate;

        double maxNodes = 1000.0;
        double maxConnections = 10000.0;
        double maxMessages = 10000.0;
        double maxContacts = 50.0;

        double[] globalFeatures = new double[10];
        globalFeatures[0] = activeConnections / maxConnections;             // active_connections_norm
        globalFeatures[1] = totalMessages / maxMessages;                    // total_messages_norm
        globalFeatures[2] = avgBufferUtil;                                  // avg_buffer_util
        globalFeatures[3] = avgContacts / maxContacts;                      // avg_contacts_norm
        globalFeatures[4] = msgChangeRateNorm;                              // msg_change_rate
        globalFeatures[5] = msgChangeMomentum;                              // msg_change_momentum
        globalFeatures[6] = bufUtilChangeRateNorm;                          // buf_util_change_rate
        globalFeatures[7] = bufUtilMomentum;                                // buf_util_momentum
        globalFeatures[8] = Math.max(0.0, 1.0 - avgBufferUtil);             // avg_free_buffer
        globalFeatures[9] = highUtilFraction;                               // high_util_fraction

        return globalFeatures;
    }

    private double safe(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return 0.0;
        }
        return value;
    }
}
