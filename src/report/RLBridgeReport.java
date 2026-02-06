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
    private static final double RELAY_CRIT_UTIL = 0.85;
    private static final double HIGH_UTIL_FLAG_THRESHOLD = 0.6;

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
    public static final String HEURISTIC_MODE_S = "heuristicMode";
    public static final String HEURISTIC_DELTA_S = "heuristicDelta";
    public static final String UNBOUNDED_DELTA_S = "unboundedDelta"; // true|false
    public static final String LOG_ACTIONS_S = "logActions"; // true|false
    public static final String LOG_ACTIONS_MAX_S = "logActionsMax"; // max lines per step
    public static final String BUF_OCC_MAX_AGE_S = "bufOccMaxAge"; // seconds to keep shared occupancy samples
    public static final String ACTIVATION_TIME_S = "activationTime"; // seconds before controller activates
    public static final String EPISODE_SECONDS_S = "episodeSeconds"; // episode duration for progress
    public static final String CREATED_RATE_MAX_PER_NODE_S = "createdRateMaxPerNodeSec"; // msgs/sec/node cap for normalization

    private final String endpoint;
    private final String localPolicyPath;
    private final int windowSizeSeconds;
    private final String contactsNormMode;
    private final double contactsCmax;
    private final double contactsTau;
    private final int maxMsgsPerNode;
    private final double deltaLimit;
    private final int timeoutMs;
    private final boolean heuristicMode;
    private final double heuristicDelta;
    private String lastPolicyId = "";
    private final boolean logActions;
    private final int logActionsMax;
    private final boolean unboundedDelta;
    private final double activationTime;
    private final double episodeSeconds;
    private final double messageTtlMinutes;
    private boolean activationNotified = false;
    private LocalPpoPolicy localPolicy = null; // feed-forward local policy
    private LocalRmappoPolicy localRmappoPolicy = null; // recurrent local policy
    private final BufferOccupancyTracker bufOccTracker = new BufferOccupancyTracker();
    private final int bufOccMaxAge;
    // Global message-creation tracking (since last sample)
    private int createdSinceLastSample = 0;
    private final double createdRateMaxPerNodeSec;

    // Histories for windowed features
    private final Map<Integer, Deque<Double>> contactsHistory = new HashMap<Integer, Deque<Double>>();

    // Host-dest updated keys since last sample: hostStr -> set of destStr
    private final Map<String, Set<String>> updatedKeysByHost = new HashMap<String, Set<String>>();

    // Reward counters since last sample (host-dest granularity), key: "hostStr#destStr"
    private final Map<String, Integer> relayedByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> droppedByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> abortedByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> deliveredByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> lowPredSkipByKey = new HashMap<String, Integer>();
    private final Map<String, Integer> highPredOppByKey = new HashMap<String, Integer>();
    private final Map<String, Double> totalDelayByKey = new HashMap<String, Double>();  // total delay for delivered messages
    private final Map<String, Double> ttlRatioSumByKey = new HashMap<String, Double>();
    private final Map<String, Double> hopCountSumByKey = new HashMap<String, Double>();
    private static final Map<String, RewardRecord> LAST_DRL_REWARD_BY_HOST = new HashMap<String, RewardRecord>();
    private static int LAST_REWARD_TIMESTAMP = -1;
    private static class RewardRecord {
        final double reward;
        final int timestamp;
        RewardRecord(double reward, int timestamp) {
            this.reward = reward;
            this.timestamp = timestamp;
        }
    }

    public static synchronized void recordHostRewards(Map<String, Double> hostRewards, int timestamp) {
        if (hostRewards == null) {
            return;
        }
        LAST_REWARD_TIMESTAMP = timestamp;
        for (Map.Entry<String, Double> e : hostRewards.entrySet()) {
            if (e.getKey() == null) {
                continue;
            }
            double val = (e.getValue() != null ? e.getValue() : 0.0);
            LAST_DRL_REWARD_BY_HOST.put(e.getKey(), new RewardRecord(val, timestamp));
        }
    }

    public static synchronized double getLatestHostReward(String host, int timestamp) {
        if (LAST_REWARD_TIMESTAMP != timestamp) {
            return Double.NaN;
        }
        RewardRecord rec = LAST_DRL_REWARD_BY_HOST.get(host);
        if (rec == null || rec.timestamp != timestamp) {
            return 0.0;
        }
        return rec.reward;
    }

    // Track last applied action (delta) for each host-dest pair for Gradient Alignment Reward
    private final Map<String, Double> lastActionByKey = new HashMap<String, Double>();
    // Track buffer utilization at the time of action
    private final Map<String, Double> lastBufferUtilByKey = new HashMap<String, Double>();
    // Track actual peer's p_base (on relay event) for neighbor-based reward
    private final Map<String, Double> lastPeerPBaseByKey = new HashMap<String, Double>();
    // Track max neighbor p_base across all nodes (at action application time) for optional max-based reward
    private final Map<String, Double> lastMaxNeighborPBaseByKey = new HashMap<String, Double>();

    // CTDE: Track previous values for change rate calculation
    private double prevTotalMessages = 0.0;
    private double prevAvgBufferUtil = 0.0;
    private double prevTotalAborted = 0.0;
    private double prevTimestamp = 0.0;
    private double prevMsgChangeRate = 0.0;
    private double prevBufUtilChangeRate = 0.0;
    private double prevGlobalFreeBuf = Double.NaN;
    private double currentRateDelta = 0.0;
    private double currentAvgBufDelta = 0.0;

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
        this.timeoutMs = (int)Math.round(s.getDouble(TIMEOUT_MS_S, 30000.0));
        this.heuristicMode = s.getBoolean(HEURISTIC_MODE_S, false);
        this.heuristicDelta = s.getDouble(HEURISTIC_DELTA_S, 0.05);
        String logA = s.getSetting(LOG_ACTIONS_S, "false").toLowerCase();
        this.logActions = ("true".equals(logA) || "1".equals(logA) || "yes".equals(logA));
        this.logActionsMax = (int)Math.round(s.getDouble(LOG_ACTIONS_MAX_S, 10.0));
        String ub = s.getSetting(UNBOUNDED_DELTA_S, "false").toLowerCase();
        this.unboundedDelta = ("true".equals(ub) || "1".equals(ub) || "yes".equals(ub));
        int age = (int)Math.round(s.getDouble(BUF_OCC_MAX_AGE_S, this.windowSizeSeconds));
        if (age < 0) { age = this.windowSizeSeconds; }
        this.bufOccMaxAge = age;
        this.createdRateMaxPerNodeSec = s.getDouble(CREATED_RATE_MAX_PER_NODE_S, 1.0);
        this.activationTime = Math.max(0.0, s.getDouble(ACTIVATION_TIME_S, 0.0));
        this.episodeSeconds = Math.max(1.0, s.getDouble(EPISODE_SECONDS_S, 100000.0));
        this.messageTtlMinutes = s.getDouble(MessageRouter.MSG_TTL_S, -1.0);
        if (this.activationTime > 0.0) {
            write("# RLBridge inactive until t >= " + format(this.activationTime));
        }

        write("# RLBridge active. endpoint=" + (endpoint.length()>0?endpoint:"(none)") +
                " sampleInterval=" + format(super.interval) + " windowSize=" + windowSizeSeconds);

        // Try load local policy for CTDE execution
        if (this.localPolicyPath.length() > 0) {
            try {
                if (LocalRmappoPolicy.isRmappoFormat(this.localPolicyPath)) {
                    this.localRmappoPolicy = new LocalRmappoPolicy(this.localPolicyPath);
                    write("# RLBridge local policy (rmappo_gru) loaded from " + this.localPolicyPath);
                } else {
                    this.localPolicy = new LocalPpoPolicy(this.localPolicyPath);
                    write("# RLBridge local policy loaded from " + this.localPolicyPath);
                }
            } catch (Exception e) {
                this.localPolicy = null;
                this.localRmappoPolicy = null;
                write("# RLBridge WARN: failed to load local policy '" + this.localPolicyPath + "': " + e.getMessage());
            }
        }
    }

    @Override
    protected void sample(List<DTNHost> hosts) {
        final int windowSamples = Math.max(1, (int)Math.round(this.windowSizeSeconds / super.interval));
        final int now = (int) SimClock.getTime();

        final boolean localGruMode = (this.localRmappoPolicy != null);
        final boolean localMlpMode = (this.localPolicy != null);
        final boolean baseLocalMode = localGruMode || localMlpMode || this.heuristicMode;
        final boolean controllerActive = (this.activationTime <= 0.0) || (now >= this.activationTime);
        if (controllerActive && !this.activationNotified && this.activationTime > 0.0) {
            write("# RLBridge activation at t=" + now);
            this.activationNotified = true;
        }
        final boolean localMode = controllerActive && baseLocalMode;
        final boolean remoteMode = controllerActive && !baseLocalMode;

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

        // Save created count before collectGlobalFeatures resets it
        int createdThisStep = this.createdSinceLastSample;

        // CTDE: Collect global network features once per step
        double[] globalFeatures = collectGlobalFeatures(hosts, maxBufferSize);

        // Start building request JSON
        if (remoteMode) {
            String simId = escape(getScenarioName());
            req.append("{\"sim_id\":\"").append(simId).append("\",\"time\":").append(now)
               .append(",\"delta_limit\":").append(effDeltaLimit)
               .append(",\"created_since_last\":").append(createdThisStep)
               .append(",\"prev_transition\":[");
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
            ingestLowPredSkips(h, hostStr);
            ingestHighPredOpportunities(h, hostStr);

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
            if (!Double.isFinite(capacityNorm)) { capacityNorm = 0.0; }
            if (!Double.isFinite(selfBufUtil)) { selfBufUtil = 0.0; }

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
            if (!Double.isFinite(contactsNorm)) { contactsNorm = 0.0; }

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
                    int lowPredSkips = getAndResetStr(lowPredSkipByKey, k);
                    int highPredOpps = getAndResetStr(highPredOppByKey, k);
                    double totalDelay = getAndResetDouble(totalDelayByKey, k);
                    double avgDelay = (de > 0) ? (totalDelay / de) : 0.0;
                    double ttlRatioSum = getAndResetDouble(ttlRatioSumByKey, k);
                    double hopCountSum = getAndResetDouble(hopCountSumByKey, k);
                    double avgTtlRatio = (de > 0) ? (ttlRatioSum / de) : 0.0;
                    double avgHops = (de > 0) ? (hopCountSum / de) : 0.0;

                    // Get Gradient Alignment Reward fields
                    double lastAction = getAndResetDouble(lastActionByKey, k);
                    double lastBufUtil = getAndResetDouble(lastBufferUtilByKey, k);
                    double peerNeighborPBase = getAndResetDouble(lastPeerPBaseByKey, k);
                    double maxNeighborPBase = getAndResetDouble(lastMaxNeighborPBaseByKey, k);

                    // Get p_base (pure PROPHET predictability without delta)
                    double pBase = 0.0;
                    DTNHost destHost = null;
                    for (DTNHost cand : hosts) {
                        if (cand.toString().equals(destStr)) {
                            destHost = cand;
                            break;
                        }
                    }
                    if (destHost != null) {
                        MessageRouter router = h.getRouter();
                        if (router instanceof ProphetRouter) {
                            pBase = ((ProphetRouter) router).getBasePredFor(destHost);
                        }
                    }

                    if (remoteMode) {
                        if (rel != 0 || dr != 0 || ab != 0 || de != 0 || lowPredSkips != 0 || highPredOpps != 0) {
                            if (!firstPrev) req.append(","); firstPrev=false;
                            req.append("{\"host\":\"").append(escape(hostStr)).append("\",")
                               .append("\"dest\":\"").append(escape(destStr)).append("\",")
                               .append("\"relayed\":").append(rel).append(",\"drops\":").append(dr)
                               .append(",\"aborted\":").append(ab).append(",\"delivered\":").append(de)
                               .append(",\"avg_delay\":").append(String.format("%.3f", avgDelay))
                                .append(",\"avg_ttl_ratio\":").append(format(avgTtlRatio))
                                .append(",\"avg_hops\":").append(format(avgHops))
                                .append(",\"skip_low_pred\":").append(lowPredSkips)
                                .append(",\"peer_high_pred\":").append(highPredOpps)
                                .append(",\"buffer_size\":").append(bufferCapacity).append(",")
                                // Gradient Alignment Reward fields
                                .append("\"p_base\":").append(format(pBase)).append(",")
                                .append("\"neighbor_p_base\":").append(format(peerNeighborPBase)).append(",")
                                .append("\"max_neighbor_p_base\":").append(format(maxNeighborPBase)).append(",")
                                .append("\"my_buffer_norm\":").append(format(lastBufUtil)).append(",")
                                .append("\"action\":").append(format(lastAction)).append("}");
                        }
                    } // if localMode or controller inactive, just drain counters without building JSON
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
                if (!Double.isFinite(pred) || !Double.isFinite(bufOccMean)) {
                    continue; // Only send complete state tuples to DRL
                }

                boolean appliedLocally = false;
                double delta = 0.0;
                if (controllerActive) {
                    if (localGruMode) {
                        double[] fullObs = new double[] {
                                contactsNorm, pred, bufOccMean, capacityNorm, selfBufUtil, selfBufUtil - bufOccMean, currentRateDelta
                        };
                        try {
                            delta = this.localRmappoPolicy.infer(hostStr + "#" + destStr, fullObs, effDeltaLimit);
                            appliedLocally = true;
                        } catch (Exception e) {
                            delta = 0.0;
                        }
                    } else if (localMlpMode) {
                        double[] obs = new double[] { contactsNorm, pred, bufOccMean, capacityNorm, selfBufUtil };
                        try {
                            delta = this.localPolicy.infer(obs, effDeltaLimit);
                            appliedLocally = true;
                        } catch (Exception e) {
                            delta = 0.0;
                        }
                    } else if (this.heuristicMode) {
                        double sign = (selfBufUtil - bufOccMean) <= 0 ? 1.0 : -1.0;
                        double base = (this.heuristicDelta > 0 ? this.heuristicDelta : effDeltaLimit);
                        delta = sign * Math.abs(base);
                        appliedLocally = true;
                    }
                }
                if (appliedLocally) {
                    if (!this.unboundedDelta) {
                        if (delta > deltaLimit) delta = deltaLimit; if (delta < -deltaLimit) delta = -deltaLimit;
                    }
                    MessageRouter r = h.getRouter();
                    if (r instanceof ProphetRouter) {
                        ((ProphetRouter) r).setExternalOffset(destHost, delta);

                        // Get max neighbor's p_base from ALL other nodes' Prophet tables (diagnostic/optional reward)
                        double maxNeighborPBase = 0.0;
                        if (destHost != null) {
                            for (DTNHost otherNode : hosts) {
                                if (otherNode == h) continue; // Skip self
                                MessageRouter otherRouter = otherNode.getRouter();
                                if (otherRouter instanceof ProphetRouter) {
                                    double otherPred = ((ProphetRouter) otherRouter).getBasePredFor(destHost);
                                    if (otherPred > maxNeighborPBase) {
                                        maxNeighborPBase = otherPred;  // Use maximum across all nodes
                                    }
                                }
                            }
                        }

                        // Store action, buffer util, and max_neighbor_p_base for reward logic
                        String k = hostStr + "#" + destStr;
                        lastActionByKey.put(k, delta);
                        lastBufferUtilByKey.put(k, selfBufUtil);
                        lastMaxNeighborPBaseByKey.put(k, maxNeighborPBase);

                        if (this.logActions && loggedLocal < this.logActionsMax) {
                            write(now + " A " + hostStr + " - " + destStr + " " + format(delta));
                            loggedLocal++;
                        }
                    }
                }
                if (remoteMode) {
                    if (!firstState) stateBatch.append(","); firstState=false;
                    double bufferSizeMB = bufferCapacity / (1024.0 * 1024.0);
                    if (!Double.isFinite(bufferSizeMB)) { bufferSizeMB = 0.0; }
                    double g0 = Double.isFinite(globalFeatures[0]) ? globalFeatures[0] : 0.0;
                    double g1 = Double.isFinite(globalFeatures[1]) ? globalFeatures[1] : 0.0;
                    double g2 = Double.isFinite(globalFeatures[2]) ? globalFeatures[2] : 0.0;
                    double g3 = Double.isFinite(globalFeatures[3]) ? globalFeatures[3] : 0.0;
                    double g4 = Double.isFinite(globalFeatures[4]) ? globalFeatures[4] : 0.0;
                    double g5 = Double.isFinite(globalFeatures[5]) ? globalFeatures[5] : 0.0;
                    double g6 = Double.isFinite(globalFeatures[6]) ? globalFeatures[6] : 0.0;
                    double g7 = Double.isFinite(globalFeatures[7]) ? globalFeatures[7] : 0.0;
                    double g8 = Double.isFinite(globalFeatures[8]) ? globalFeatures[8] : 0.0;
                    double g9 = Double.isFinite(globalFeatures[9]) ? globalFeatures[9] : 0.0;
                    double g10 = Double.isFinite(globalFeatures[10]) ? globalFeatures[10] : 0.0;
                    double g11 = Double.isFinite(globalFeatures[11]) ? globalFeatures[11] : 0.0;
                    double g12 = Double.isFinite(globalFeatures[12]) ? globalFeatures[12] : 0.0;
                    double g13 = (globalFeatures.length > 13 && Double.isFinite(globalFeatures[13])) ? globalFeatures[13] : 0.0;
                    stateBatch.append("{\"host\":\"").append(escape(hostStr)).append("\",")
                            .append("\"dest\":\"").append(escape(destStr)).append("\",")
                            .append("\"buffer_size_mb\":").append(String.format("%.2f", bufferSizeMB)).append(",")
                            // Local features (5D)
                            .append("\"contacts_norm\":").append(format(contactsNorm)).append(",")
                            .append("\"pred\":").append(format(pred)).append(",")
                            .append("\"bufocc_mean\":").append(format(bufOccMean)).append(",")
                            .append("\"capacity_norm\":").append(format(capacityNorm)).append(",")
                            .append("\"self_buf_util\":").append(format(selfBufUtil)).append(",")
                            .append("\"pressure_diff\":").append(format(selfBufUtil - bufOccMean)).append(",")
                            .append("\"rate_delta\":").append(format(currentRateDelta)).append(",")
                            // CTDE: Global features (10D)
                            .append("\"global_active_conns\":").append(format(g0)).append(",")
                            .append("\"global_total_msgs\":").append(format(g1)).append(",")
                            .append("\"global_avg_buf\":").append(format(g2)).append(",")
                            .append("\"global_avg_contacts\":").append(format(g3)).append(",")
                            .append("\"global_msg_change_rate\":").append(format(g4)).append(",")
                            .append("\"global_msg_change_momentum\":").append(format(g5)).append(",")
                            .append("\"global_buf_util_change_rate\":").append(format(g6)).append(",")
                            .append("\"global_buf_util_momentum\":").append(format(g7)).append(",")
                            .append("\"global_avg_free_buf\":").append(format(g8)).append(",")
                            .append("\"global_high_util_frac\":").append(format(g9)).append(",")
                            .append("\"global_high_util_flag\":").append(format(g10)).append(",")
                            .append("\"global_avg_buf_delta\":").append(format(g11)).append(",")
                            .append("\"global_episode_time_norm\":").append(format(g12)).append(",")
                            .append("\"global_created_rate\":").append(format(g13)).append("}");
                    emitted++; stateCount++;
                } else if (appliedLocally) {
                    emitted++; stateCount++;
                }
            }
        }
        }
        if (localMode) {
            // Local mode already applied actions inside the loop
            write("# RLBridge LOCAL t=" + now + " policy=(local) states=" + stateCount + " recv_actions=" + stateCount + " applied_actions=" + stateCount);
        } else if (remoteMode) {
            req.append("],");
            req.append(stateBatch.toString()).append("]}");

            // Call DRL module and apply actions
            ActionResponse response = callDrl(endpoint, req.toString());
            if (response != null && response.actions != null) {
                Map<String, Double> actions = response.actions;
                // actions map key: host#dest, value: delta
                int applied = 0;
                int recvActions = actions.size();
                int logged = 0;
                Map<String, Double> hostRewards = new HashMap<String, Double>();
                if (response.rewards != null) {
                    for (Map.Entry<String, Double> e : response.rewards.entrySet()) {
                        String key = e.getKey();
                        if (key == null) { continue; }
                        int hash = key.indexOf('#');
                        if (hash <= 0) { continue; }
                        String hostKey = key.substring(0, hash);
                        Double prev = hostRewards.get(hostKey);
                        double val = (e.getValue() != null ? e.getValue() : 0.0);
                        hostRewards.put(hostKey, (prev == null ? val : prev + val));
                    }
                }
                recordHostRewards(hostRewards, now);
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

                                // Calculate buffer utilization for Gradient Alignment Reward
                                double bufUtil = 0.0;
                                if (r != null) {
                                    long size = r.getBufferSize();
                                    long free = r.getFreeBufferSize();
                                    if (size > 0 && size < Integer.MAX_VALUE) {
                                        double util = 1.0 - ((double) free / size);
                                        if (util < 0) { util = 0; }
                                        if (util > 1) { util = 1; }
                                        bufUtil = util;
                                    }
                                }
                                if (!Double.isFinite(bufUtil)) { bufUtil = 0.0; }

                                // Get max neighbor's p_base from ALL other nodes' Prophet tables (diagnostic/optional reward)
                                double maxNeighborPBase = 0.0;
                                if (destHost != null) {
                                    for (DTNHost otherNode : hosts) {
                                        if (otherNode == h) continue; // Skip self
                                        MessageRouter otherRouter = otherNode.getRouter();
                                        if (otherRouter instanceof ProphetRouter) {
                                            double otherPred = ((ProphetRouter) otherRouter).getBasePredFor(destHost);
                                            if (otherPred > maxNeighborPBase) {
                                                maxNeighborPBase = otherPred;  // Use maximum across all nodes
                                            }
                                        }
                                    }
                                }

                                // Store action, buffer util, and max_neighbor_p_base for reward logic
                                lastActionByKey.put(key, delta);
                                lastBufferUtilByKey.put(key, bufUtil);
                                lastMaxNeighborPBaseByKey.put(key, maxNeighborPBase);

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

    private static class ActionResponse {
        final Map<String, Double> actions;
        final Map<String, Double> rewards;
        ActionResponse(Map<String, Double> actions, Map<String, Double> rewards) {
            this.actions = actions;
            this.rewards = rewards;
        }
    }

    private ActionResponse callDrl(String urlStr, String payload) {
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
            Map<String, Double> parsedActions = parseDoubleMap(body, "actions_kv");
            if (parsedActions.isEmpty()) {
                parsedActions = parseActions(body);
            }
            if (parsedActions.isEmpty()) {
                // Emit a short debug snippet to help diagnose parsing issues
                String snippet;
                if (body.length() > 240) { snippet = body.substring(0, 240) + "..."; } else { snippet = body; }
                write("# RLBridge WARN: parsed zero actions; body_snippet=" + snippet.replace('\n',' ').replace('\r',' '));
            }
            Map<String, Double> parsedRewards = parseDoubleMap(body, "rewards_kv");
            return new ActionResponse(parsedActions, parsedRewards);
        } catch (Exception e) {
            write("# RLBridge HTTP error: " + e.getMessage());
            return null;
        } finally {
            if (con != null) try { con.disconnect(); } catch (Exception ignore) {}
        }
    }

    // Simpler parsing for flat map: {"actions_kv": {"p0#H1": 0.01, "p1#H2": -0.003}}
    private Map<String, Double> parseDoubleMap(String json, String fieldName) {
        Map<String, Double> result = new HashMap<String, Double>();
        if (json == null) return result;
        String needle = "\"" + fieldName + "\"";
        int idx = json.indexOf(needle); if (idx < 0) return result;
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
    public void newMessage(Message m) { this.createdSinceLastSample++; }
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

        // Capture the actual peer's base predictability for neighbor-based reward
        try {
            DTNHost destHost = m.getTo();
            MessageRouter toRouter = to.getRouter();
            if (toRouter instanceof ProphetRouter && destHost != null) {
                double peerBase = ((ProphetRouter) toRouter).getBasePredFor(destHost);
                if (!Double.isFinite(peerBase)) { peerBase = 0.0; }
                lastPeerPBaseByKey.put(key, peerBase);
            }
        } catch (Exception ignore) { /* best effort only */ }
        if (firstDelivery) {
            Integer vd = deliveredByKey.get(key); deliveredByKey.put(key, (vd==null?1:vd+1));
            // Calculate delivery delay: current time - message creation time
            double deliveryDelay = SimClock.getTime() - m.getCreationTime();
            Double totalDelay = totalDelayByKey.get(key);
            totalDelayByKey.put(key, (totalDelay == null ? deliveryDelay : totalDelay + deliveryDelay));

            double ttlRatio = 1.0;
            if (this.messageTtlMinutes > 0.0) {
                double ttlRemaining = this.messageTtlMinutes - (deliveryDelay / 60.0);
                ttlRemaining = Math.max(0.0, ttlRemaining);
                ttlRatio = (ttlRemaining / this.messageTtlMinutes);
            }
            if (!Double.isFinite(ttlRatio) || ttlRatio < 0.0) { ttlRatio = 0.0; }
            if (ttlRatio > 1.0) { ttlRatio = 1.0; }
            Double ttlSum = ttlRatioSumByKey.get(key);
            ttlRatioSumByKey.put(key, ttlSum == null ? ttlRatio : ttlSum + ttlRatio);

            double hopCount = m.getHopCount();
            Double hopSum = hopCountSumByKey.get(key);
            hopCountSumByKey.put(key, hopSum == null ? hopCount : hopSum + hopCount);
        }
        markUpdated(hostStr, destStr);
    }

    private void markUpdated(String hostStr, String destStr) {
        Set<String> s = updatedKeysByHost.get(hostStr);
        if (s == null) { s = new HashSet<String>(); updatedKeysByHost.put(hostStr, s); }
        s.add(destStr);
    }

    private void ingestLowPredSkips(DTNHost host, String hostStr) {
        if (host == null) {
            return;
        }
        MessageRouter router = host.getRouter();
        if (!(router instanceof ProphetRouter)) {
            return;
        }
        Map<DTNHost, Integer> skips = ((ProphetRouter) router).drainLowPredSkips();
        if (skips == null || skips.isEmpty()) {
            return;
        }
        for (Map.Entry<DTNHost, Integer> e : skips.entrySet()) {
            DTNHost destHost = e.getKey();
            if (destHost == null) {
                continue;
            }
            int count = (e.getValue() != null) ? e.getValue().intValue() : 0;
            if (count <= 0) {
                continue;
            }
            String destStr = destHost.toString();
            String key = hostStr + "#" + destStr;
            Integer prev = lowPredSkipByKey.get(key);
            lowPredSkipByKey.put(key, (prev == null ? count : prev + count));
            markUpdated(hostStr, destStr);
        }
    }

    private void ingestHighPredOpportunities(DTNHost host, String hostStr) {
        if (host == null) {
            return;
        }
        MessageRouter router = host.getRouter();
        if (!(router instanceof ProphetRouter)) {
            return;
        }
        Map<DTNHost, Integer> opps = ((ProphetRouter) router).drainHighPredOpportunities();
        if (opps == null || opps.isEmpty()) {
            return;
        }
        for (Map.Entry<DTNHost, Integer> e : opps.entrySet()) {
            DTNHost destHost = e.getKey();
            if (destHost == null) {
                continue;
            }
            int count = (e.getValue() != null) ? e.getValue().intValue() : 0;
            if (count <= 0) {
                continue;
            }
            String destStr = destHost.toString();
            String key = hostStr + "#" + destStr;
            Integer prev = highPredOppByKey.get(key);
            highPredOppByKey.put(key, (prev == null ? count : prev + count));
            markUpdated(hostStr, destStr);
        }
    }

    /**
     * CTDE: Collect global network state features (base 12 + episode progress + created_rate).
     * These features provide network-wide context for the centralized critic.
     *
     * @param hosts List of all hosts in the network
     * @param maxBufferSize Maximum buffer size for normalization
     * @return Array of global features (length 14)
     */
    private double[] collectGlobalFeatures(List<DTNHost> hosts, double maxBufferSize) {
        if (hosts == null || hosts.isEmpty()) {
            return new double[14]; // Return zeros if no hosts
        }

        double now = SimClock.getTime();
        double simEndTime = 100000.0; // Default, will try to get from settings

        // Statistics accumulators
        int totalNodes = hosts.size();
        int activeConnections = 0;
        int totalMessages = 0;
        double sumBufferUtil = 0.0;
        int validBufferCount = 0;
        double sumContacts = 0.0;
        int totalRelayed = 0;
        int totalDelivered = 0;
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
                Collection<Message> messages = router.getMessageCollection();
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
                    if (util >= RELAY_CRIT_UTIL) {
                        highUtilNodes++;
                    }
                }
            }
        }

        // Count total relayed, delivered, and aborted from reward counters
        for (Integer count : relayedByKey.values()) {
            totalRelayed += count;
        }
        for (Integer count : deliveredByKey.values()) {
            totalDelivered += count;
        }
        for (Integer count : abortedByKey.values()) {
            totalAborted += count;
        }

        // Calculate statistics
        double avgBufferUtil = validBufferCount > 0 ? sumBufferUtil / validBufferCount : 0.0;
        double avgContacts = totalNodes > 0 ? sumContacts / totalNodes : 0.0;
        double highUtilFraction = totalNodes > 0 ? (double) highUtilNodes / totalNodes : 0.0;

        // Calculate change rates (trends)
        double timeDelta = (prevTimestamp > 0) ? (now - prevTimestamp) : 1.0; // Avoid division by zero
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

        // Normalization constants
        double maxNodes = 1000.0;
        double maxConnections = 10000.0;
        double maxMessages = 10000.0;
        double maxContacts = 50.0;

        // Construct 10-dimensional global feature vector
        double avgFreeBuf = Math.max(0.0, 1.0 - avgBufferUtil);
        double rateDelta = 0.0;
        if (!Double.isNaN(avgFreeBuf) && !Double.isNaN(prevGlobalFreeBuf)) {
            rateDelta = avgFreeBuf - prevGlobalFreeBuf;
        }
        prevGlobalFreeBuf = avgFreeBuf;
        currentRateDelta = rateDelta;

        double avgBufDelta = (this.prevTimestamp > 0) ? (avgBufferUtil - this.prevAvgBufferUtil) : 0.0;
        if (!Double.isFinite(avgBufDelta)) { avgBufDelta = 0.0; }
        this.currentAvgBufDelta = avgBufDelta;
        double highUtilFlag = (highUtilFraction >= HIGH_UTIL_FLAG_THRESHOLD) ? 1.0 : 0.0;

        double[] globalFeatures = new double[14];
        globalFeatures[0] = activeConnections / maxConnections;       // active_connections_norm
        globalFeatures[1] = totalMessages / maxMessages;              // total_messages_norm
        globalFeatures[2] = avgBufferUtil;                            // avg_buffer_util
        globalFeatures[3] = avgContacts / maxContacts;                // avg_contacts_norm
        globalFeatures[4] = msgChangeRateNorm;                        // msg_change_rate
        globalFeatures[5] = msgChangeMomentum;                        // msg_change_momentum
        globalFeatures[6] = bufUtilChangeRateNorm;                    // buf_util_change_rate
        globalFeatures[7] = bufUtilMomentum;                          // buf_util_momentum
        globalFeatures[8] = avgFreeBuf;                               // avg_free_buffer
        globalFeatures[9] = highUtilFraction;                         // high_util_fraction
        globalFeatures[10] = highUtilFlag;                            // high_util_flag
        globalFeatures[11] = avgBufDelta;                             // avg_buf_delta
        // Episode progress in [0,1]
        double ep = (this.episodeSeconds > 0.0 ? this.episodeSeconds : 100000.0);
        double tInEp = now % ep;
        double episodeProgress = (ep > 0.0 ? (tInEp / ep) : 0.0);
        if (!Double.isFinite(episodeProgress)) { episodeProgress = 0.0; }
        if (episodeProgress < 0.0) episodeProgress = 0.0; if (episodeProgress > 1.0) episodeProgress = 1.0;
        globalFeatures[12] = episodeProgress;                         // episode_time_norm

        // Created messages: per-second per-node, normalized by a configurable cap
        double createdRatePerNode = 0.0;
        if (timeDelta > 0.0 && hosts.size() > 0) {
            createdRatePerNode = (double)this.createdSinceLastSample / (timeDelta * (double)hosts.size());
        }
        // Reset counter for next window
        this.createdSinceLastSample = 0;
        double denomCreated = (this.createdRateMaxPerNodeSec > 1e-9 ? this.createdRateMaxPerNodeSec : 1.0);
        double createdNorm = createdRatePerNode / denomCreated;
        if (!Double.isFinite(createdNorm)) { createdNorm = 0.0; }
        if (createdNorm < 0.0) createdNorm = 0.0; if (createdNorm > 1.0) createdNorm = 1.0;
        globalFeatures[13] = createdNorm;                             // global_created_rate

        return globalFeatures;
    }

}
