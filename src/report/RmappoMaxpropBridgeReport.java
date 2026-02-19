/*
 * RMAPPO MaxProp++ Bridge report: synchronous per-step update with DRL module.
 * At each sample interval: send prev transitions + current state; receive per-host
 * MaxProp++ parameters and apply them to MaxPropRouter instances.
 */
package report;

import core.Connection;
import core.DTNHost;
import core.Message;
import core.Settings;
import core.SimClock;
import core.UpdateListener;
import routing.MaxPropRouter;
import routing.MessageRouter;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Bridge report for training/executing an R-MAPPO controller that tunes
 * MaxProp++ scalar parameters (not message selection).
 *
 * <p>Protocol (v1) request fields:
 * <ul>
 *   <li>protocol: "rmappo_maxprop_v1"</li>
 *   <li>sim_id, time, step_id, prev_time</li>
 *   <li>global_features: critic-only globals</li>
 *   <li>prev_transition: per-host counters (delivery/overhead/energy)</li>
 *   <li>state_batch: per-host observation vectors</li>
 * </ul>
 *
 * Response is expected to include an "actions" array:
 * <pre>
 * {"actions":[{"host":"p0","lambda_cost":...,"tau_age":...,"beta_x":...,"k_x":...,"m_relay":...}, ...]}
 * </pre>
 */
public class RmappoMaxpropBridgeReport extends SamplingReport implements UpdateListener, core.MessageListener {

    // Settings
    public static final String URL_S = "url";
    public static final String TIMEOUT_MS_S = "timeoutMs";
    public static final String ACTIVATION_TIME_S = "activationTime";
    public static final String EPISODE_SECONDS_S = "episodeSeconds";
    public static final String MAX_HOSTS_PER_STEP_S = "maxHostsPerStep";
    public static final String SAMPLE_SEED_S = "sampleSeed";
    public static final String ACTION_SCOPE_S = "actionScope";
    public static final String NOTIFY_EPISODE_END_S = "notifyEpisodeEnd";
    public static final String PROFILE_S = "profile";
    public static final String PROFILE_EVERY_S = "profileEvery";
    public static final String INCLUDE_COST_FEATURES_S = "includeCostFeatures";

    public static final String MAX_MSG_TOKENS_S = "maxMsgTokens"; // default 32
    public static final String MAX_MSGS_PER_NODE_S = "maxMessagesPerNode"; // for normalization

    public static final String TTL_NORM_MINUTES_S = "ttlNormMinutes";
    public static final String HOP_NORM_MAX_S = "hopNormMax";

    public static final String ENERGY_RATE_NORM_S = "energyRateNorm";
    public static final String TEMPTY_NORM_SECONDS_S = "tEmptyNormSeconds";
    public static final String INFO_AGE_NORM_SECONDS_S = "infoAgeNormSeconds";
    public static final String LINK_RATE_NORM_BPS_S = "linkRateNormBps";

    public static final String EVENT_RATE_EMA_BETA_S = "eventRateEmaBeta";
    public static final String GLOBAL_RATE_EMA_BETA_S = "globalRateEmaBeta";
    public static final String DELTA_COST_SCALE_S = "deltaCostScale";

    public static final String PHASE_A_END_S = "phaseAEnd";
    public static final String PHASE_B_END_S = "phaseBEnd";

    // Action clamp ranges (server returns values; Java clamps for safety)
    public static final String LAMBDA_COST_MIN_S = "lambdaCostMin";
    public static final String LAMBDA_COST_MAX_S = "lambdaCostMax";
    public static final String TAU_AGE_MIN_S = "tauAgeMin";
    public static final String TAU_AGE_MAX_S = "tauAgeMax";
    public static final String BETA_X_MIN_S = "betaXMin";
    public static final String BETA_X_MAX_S = "betaXMax";
    public static final String KX_MIN_S = "kxMin";
    public static final String KX_MAX_S = "kxMax";
    public static final String M_RELAY_MIN_S = "relayMarginMin";
    public static final String M_RELAY_MAX_S = "relayMarginMax";

    private final String endpoint;
    private final int timeoutMs;
    private final double activationTime;
    private final double episodeSeconds;
    private final int maxHostsPerStep;
    private final long sampleSeed;
    private final String actionScope;
    private final boolean notifyEpisodeEnd;
    private final boolean profile;
    private final int profileEvery;
    private final boolean includeCostFeatures;

    private final int maxMsgTokens;
    private final int maxMsgsPerNode;

    private final double ttlNormMinutes;
    private final double hopNormMax;
    private final double energyRateNorm;
    private final double tEmptyNormSeconds;
    private final double infoAgeNormSeconds;
    private final double linkRateNormBps;
    private final double eventRateEmaBeta;
    private final double globalRateEmaBeta;
    private final double deltaCostScale;
    private final double phaseAEnd;
    private final double phaseBEnd;

    private final double lambdaCostMin;
    private final double lambdaCostMax;
    private final double tauAgeMin;
    private final double tauAgeMax;
    private final double betaXMin;
    private final double betaXMax;
    private final double kxMin;
    private final double kxMax;
    private final double relayMarginMin;
    private final double relayMarginMax;

	    // Step counter and previous sample time (for dt)
	    private int stepIdCounter = 0;
	    private double prevSampleTime = Double.NaN;
    private boolean activationNotified = false;
    private List<DTNHost> lastHostsSnapshot = null;
    private long msgsScannedInSample = 0L;
    private long costCallsInSample = 0L;

    // Per-host counters since last sample
    private final Map<Integer, Long> relayedBytesByHost = new HashMap<Integer, Long>();
    private final Map<Integer, Long> deliveredBytesByHost = new HashMap<Integer, Long>();
    private final Map<Integer, Integer> createdCntByHost = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> transferredCntByHost = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> deliveredCntByHost = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> droppedCntByHost = new HashMap<Integer, Integer>();
    private final Map<Integer, Integer> abortedCntByHost = new HashMap<Integer, Integer>();
    private int createdSinceLastSample = 0;

    // Per-host EWMA rates (for state)
    private final Map<Integer, Double> dropRateEwmaByHost = new HashMap<Integer, Double>();
    private final Map<Integer, Double> abortRateEwmaByHost = new HashMap<Integer, Double>();

    // Global EMA rates (for critic features)
    private double globalDropRateEma = 0.0;
    private double globalAbortRateEma = 0.0;
    private double globalRelayRateEma = 0.0;

    // Energy tracking
    private final Map<Integer, Double> initialEnergyByHost = new HashMap<Integer, Double>();
    private final Map<Integer, Double> lastEnergyByHost = new HashMap<Integer, Double>();

    // Track last applied parameters per host (sent back in prev_transition)
    private final Map<Integer, double[]> lastParamsByHost = new HashMap<Integer, double[]>();

	    public RmappoMaxpropBridgeReport() {
	        super();
	        final Settings s = getSettings();
	        this.endpoint = s.getSetting(URL_S, "").trim();
	        this.timeoutMs = (int) Math.round(s.getDouble(TIMEOUT_MS_S, 30000.0));
	        this.activationTime = Math.max(0.0, s.getDouble(ACTIVATION_TIME_S, 0.0));
	        this.episodeSeconds = Math.max(1.0, s.getDouble(EPISODE_SECONDS_S, 43200.0));
	        this.maxHostsPerStep = Math.max(0, (int) Math.round(s.getDouble(MAX_HOSTS_PER_STEP_S, 0.0)));
	        // sampleSeed may be written using unresolved batch placeholders (e.g., "%%MovementModel.rngSeed%%"),
	        // which will crash Settings.getDouble(). Avoid getDouble() entirely and fall back gracefully.
	        long seed = 12345L;
	        try {
	            String seedStr = s.getSetting(SAMPLE_SEED_S, "12345").trim();
	            if (seedStr.length() == 0) {
	                seed = 12345L;
	            } else {
	                // If unresolved placeholders remain, treat as "no fixed seed".
	                if (seedStr.indexOf('%') >= 0) {
	                    seed = 0L;
	                } else {
	                    try {
	                        seed = (long) Math.round(Double.parseDouble(seedStr));
	                    } catch (Exception ignore) {
	                        seed = 0L;
	                    }
	                }
	            }
	        } catch (Exception ignore) {
	            seed = 0L;
	        }
	        this.sampleSeed = seed;
	        this.actionScope = safeScope(s.getSetting(ACTION_SCOPE_S, "per_host"));
	        this.notifyEpisodeEnd = s.getBoolean(NOTIFY_EPISODE_END_S, false);
	        this.profile = s.getBoolean(PROFILE_S, false);
	        int pe = (int) Math.round(s.getDouble(PROFILE_EVERY_S, 1.0));
	        this.profileEvery = Math.max(1, pe);
	        this.includeCostFeatures = s.getBoolean(INCLUDE_COST_FEATURES_S, true);

        this.maxMsgTokens = Math.max(0, (int) Math.round(s.getDouble(MAX_MSG_TOKENS_S, 32.0)));
        this.maxMsgsPerNode = Math.max(1, (int) Math.round(s.getDouble(MAX_MSGS_PER_NODE_S, 200.0)));

        this.ttlNormMinutes = Math.max(1.0, s.getDouble(TTL_NORM_MINUTES_S, 300.0));
        this.hopNormMax = Math.max(1.0, s.getDouble(HOP_NORM_MAX_S, 10.0));

        this.energyRateNorm = Math.max(1e-9, s.getDouble(ENERGY_RATE_NORM_S, 5.0));
        this.tEmptyNormSeconds = Math.max(1.0, s.getDouble(TEMPTY_NORM_SECONDS_S, 7200.0));
        this.infoAgeNormSeconds = Math.max(1.0, s.getDouble(INFO_AGE_NORM_SECONDS_S, 600.0));
        this.linkRateNormBps = Math.max(1.0, s.getDouble(LINK_RATE_NORM_BPS_S, 12e6));

        this.eventRateEmaBeta = clamp01(s.getDouble(EVENT_RATE_EMA_BETA_S, 0.3));
        this.globalRateEmaBeta = clamp01(s.getDouble(GLOBAL_RATE_EMA_BETA_S, 0.9));
        this.deltaCostScale = Math.max(1e-6, s.getDouble(DELTA_COST_SCALE_S, 5.0));

        this.phaseAEnd = Math.max(0.0, s.getDouble(PHASE_A_END_S, 14400.0));
        this.phaseBEnd = Math.max(this.phaseAEnd, s.getDouble(PHASE_B_END_S, 18000.0));

        this.lambdaCostMin = s.getDouble(LAMBDA_COST_MIN_S, 0.0);
        this.lambdaCostMax = s.getDouble(LAMBDA_COST_MAX_S, 2.0);
        this.tauAgeMin = s.getDouble(TAU_AGE_MIN_S, 0.0);
        this.tauAgeMax = s.getDouble(TAU_AGE_MAX_S, 3600.0);
        this.betaXMin = s.getDouble(BETA_X_MIN_S, 0.0);
        this.betaXMax = s.getDouble(BETA_X_MAX_S, 1.0);
        this.kxMin = s.getDouble(KX_MIN_S, 0.1);
        this.kxMax = s.getDouble(KX_MAX_S, 2.0);
        this.relayMarginMin = s.getDouble(M_RELAY_MIN_S, -1e9);
        this.relayMarginMax = s.getDouble(M_RELAY_MAX_S, 5.0);

	        write("# RmappoMaxpropBridgeReport active. endpoint=" + (endpoint.length() > 0 ? endpoint : "(none)") +
	                " sampleInterval=" + format(super.interval));
	        if (this.activationTime > 0.0) {
	            write("# RmappoMaxpropBridgeReport inactive until t >= " + format(this.activationTime));
	        }
	        if (this.maxHostsPerStep > 0) {
	            write("# RmappoMaxpropBridgeReport host sampling enabled: maxHostsPerStep=" + this.maxHostsPerStep);
	        }
	        write("# RmappoMaxpropBridgeReport actionScope=" + this.actionScope);
	        if (this.profile) {
	            write("# RmappoMaxpropBridgeReport profiling enabled: every=" + this.profileEvery);
	        }
	        write("# RmappoMaxpropBridgeReport includeCostFeatures=" + this.includeCostFeatures);
	    }

    @Override
	    protected void sample(List<DTNHost> hosts) {
	        final double now = SimClock.getTime();
	        final long t0 = System.nanoTime();
	        this.msgsScannedInSample = 0L;
	        this.costCallsInSample = 0L;
        final boolean controllerActive = (this.activationTime <= 0.0) || (now >= this.activationTime);
        if (controllerActive && !this.activationNotified && this.activationTime > 0.0) {
            write("# RmappoMaxpropBridgeReport activation at t=" + (int) now);
            this.activationNotified = true;
        }

	        final boolean remoteMode = controllerActive && (this.endpoint.length() > 0);

        final double prevTime = (Double.isFinite(this.prevSampleTime) ? this.prevSampleTime : now);
        final double dt = Math.max(1e-6, now - prevTime);

        // Ensure initial energy baselines exist (for normalization)
        if (hosts != null) {
            for (DTNHost h : hosts) {
                if (h == null) { continue; }
                int addr = h.getAddress();
                if (initialEnergyByHost.containsKey(addr)) {
                    continue;
                }
                double e = getEnergyValue(h);
                if (Double.isFinite(e) && e > 0.0) {
                    initialEnergyByHost.put(addr, e);
                }
            }
        }

        // Collect global critic features (5D)
        double[] globalFeatures = collectGlobalFeatures(hosts, dt);

	        // Build request JSON (only in remote mode)
	        String payload = null;
	        Map<Integer, HostTransition> transitionsByAddr = null;
	        List<DTNHost> selectedHosts = null;
	        if (remoteMode) {
	            transitionsByAddr = computeAndResetTransitions(hosts, dt);
	            selectedHosts = selectHostsForStep(hosts);
	            payload = buildRequest(selectedHosts, now, prevTime, globalFeatures, dt, transitionsByAddr);
	            this.lastHostsSnapshot = hosts;
	        }
	        final long tAfterBuild = System.nanoTime();

	        Map<String, double[]> actionsByHost = null;
	        double[] globalAction = null;
	        String policyId = "";
	        Double rewardTotal = null;
	        if (remoteMode) {
	            ActionResponse resp = callDrl(this.endpoint, payload);
	            if (resp != null) {
	                actionsByHost = resp.actionsByHost;
	                globalAction = resp.globalAction;
	                policyId = (resp.policyId != null ? resp.policyId : "");
	                rewardTotal = resp.rewardTotal;
	            } else {
	                actionsByHost = new HashMap<String, double[]>();
	            }
	        }
	        final long tAfterHttp = System.nanoTime();

	        int applied = 0;
	        if (controllerActive && hosts != null && actionsByHost != null) {
	            if (isGlobalScope(this.actionScope) && globalAction != null && globalAction.length >= 5) {
	                double lambdaCost = clamp(globalAction[0], lambdaCostMin, lambdaCostMax);
	                double tauAge = clamp(globalAction[1], tauAgeMin, tauAgeMax);
	                double betaX = clamp(globalAction[2], betaXMin, betaXMax);
	                double kx = clamp(globalAction[3], kxMin, kxMax);
	                double mRelay = clamp(globalAction[4], relayMarginMin, relayMarginMax);

	                for (DTNHost h : hosts) {
	                    if (h == null) { continue; }
	                    MessageRouter r = h.getRouter();
	                    if (!(r instanceof MaxPropRouter)) { continue; }
	                    MaxPropRouter mp = (MaxPropRouter) r;
	                    mp.setRlParams(lambdaCost, tauAge, betaX, kx, mRelay);
	                    lastParamsByHost.put(h.getAddress(), new double[]{lambdaCost, tauAge, betaX, kx, mRelay});
	                    applied++;
	                }
	            } else if (selectedHosts != null) {
	                for (DTNHost h : selectedHosts) {
	                    if (h == null) { continue; }
	                    MessageRouter r = h.getRouter();
	                    if (!(r instanceof MaxPropRouter)) { continue; }
	                    MaxPropRouter mp = (MaxPropRouter) r;

	                    double[] a = actionsByHost.get(h.toString());
	                    if (a == null || a.length < 5) {
	                        continue;
	                    }

	                    double lambdaCost = clamp(a[0], lambdaCostMin, lambdaCostMax);
	                    double tauAge = clamp(a[1], tauAgeMin, tauAgeMax);
	                    double betaX = clamp(a[2], betaXMin, betaXMax);
	                    double kx = clamp(a[3], kxMin, kxMax);
	                    double mRelay = clamp(a[4], relayMarginMin, relayMarginMax);

	                    mp.setRlParams(lambdaCost, tauAge, betaX, kx, mRelay);
	                    lastParamsByHost.put(h.getAddress(), new double[]{lambdaCost, tauAge, betaX, kx, mRelay});
	                    applied++;
	                }
	            }
	        }

	        if (remoteMode) {
	            String rStr = (rewardTotal != null && Double.isFinite(rewardTotal.doubleValue())
	                    ? format(rewardTotal.doubleValue()) : "NaN");
	            write("# RmappoMaxpropBridgeReport OK t=" + (int) now + " policy=" + policyId +
	                    " reward=" + rStr +
	                    " hosts=" + (hosts == null ? 0 : hosts.size()) +
	                    " selected_hosts=" + (selectedHosts == null ? 0 : selectedHosts.size()) +
	                    " applied_actions=" + applied);
	        }
	        if (this.profile && (this.stepIdCounter % this.profileEvery == 0)) {
	            long buildMs = (tAfterBuild - t0) / 1000000L;
	            long httpMs = (tAfterHttp - tAfterBuild) / 1000000L;
	            int payloadLen = (payload != null ? payload.length() : 0);
	            write("# RmappoMaxpropBridgeReport profile t=" + (int) now +
	                    " build_ms=" + buildMs +
	                    " http_ms=" + httpMs +
	                    " payload_kb=" + (payloadLen / 1024) +
	                    " msgs_scanned=" + this.msgsScannedInSample +
	                    " cost_calls=" + this.costCallsInSample);
	        }

        // Reset step counters and advance time
        this.prevSampleTime = now;
        this.stepIdCounter++;

        // If Python is not consuming transitions, avoid accumulating counters forever.
        if (!remoteMode) {
            resetStepCounters();
        }
    }

		    private void resetStepCounters() {
		        this.createdSinceLastSample = 0;
		        this.relayedBytesByHost.clear();
		        this.deliveredBytesByHost.clear();
		        this.createdCntByHost.clear();
		        this.transferredCntByHost.clear();
		        this.deliveredCntByHost.clear();
		        this.droppedCntByHost.clear();
		        this.abortedCntByHost.clear();
		    }

	    @Override
	    public void done() {
	        if (this.notifyEpisodeEnd && this.endpoint != null && this.endpoint.trim().length() > 0) {
	            try {
	                double now = SimClock.getTime();
	                String payload = "{\"protocol\":\"rmappo_maxprop_v1\"" +
	                        ",\"mode\":\"episode_end\"" +
	                        ",\"sim_id\":\"" + escape(getScenarioName()) + "\"" +
	                        ",\"time\":" + (int) now +
	                        ",\"step_id\":" + this.stepIdCounter +
	                        "}";
	                callDrl(this.endpoint, payload);
	            } catch (Exception ignore) {
	            }
	        }
	        super.done();
	    }

    private double[] collectGlobalFeatures(List<DTNHost> hosts, double dt) {
        int total = 0;
        int alive = 0;
        double energyFracSum = 0.0;
        int energyFracCount = 0;

        long droppedTotal = 0;
        long abortedTotal = 0;
        long relayedBytesTotal = 0;

        if (hosts != null) {
            total = hosts.size();
            for (DTNHost h : hosts) {
                if (h == null) { continue; }
                double e = getEnergyValue(h);
                if (h.isRadioActive() && Double.isFinite(e) && e > 0.0) {
                    alive++;
                }

                Double e0 = initialEnergyByHost.get(h.getAddress());
                if (Double.isFinite(e) && e0 != null && Double.isFinite(e0.doubleValue()) && e0.doubleValue() > 0.0) {
                    energyFracSum += clamp01(e / e0.doubleValue());
                    energyFracCount++;
                }

                droppedTotal += getIntOr0(droppedCntByHost, h.getAddress());
                abortedTotal += getIntOr0(abortedCntByHost, h.getAddress());
                relayedBytesTotal += getLongOr0(relayedBytesByHost, h.getAddress());
            }
        }

        double aliveFrac = (total > 0 ? ((double) alive) / (double) total : 0.0);
        double energyFrac = (energyFracCount > 0 ? energyFracSum / (double) energyFracCount : 0.0);

        double dropRate = droppedTotal / Math.max(1e-6, dt);
        double abortRate = abortedTotal / Math.max(1e-6, dt);
        double relayRate = relayedBytesTotal / Math.max(1e-6, dt);

        this.globalDropRateEma = ema(this.globalDropRateEma, dropRate, this.globalRateEmaBeta);
        this.globalAbortRateEma = ema(this.globalAbortRateEma, abortRate, this.globalRateEmaBeta);
        this.globalRelayRateEma = ema(this.globalRelayRateEma, relayRate, this.globalRateEmaBeta);

        return new double[]{
                clamp01(aliveFrac),
                clamp01(energyFrac),
                clamp01(this.globalDropRateEma / 5.0),   // scale to roughly [0,1]
                clamp01(this.globalAbortRateEma / 5.0),  // scale to roughly [0,1]
                clamp01(this.globalRelayRateEma / (12e6)) // bytes/sec scaled by nominal max
        };
    }

    private String buildRequest(List<DTNHost> hosts, double now, double prevTime, double[] globalFeatures,
            double dt, Map<Integer, HostTransition> transitionsByAddr) {
        String simId = escape(getScenarioName());
        StringBuilder sb = new StringBuilder(1 << 16);

        sb.append("{\"protocol\":\"rmappo_maxprop_v1\"")
                .append(",\"mode\":\"infer\"")
                .append(",\"sim_id\":\"").append(simId).append("\"")
                .append(",\"time\":").append((int) now)
                .append(",\"step_id\":").append(this.stepIdCounter)
                .append(",\"prev_time\":").append(format(prevTime))
                .append(",\"dt\":").append(format(dt))
                .append(",\"action_scope\":\"").append(escape(this.actionScope)).append("\"")
                .append(",\"created_since_last\":").append(this.createdSinceLastSample)
                .append(",\"global_features\":[");
        for (int i = 0; i < globalFeatures.length; i++) {
            if (i > 0) { sb.append(','); }
            sb.append(format(safe(globalFeatures[i])));
        }
        sb.append("]");

        // Action spec (for server-side clamping/mapping)
        sb.append(",\"action_spec\":{")
                .append("\"lambda_cost_min\":").append(format(lambdaCostMin)).append(',')
                .append("\"lambda_cost_max\":").append(format(lambdaCostMax)).append(',')
                .append("\"tau_age_min\":").append(format(tauAgeMin)).append(',')
                .append("\"tau_age_max\":").append(format(tauAgeMax)).append(',')
                .append("\"beta_x_min\":").append(format(betaXMin)).append(',')
                .append("\"beta_x_max\":").append(format(betaXMax)).append(',')
                .append("\"k_x_min\":").append(format(kxMin)).append(',')
                .append("\"k_x_max\":").append(format(kxMax)).append(',')
                .append("\"m_relay_min\":").append(format(relayMarginMin)).append(',')
                .append("\"m_relay_max\":").append(format(relayMarginMax))
                .append("}");

        // prev_transition per host
        sb.append(",\"prev_transition\":[");
        boolean firstPrev = true;
        if (hosts != null) {
            for (DTNHost h : hosts) {
                if (h == null) { continue; }
                MessageRouter r = h.getRouter();
                if (!(r instanceof MaxPropRouter)) { continue; }

                int addr = h.getAddress();
                HostTransition tr = (transitionsByAddr != null ? transitionsByAddr.get(addr) : null);
                if (tr == null) { continue; }

                if (!firstPrev) { sb.append(','); }
                firstPrev = false;
	                sb.append("{\"host\":\"").append(escape(h.toString())).append("\"")
	                        .append(",\"created_cnt\":").append(tr.createdCnt)
	                        .append(",\"transferred_cnt\":").append(tr.transferredCnt)
	                        .append(",\"delivered_cnt\":").append(tr.deliveredCnt)
	                        .append(",\"delivered_bytes\":").append(tr.deliveredBytes)
	                        .append(",\"relayed_bytes\":").append(tr.relayedBytes)
	                        .append(",\"dropped\":").append(tr.dropped)
	                        .append(",\"aborted\":").append(tr.aborted)
	                        .append(",\"energy_used\":").append(format(safe(tr.energyUsed)));
                if (tr.lastParams != null && tr.lastParams.length >= 5) {
                    sb.append(",\"lambda_cost\":").append(format(safe(tr.lastParams[0])))
                            .append(",\"tau_age\":").append(format(safe(tr.lastParams[1])))
                            .append(",\"beta_x\":").append(format(safe(tr.lastParams[2])))
                            .append(",\"k_x\":").append(format(safe(tr.lastParams[3])))
                            .append(",\"m_relay\":").append(format(safe(tr.lastParams[4])));
                }
                sb.append('}');
            }
        }
        sb.append(']');

        // state_batch
        sb.append(",\"state_batch\":[");
        boolean firstState = true;
        if (hosts != null) {
            for (DTNHost h : hosts) {
                if (h == null) { continue; }
                MessageRouter r = h.getRouter();
                if (!(r instanceof MaxPropRouter)) { continue; }
                MaxPropRouter mp = (MaxPropRouter) r;

                HostObs obs = buildHostObs(h, mp, now);
                if (obs == null) { continue; }

                if (!firstState) { sb.append(','); }
                firstState = false;
                sb.append("{\"host\":\"").append(escape(h.toString())).append("\"");
                if (obs.contactHost != null) {
                    sb.append(",\"contact\":\"").append(escape(obs.contactHost)).append("\"");
                }
                sb.append(",\"obs\":[");
                for (int i = 0; i < obs.obs.length; i++) {
                    if (i > 0) { sb.append(','); }
                    sb.append(format(safe(obs.obs[i])));
                }
                sb.append("]}");
            }
        }
        sb.append("]}");

        // Reset per-step created counter after embedding into request
        this.createdSinceLastSample = 0;

        return sb.toString();
    }

    private static class HostObs {
        final String contactHost;
        final double[] obs;
        HostObs(String contactHost, double[] obs) {
            this.contactHost = contactHost;
            this.obs = obs;
        }
    }

    private HostObs buildHostObs(final DTNHost host, final MaxPropRouter mp, final double now) {
        if (host == null || mp == null) {
            return null;
        }

        final int addr = host.getAddress();

        // ----- Select representative contact (deterministic: lowest address) -----
        DTNHost contact = null;
        Connection contactCon = null;
        for (Connection c : host.getConnections()) {
            if (c == null) { continue; }
            DTNHost other = c.getOtherNode(host);
            if (other == null) { continue; }
            if (contact == null || other.getAddress() < contact.getAddress()) {
                contact = other;
                contactCon = c;
            }
        }

        double linkRate = 0.0;
        if (contactCon != null) {
            linkRate = contactCon.getSpeed();
            if (!Double.isFinite(linkRate) || linkRate < 0.0) { linkRate = 0.0; }
        }

        // ----- SELF features -----
        double e = mp.getLocalEnergyValue();
        Double e0Obj = initialEnergyByHost.get(addr);
        double e0 = (e0Obj != null ? e0Obj.doubleValue() : Double.NaN);
        double eNorm = (Double.isFinite(e) && Double.isFinite(e0) && e0 > 0.0) ? clamp01(e / e0) : 0.0;

        double r = mp.getLocalEnergyRateEstimate();
        if (!Double.isFinite(r) || r < 0.0) { r = 0.0; }
        double rNorm = clamp01(r / this.energyRateNorm);

        double tEmpty = (r > 1e-9 && Double.isFinite(e) && e >= 0.0) ? (e / r) : this.tEmptyNormSeconds;
        if (!Double.isFinite(tEmpty) || tEmpty < 0.0) { tEmpty = 0.0; }
        double tNorm = clamp01(tEmpty / this.tEmptyNormSeconds);

        double bufOcc = 0.0;
        long bufSize = mp.getBufferSize();
        long free = mp.getFreeBufferSize();
        if (bufSize > 0 && bufSize < Integer.MAX_VALUE) {
            double util = 1.0 - ((double) free / (double) bufSize);
            bufOcc = clamp01(util);
        }

        double nMsgsNorm = clamp01(((double) host.getNrofMessages()) / (double) this.maxMsgsPerNode);

        double dropE = getOr0(this.dropRateEwmaByHost, addr);
        double abortE = getOr0(this.abortRateEwmaByHost, addr);
        double dropEFeat = clamp01(dropE / 1.0);
        double abortEFeat = clamp01(abortE / 1.0);

        double xEst = mp.getAvgTransferredBytesEstimate();
        if (!Double.isFinite(xEst) || xEst < 0.0) { xEst = 0.0; }
        double xNorm = (bufSize > 0) ? clamp01(xEst / (double) bufSize) : 0.0;

        int thresholdCur = mp.calcThreshold();
        double thresholdNorm = clamp01(((double) thresholdCur) / this.hopNormMax);

        double tEpisodeNorm = clamp01(now / this.episodeSeconds);

        double phaseA = (now < this.phaseAEnd ? 1.0 : 0.0);
        double phaseB = (now >= this.phaseAEnd && now < this.phaseBEnd ? 1.0 : 0.0);
        double phaseC = (now >= this.phaseBEnd ? 1.0 : 0.0);

        // ----- CONTACT features -----
        double eNbNorm = 0.0;
        double rNbNorm = 0.0;
        double tNbNorm = 0.0;
        double ageInfoNorm = 1.0;
        double linkRateNorm = clamp01(linkRate / this.linkRateNormBps);
        double hasContact = 0.0;

        if (contact != null) {
            hasContact = 1.0;
            int nbAddr = contact.getAddress();
            double eNb = mp.getKnownEnergyValue(nbAddr);
            double rNb = mp.getKnownEnergyRate(nbAddr);
            double tInfo = mp.getKnownEnergyInfoTime(nbAddr);

            // Fallback to peer's self-reported values if local table has no entry yet.
            MessageRouter peerRouter = contact.getRouter();
            if ((!Double.isFinite(eNb) || !Double.isFinite(rNb)) && (peerRouter instanceof MaxPropRouter)) {
                MaxPropRouter peerMp = (MaxPropRouter) peerRouter;
                if (!Double.isFinite(eNb)) { eNb = peerMp.getLocalEnergyValue(); }
                if (!Double.isFinite(rNb)) { rNb = peerMp.getLocalEnergyRateEstimate(); }
                if (!Double.isFinite(tInfo)) { tInfo = now; }
            }

            Double e0NbObj = initialEnergyByHost.get(nbAddr);
            double e0Nb = (e0NbObj != null ? e0NbObj.doubleValue() : Double.NaN);
            eNbNorm = (Double.isFinite(eNb) && Double.isFinite(e0Nb) && e0Nb > 0.0) ? clamp01(eNb / e0Nb) : 0.0;

            if (!Double.isFinite(rNb) || rNb < 0.0) { rNb = 0.0; }
            rNbNorm = clamp01(rNb / this.energyRateNorm);

            double tNb = (rNb > 1e-9 && Double.isFinite(eNb) && eNb >= 0.0) ? (eNb / rNb) : this.tEmptyNormSeconds;
            if (!Double.isFinite(tNb) || tNb < 0.0) { tNb = 0.0; }
            tNbNorm = clamp01(tNb / this.tEmptyNormSeconds);

            double age = (Double.isFinite(tInfo) ? Math.max(0.0, now - tInfo) : this.infoAgeNormSeconds);
            ageInfoNorm = clamp01(age / this.infoAgeNormSeconds);
        }

        // ----- MSG tokens (top-N deterministic selection) -----
        List<Message> selected = selectMessages(host, mp, thresholdCur, this.maxMsgTokens);
        int msgDim = 10;
        int selfDim = 13;
        int contactDim = 6;
        int totalObsDim = selfDim + contactDim + (this.maxMsgTokens * msgDim) + this.maxMsgTokens;
        double[] obs = new double[totalObsDim];
        int idx = 0;

        // SELF (13D)
        obs[idx++] = eNorm;
        obs[idx++] = rNorm;
        obs[idx++] = tNorm;
        obs[idx++] = bufOcc;
        obs[idx++] = nMsgsNorm;
        obs[idx++] = abortEFeat;
        obs[idx++] = dropEFeat;
        obs[idx++] = xNorm;
        obs[idx++] = thresholdNorm;
        obs[idx++] = tEpisodeNorm;
        obs[idx++] = phaseA;
        obs[idx++] = phaseB;
        obs[idx++] = phaseC;

        // CONTACT (6D)
        obs[idx++] = eNbNorm;
        obs[idx++] = rNbNorm;
        obs[idx++] = tNbNorm;
        obs[idx++] = ageInfoNorm;
        obs[idx++] = linkRateNorm;
        obs[idx++] = hasContact;

        // MSG tokens (32 x 10D)
        int realCount = 0;
        for (Message m : selected) {
            if (m == null) { continue; }
            if (realCount >= this.maxMsgTokens) { break; }
            double[] mf = buildMessageFeatures(host, mp, m, contact, linkRate, thresholdCur, bufSize);
            for (int k = 0; k < msgDim; k++) {
                obs[idx++] = mf[k];
            }
            realCount++;
        }
        // pad remaining tokens with zeros
        for (int p = realCount; p < this.maxMsgTokens; p++) {
            for (int k = 0; k < msgDim; k++) {
                obs[idx++] = 0.0;
            }
        }

        // MSG mask (32D)
        for (int i = 0; i < this.maxMsgTokens; i++) {
            obs[idx++] = (i < realCount ? 1.0 : 0.0);
        }

        return new HostObs(contact != null ? contact.toString() : null, obs);
    }

    private List<Message> selectMessages(final DTNHost host, final MaxPropRouter mp, final int thresholdCur, final int limit) {
        if (host == null || mp == null || limit <= 0) {
            return Collections.emptyList();
        }
        Collection<Message> msgs = host.getMessageCollection();
        if (msgs == null || msgs.isEmpty()) {
            return Collections.emptyList();
        }

        if (!this.includeCostFeatures) {
            final Comparator<MessageScore> bestFirst = new Comparator<MessageScore>() {
                public int compare(MessageScore a, MessageScore b) {
                    if (a == b) { return 0; }
                    if (a.prio != b.prio) { return a.prio ? -1 : 1; }
                    int d = a.hop - b.hop;
                    if (d != 0) { return d; }
                    return a.m.getId().compareTo(b.m.getId());
                }
            };
            final Comparator<MessageScore> worstFirst = new Comparator<MessageScore>() {
                public int compare(MessageScore a, MessageScore b) {
                    return bestFirst.compare(b, a);
                }
            };

            java.util.PriorityQueue<MessageScore> heap =
                    new java.util.PriorityQueue<MessageScore>(limit + 1, worstFirst);
            for (Message m : msgs) {
                if (m == null) { continue; }
                this.msgsScannedInSample++;
                int hop = m.getHopCount();
                boolean prio = hop < thresholdCur;
                MessageScore ms = new MessageScore(m, prio, hop, 0.0);
                if (heap.size() < limit) {
                    heap.add(ms);
                } else {
                    MessageScore worst = heap.peek();
                    if (worst != null && bestFirst.compare(ms, worst) < 0) {
                        heap.poll();
                        heap.add(ms);
                    }
                }
            }

            if (heap.isEmpty()) {
                return Collections.emptyList();
            }
            List<MessageScore> top = new ArrayList<MessageScore>(heap.size());
            while (!heap.isEmpty()) {
                top.add(heap.poll());
            }
            Collections.sort(top, bestFirst);

            List<Message> out = new ArrayList<Message>(top.size());
            for (MessageScore ms : top) {
                if (ms == null || ms.m == null) { continue; }
                out.add(ms.m);
            }
            return out;
        }

        final Comparator<MessageScore> bestFirst = new Comparator<MessageScore>() {
            public int compare(MessageScore a, MessageScore b) {
                if (a == b) { return 0; }
                if (a.prio != b.prio) { return a.prio ? -1 : 1; }
                if (a.prio && b.prio) {
                    int d = a.hop - b.hop;
                    if (d != 0) { return d; }
                } else {
                    int d = Double.compare(a.cost, b.cost);
                    if (d != 0) { return d; }
                }
                int dHop = a.hop - b.hop;
                if (dHop != 0) { return dHop; }
                return a.m.getId().compareTo(b.m.getId());
            }
        };
        final Comparator<MessageScore> worstFirst = new Comparator<MessageScore>() {
            public int compare(MessageScore a, MessageScore b) {
                return bestFirst.compare(b, a);
            }
        };

        java.util.PriorityQueue<MessageScore> heap =
                new java.util.PriorityQueue<MessageScore>(limit + 1, worstFirst);
        for (Message m : msgs) {
            if (m == null) { continue; }
            this.msgsScannedInSample++;
            int hop = m.getHopCount();
            boolean prio = hop < thresholdCur;
            double cost;
            if (prio) {
                cost = 0.0;
            } else {
                this.costCallsInSample++;
                cost = mp.getCost(host, m.getTo());
            }
            MessageScore ms = new MessageScore(m, prio, hop, cost);
            if (heap.size() < limit) {
                heap.add(ms);
            } else {
                MessageScore worst = heap.peek();
                if (worst != null && bestFirst.compare(ms, worst) < 0) {
                    heap.poll();
                    heap.add(ms);
                }
            }
        }

        if (heap.isEmpty()) {
            return Collections.emptyList();
        }
        List<MessageScore> top = new ArrayList<MessageScore>(heap.size());
        while (!heap.isEmpty()) {
            top.add(heap.poll());
        }
        Collections.sort(top, bestFirst);

        List<Message> out = new ArrayList<Message>(top.size());
        for (MessageScore ms : top) {
            if (ms == null || ms.m == null) { continue; }
            out.add(ms.m);
        }
        return out;
    }

    private static class MessageScore {
        final Message m;
        final boolean prio;
        final int hop;
        final double cost;
        MessageScore(Message m, boolean prio, int hop, double cost) {
            this.m = m;
            this.prio = prio;
            this.hop = hop;
            this.cost = cost;
        }
    }

    private double[] buildMessageFeatures(final DTNHost host,
                                          final MaxPropRouter mp,
                                          final Message m,
                                          final DTNHost contact,
                                          final double linkRate,
                                          final int thresholdCur,
                                          final long bufSize) {
        double[] f = new double[10];
        int i = 0;

        double sizeNorm = 0.0;
        if (bufSize > 0) {
            sizeNorm = clamp01(((double) m.getSize()) / (double) bufSize);
        }

        double ttlRem = (double) m.getTtl();
        if (!Double.isFinite(ttlRem) || ttlRem < 0.0) { ttlRem = 0.0; }
        if (ttlRem > Integer.MAX_VALUE / 2) { ttlRem = this.ttlNormMinutes; }
        double ttlNorm = clamp01(ttlRem / this.ttlNormMinutes);

        double ageSec = Math.max(0.0, SimClock.getTime() - m.getCreationTime());
        double ageNorm = clamp01(ageSec / (this.ttlNormMinutes * 60.0));

        int hop = m.getHopCount();
        double hopNorm = clamp01(((double) hop) / this.hopNormMax);
	        double isPrio = (hop < thresholdCur ? 1.0 : 0.0);

        double costSelfFeat = 0.0;
        double costViaFeat = 0.0;
        double deltaFeat = 0.0;
        if (this.includeCostFeatures) {
            this.costCallsInSample++;
            double costSelf = mp.getCost(host, m.getTo());
            double costVia = costSelf;
            if (contact != null) {
                MessageRouter peerRouter = contact.getRouter();
                if (peerRouter instanceof MaxPropRouter) {
                    this.costCallsInSample++;
                    MaxPropRouter peerMp = (MaxPropRouter) peerRouter;
                    costVia = peerMp.getCost(contact, m.getTo());
                } else {
                    this.costCallsInSample++;
                    costVia = mp.getCost(contact, m.getTo());
                }
            }

            costSelfFeat = costToFeat(costSelf);
            costViaFeat = costToFeat(costVia);

            double delta = costSelf - costVia;
            deltaFeat = Math.tanh(delta / this.deltaCostScale);
            if (!Double.isFinite(deltaFeat)) { deltaFeat = 0.0; }
        }

        double alreadySent = 0.0;
        if (contact != null) {
            alreadySent = mp.hasSentMessageTo(contact, m.getId()) ? 1.0 : 0.0;
        }

        double txTimeNorm = 0.0;
        if (contact != null && linkRate > 1e-9) {
            double ttx = ((double) m.getSize()) / linkRate;
            if (Double.isFinite(ttx) && ttx > 0.0) {
                txTimeNorm = clamp01(ttx / super.interval);
            }
        }

        f[i++] = sizeNorm;
        f[i++] = ttlNorm;
        f[i++] = ageNorm;
        f[i++] = hopNorm;
        f[i++] = isPrio;
        f[i++] = costSelfFeat;
        f[i++] = costViaFeat;
        f[i++] = deltaFeat;
        f[i++] = alreadySent;
        f[i++] = txTimeNorm;
        return f;
    }

    private double costToFeat(double cost) {
        if (!Double.isFinite(cost) || cost < 0.0) {
            return 0.0;
        }
        // Map cost in [0, +inf) to (0,1], where lower cost is better (closer to 1).
        double v = 1.0 / (1.0 + cost);
        return clamp01(v);
    }

    private double getEnergyValue(DTNHost h) {
        if (h == null || h.getComBus() == null) {
            return Double.NaN;
        }
        Object ev = h.getComBus().getProperty(routing.util.EnergyModel.ENERGY_VALUE_ID);
        if (ev instanceof Double) {
            return ((Double) ev).doubleValue();
        }
        return Double.NaN;
    }

    private double getEnergyUsedSinceLast(DTNHost h) {
        if (h == null) {
            return 0.0;
        }
        int addr = h.getAddress();
        double e = getEnergyValue(h);
        Double prev = lastEnergyByHost.get(addr);
        lastEnergyByHost.put(addr, e);
        if (prev == null || !Double.isFinite(prev.doubleValue()) || !Double.isFinite(e)) {
            return 0.0;
        }
        double used = prev.doubleValue() - e;
        if (!Double.isFinite(used) || used < 0.0) {
            return 0.0;
        }
        return used;
    }

    private int getAndResetInt(Map<Integer, Integer> map, int key) {
        Integer v = map.get(key);
        if (v == null) { v = 0; }
        map.put(key, 0);
        return v.intValue();
    }

    private long getAndResetLong(Map<Integer, Long> map, int key) {
        Long v = map.get(key);
        if (v == null) { v = 0L; }
        map.put(key, 0L);
        return v.longValue();
    }

    private int getIntOr0(Map<Integer, Integer> map, int key) {
        Integer v = map.get(key);
        return (v != null ? v.intValue() : 0);
    }

    private long getLongOr0(Map<Integer, Long> map, int key) {
        Long v = map.get(key);
        return (v != null ? v.longValue() : 0L);
    }

    private double getOr0(Map<Integer, Double> map, int key) {
        Double v = map.get(key);
        if (v == null || !Double.isFinite(v.doubleValue())) { return 0.0; }
        return v.doubleValue();
    }

    private double ema(double prev, double x, double beta) {
        double b = clamp01(beta);
        double v = (1.0 - b) * safe(prev) + b * safe(x);
        if (!Double.isFinite(v) || v < 0.0) { v = 0.0; }
        return v;
    }

    private double safe(double v) {
        if (!Double.isFinite(v)) { return 0.0; }
        return v;
    }

    private double clamp(double v, double lo, double hi) {
        if (!Double.isFinite(v)) { v = 0.0; }
        if (!Double.isFinite(lo)) { lo = v; }
        if (!Double.isFinite(hi)) { hi = v; }
        if (lo > hi) { double t = lo; lo = hi; hi = t; }
        if (v < lo) { return lo; }
        if (v > hi) { return hi; }
        return v;
    }

    private double clamp01(double v) {
        if (!Double.isFinite(v)) { return 0.0; }
        if (v < 0.0) { return 0.0; }
        if (v > 1.0) { return 1.0; }
        return v;
    }

    private String escape(String s) {
        if (s == null) { return ""; }
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

	    private static class ActionResponse {
	        final String policyId;
	        final Map<String, double[]> actionsByHost;
	        final double[] globalAction;
	        final Double rewardTotal;

	        ActionResponse(String policyId, Map<String, double[]> actionsByHost, double[] globalAction, Double rewardTotal) {
	            this.policyId = policyId;
	            this.actionsByHost = actionsByHost;
	            this.globalAction = globalAction;
	            this.rewardTotal = rewardTotal;
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
            out.flush();
            out.close();
            int code = con.getResponseCode();
            if (code != 200) { return null; }
            BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
            StringBuilder resp = new StringBuilder();
            String line;
            while ((line = in.readLine()) != null) {
                resp.append(line);
            }
            in.close();
	            String body = resp.toString();
	            String policyId = "";
	            try { policyId = extractString(body, "policy_id"); } catch (Exception ignore) {}
	            Map<String, double[]> actions = parseActions(body);
	            double[] globalAction = parseGlobalAction(body);
	            Double rewardTotal = parseRewardTotal(body);
	            return new ActionResponse(policyId, actions, globalAction, rewardTotal);
	        } catch (Exception e) {
	            write("# RmappoMaxpropBridgeReport HTTP error: " + e.getMessage());
	            return null;
	        } finally {
            if (con != null) {
                try { con.disconnect(); } catch (Exception ignore) {}
            }
        }
    }

    private Map<String, double[]> parseActions(String json) {
        Map<String, double[]> result = new HashMap<String, double[]>();
        if (json == null) { return result; }
        int idx = json.indexOf("\"actions\"");
        if (idx < 0) { return result; }
        int arrStart = json.indexOf('[', idx);
        if (arrStart < 0) { return result; }
        int arrEnd = findMatchingBracket(json, arrStart, '[', ']');
        if (arrEnd < 0) { return result; }
        String arr = json.substring(arrStart + 1, arrEnd);
        List<String> hostObjs = extractTopLevelObjects(arr);
        for (String hb : hostObjs) {
            String host = extractString(hb, "host");
            if (host == null) { continue; }
            Double lambda = extractDouble(hb, "lambda_cost");
            Double tau = extractDouble(hb, "tau_age");
            Double beta = extractDouble(hb, "beta_x");
            Double kx = extractDouble(hb, "k_x");
            Double mr = extractDouble(hb, "m_relay");
            if (lambda == null || tau == null || beta == null || kx == null || mr == null) {
                continue;
            }
            result.put(host, new double[]{lambda.doubleValue(), tau.doubleValue(), beta.doubleValue(),
                    kx.doubleValue(), mr.doubleValue()});
        }
        return result;
    }

    private double[] parseGlobalAction(String json) {
        if (json == null) { return null; }
        int idx = json.indexOf("\"action_global\"");
        if (idx < 0) { return null; }
        int objStart = json.indexOf('{', idx);
        if (objStart < 0) { return null; }
        int objEnd = findMatchingBracket(json, objStart, '{', '}');
        if (objEnd < 0) { return null; }
        String obj = json.substring(objStart, objEnd + 1);

        Double lambda = extractDouble(obj, "lambda_cost");
        Double tau = extractDouble(obj, "tau_age");
        Double beta = extractDouble(obj, "beta_x");
        Double kx = extractDouble(obj, "k_x");
        Double mr = extractDouble(obj, "m_relay");
        if (lambda == null || tau == null || beta == null || kx == null || mr == null) {
            return null;
        }
        return new double[]{lambda.doubleValue(), tau.doubleValue(), beta.doubleValue(),
                kx.doubleValue(), mr.doubleValue()};
    }

    private int findMatchingBracket(String s, int start, char open, char close) {
        int depth = 0;
        for (int i = start; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == open) { depth++; }
            else if (ch == close) {
                depth--;
                if (depth == 0) { return i; }
            }
        }
        return -1;
    }

    private List<String> extractTopLevelObjects(String s) {
        List<String> out = new ArrayList<String>();
        int i = 0;
        while (i < s.length()) {
            while (i < s.length()) {
                char ch = s.charAt(i);
                if (Character.isWhitespace(ch) || ch == ',') { i++; } else { break; }
            }
            if (i >= s.length()) { break; }
            if (s.charAt(i) != '{') { i++; continue; }
            int end = findMatchingBracket(s, i, '{', '}');
            if (end < 0) { break; }
            out.add(s.substring(i, end + 1));
            i = end + 1;
        }
        return out;
    }

    private String extractString(String block, String key) {
        String k = "\"" + key + "\"";
        int i = block.indexOf(k);
        if (i < 0) { return null; }
        int q1 = block.indexOf('"', i + k.length());
        if (q1 < 0) { return null; }
        int q2 = block.indexOf('"', q1 + 1);
        if (q2 < 0) { return null; }
        return block.substring(q1 + 1, q2);
    }

    private Double extractDouble(String block, String key) {
        String k = "\"" + key + "\"";
        int i = block.indexOf(k);
        if (i < 0) { return null; }
        int c = block.indexOf(':', i + k.length());
        if (c < 0) { return null; }
        int e = c + 1;
        while (e < block.length() && Character.isWhitespace(block.charAt(e))) { e++; }
        int j = e;
        while (j < block.length()) {
            char ch = block.charAt(j);
            if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '.' || ch == 'e' || ch == 'E' || ch == '+') {
                j++;
            } else {
                break;
            }
        }
        try {
            return Double.parseDouble(block.substring(e, j).trim());
        } catch (Exception ex) {
            return null;
        }
    }

	    private static class HostTransition {
	        final int addr;
	        final int createdCnt;
	        final int transferredCnt;
	        final int deliveredCnt;
	        final long deliveredBytes;
	        final long relayedBytes;
	        final int dropped;
	        final int aborted;
	        final double energyUsed;
	        final double[] lastParams;

	        HostTransition(int addr, int createdCnt, int transferredCnt, int deliveredCnt,
	                long deliveredBytes, long relayedBytes, int dropped,
	                int aborted, double energyUsed, double[] lastParams) {
	            this.addr = addr;
	            this.createdCnt = createdCnt;
	            this.transferredCnt = transferredCnt;
	            this.deliveredCnt = deliveredCnt;
	            this.deliveredBytes = deliveredBytes;
	            this.relayedBytes = relayedBytes;
	            this.dropped = dropped;
	            this.aborted = aborted;
	            this.energyUsed = energyUsed;
	            this.lastParams = lastParams;
	        }
	    }

	    private Double parseRewardTotal(String json) {
	        if (json == null) { return null; }
	        int idx = json.indexOf("\"reward_debug\"");
	        if (idx < 0) { return null; }
	        int objStart = json.indexOf('{', idx);
	        if (objStart < 0) { return null; }
	        int objEnd = findMatchingBracket(json, objStart, '{', '}');
	        if (objEnd < 0) { return null; }
	        String obj = json.substring(objStart, objEnd + 1);
	        return extractDouble(obj, "total");
	    }

	    private Map<Integer, HostTransition> computeAndResetTransitions(List<DTNHost> hosts, double dt) {
	        Map<Integer, HostTransition> out = new HashMap<Integer, HostTransition>();
	        if (hosts == null) { return out; }

        for (DTNHost h : hosts) {
            if (h == null) { continue; }
	            MessageRouter r = h.getRouter();
	            if (!(r instanceof MaxPropRouter)) { continue; }

	            int addr = h.getAddress();
	            int createdCnt = getAndResetInt(createdCntByHost, addr);
	            int transferredCnt = getAndResetInt(transferredCntByHost, addr);
	            int deliveredCnt = getAndResetInt(deliveredCntByHost, addr);
	            long deliveredBytes = getAndResetLong(deliveredBytesByHost, addr);
	            long relayedBytes = getAndResetLong(relayedBytesByHost, addr);
	            int dropped = getAndResetInt(droppedCntByHost, addr);
	            int aborted = getAndResetInt(abortedCntByHost, addr);
	            double energyUsed = getEnergyUsedSinceLast(h);

            double dropRate = dropped / Math.max(1e-6, dt);
            double abortRate = aborted / Math.max(1e-6, dt);
            double dropE = ema(getOr0(dropRateEwmaByHost, addr), dropRate, this.eventRateEmaBeta);
            double abortE = ema(getOr0(abortRateEwmaByHost, addr), abortRate, this.eventRateEmaBeta);
            dropRateEwmaByHost.put(addr, dropE);
	            abortRateEwmaByHost.put(addr, abortE);

	            double[] last = lastParamsByHost.get(addr);
	            out.put(addr, new HostTransition(addr, createdCnt, transferredCnt, deliveredCnt,
	                    deliveredBytes, relayedBytes, dropped, aborted, energyUsed, last));
	        }

	        return out;
	    }

    private List<DTNHost> selectHostsForStep(List<DTNHost> hosts) {
        if (hosts == null) { return null; }

        List<DTNHost> candidates = new ArrayList<DTNHost>();
        for (DTNHost h : hosts) {
            if (h == null) { continue; }
            MessageRouter r = h.getRouter();
            if (!(r instanceof MaxPropRouter)) { continue; }
            candidates.add(h);
        }

        if (this.maxHostsPerStep <= 0 || candidates.size() <= this.maxHostsPerStep) {
            return candidates;
        }

        long baseSeed;
        if (this.sampleSeed == 0L) {
            baseSeed = System.nanoTime() ^ Double.doubleToLongBits(SimClock.getTime());
        } else {
            baseSeed = this.sampleSeed;
        }
        long salt = ((long) this.stepIdCounter) * 1103515245L + 12345L;
        Random rnd = new Random(baseSeed ^ salt);
        Collections.shuffle(candidates, rnd);
        return new ArrayList<DTNHost>(candidates.subList(0, this.maxHostsPerStep));
    }

    private boolean isGlobalScope(String v) {
        if (v == null) { return false; }
        String s = v.trim().toLowerCase();
        return s.equals("global") || s.equals("shared");
    }

    private String safeScope(String v) {
        if (v == null) { return "per_host"; }
        String s = v.trim().toLowerCase();
        if (s.length() == 0) { return "per_host"; }
        if (isGlobalScope(s)) { return "global"; }
        return "per_host";
    }

	    // MessageListener hooks to collect step counters
	    public void newMessage(Message m) {
	        this.createdSinceLastSample++;
	        if (m == null || m.getFrom() == null) {
	            return;
	        }
	        int addr = m.getFrom().getAddress();
	        Integer v = createdCntByHost.get(addr);
	        createdCntByHost.put(addr, (v == null ? 1 : v + 1));
	    }

    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) {
        // no-op
    }

    public void messageDeleted(Message m, DTNHost where, boolean dropped) {
        if (!dropped || where == null) {
            return;
        }
        int addr = where.getAddress();
        Integer v = droppedCntByHost.get(addr);
        droppedCntByHost.put(addr, (v == null ? 1 : v + 1));
    }

    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) {
        if (from == null) {
            return;
        }
        int addr = from.getAddress();
        Integer v = abortedCntByHost.get(addr);
        abortedCntByHost.put(addr, (v == null ? 1 : v + 1));
    }

	    public void messageTransferred(Message m, DTNHost from, DTNHost to, boolean firstDelivery) {
	        if (m == null || from == null) {
	            return;
	        }
	        int addr = from.getAddress();
	        Integer tx = transferredCntByHost.get(addr);
	        transferredCntByHost.put(addr, (tx == null ? 1 : tx + 1));
	        if (firstDelivery) {
	            Integer d = deliveredCntByHost.get(addr);
	            deliveredCntByHost.put(addr, (d == null ? 1 : d + 1));
	            Long v = deliveredBytesByHost.get(addr);
	            deliveredBytesByHost.put(addr, (v == null ? (long) m.getSize() : v + (long) m.getSize()));
	        } else {
	            Long v = relayedBytesByHost.get(addr);
	            relayedBytesByHost.put(addr, (v == null ? (long) m.getSize() : v + (long) m.getSize()));
	        }
	    }
}
