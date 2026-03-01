package report;


import core.DTNHost;
import core.Message;
import core.Settings;
import core.SettingsError;
import core.SimClock;
import core.UpdateListener;
import core.NetworkInterface;
import routing.MessageRouter;
import routing.ProphetRouter;





import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Heuristic buffer-aware controller. At each sampling interval, nodes share
 * buffer occupancy knowledge with contacts and adjust PRoPHET predictabilities
 * using fixed deltas based on the average load category.
 *
 * Modes (BufferLoadHeuristicReport.mode):
 *  - category (default): use mean occupancy thresholds (low/medium/high) and apply a uniform delta per host.
 *  - pressure: compare self occupancy vs. mean occupancy and apply a uniform delta per host.
 *  - pbase_pressure: per-(host,dest) delta based on base PRoPHET predictability, pressure, role (contacts/capacity),
 *    and per-destination buffer load.
 *
 * Categories:
 *  - low (<= lowThreshold): decrease own predictability by delta (encourage forwarding).
 *  - medium (between lowThreshold and highThreshold): no change.
 *  - high (>= highThreshold): increase own predictability by delta (discourage forwarding).
 */
public class BufferLoadHeuristicReport extends SamplingReport implements UpdateListener {

    public static final String MODE_S = "mode";
    public static final String DELTA_VALUE_S = "deltaValue";
    public static final String LOW_THRESHOLD_S = "lowThreshold";
    public static final String HIGH_THRESHOLD_S = "highThreshold";
    public static final String BUF_OCC_MAX_AGE_S = "bufOccMaxAge";
    public static final String LOG_ACTIONS_S = "logActions";
    public static final String LOG_ACTIONS_MAX_S = "logActionsMax";
    public static final String ACTIVATION_TIME_S = "activationTime";
    public static final String THRESHOLD_SCHEDULE_S = "thresholdSchedule";
    public static final String CONTACTS_CMAX_S = "contactsCmax";
    public static final String PRESSURE_SCALE_S = "pressureScale";
    public static final String PBASE_THRESHOLD_S = "pBaseThreshold";
    public static final String ROLE_THRESHOLD_S = "roleThreshold";
    public static final String CARRIER_OFFLOAD_THRESHOLD_S = "carrierOffloadThreshold";
    public static final String CARRIER_OFFLOAD_SCALE_S = "carrierOffloadScale";
    public static final String RANGE_WIFI_S = "rangeWifi";
    public static final String RANGE_BACKHAUL_S = "rangeBackhaul";
    public static final String PARAMS_ENABLE_S = "paramsEnable";
    public static final String PARAMS_EMA_BETA_S = "paramsEmaBeta";
    public static final String PARAMS_PRESSURE_BAND_S = "paramsPressureBand";
    public static final String PARAMS_OVERLOAD_FACTOR_S = "paramsOverloadFactor";
    public static final String PARAMS_UNDERLOAD_FACTOR_S = "paramsUnderloadFactor";
    public static final String PARAMS_GAMMA_EXP_OVERLOAD_S = "paramsGammaExpOverload";
    public static final String PARAMS_GAMMA_EXP_UNDERLOAD_S = "paramsGammaExpUnderload";
    public static final String PARAMS_PINIT_BT_S = "paramsPInitBt";
    public static final String PARAMS_BETA_BT_S = "paramsBetaBt";
    public static final String PARAMS_GAMMA_BT_S = "paramsGammaBt";
    public static final String PARAMS_PINIT_WIFI_S = "paramsPInitWifi";
    public static final String PARAMS_BETA_WIFI_S = "paramsBetaWifi";
    public static final String PARAMS_GAMMA_WIFI_S = "paramsGammaWifi";
    public static final String PARAMS_PINIT_BACKHAUL_S = "paramsPInitBackhaul";
    public static final String PARAMS_BETA_BACKHAUL_S = "paramsBetaBackhaul";
    public static final String PARAMS_GAMMA_BACKHAUL_S = "paramsGammaBackhaul";
    public static final String HYBRID_W_PRESSURE_S = "hybridWPressure";
    public static final String HYBRID_W_PBASE_S = "hybridWPBase";
    public static final String HYBRID_W_ROLE_S = "hybridWRole";
    public static final String HYBRID_W_LOAD_S = "hybridWLoad";
    public static final String NET_W_MEAN_S = "netWMean";
    public static final String NET_W_PRESSURE_S = "netWPressure";
    private static final double SCHEDULE_TIME_EPS = 1e-7;

    private final String mode;
    private final double deltaValue;
    private double lowThreshold;
    private double highThreshold;
    private final int bufOccMaxAge;
    private final boolean logActions;
    private final int logActionsMax;
    private final double activationTime;
    private final double contactsCmax;
    private final double pressureScale;
    private final double pBaseThreshold;
    private final double roleThreshold;
    private final double carrierOffloadThreshold;
    private final double carrierOffloadScale;
    private final double rangeWifi;
    private final double rangeBackhaul;
    private final boolean paramsEnable;
    private final double paramsEmaBeta;
    private final double paramsPressureBand;
    private final double paramsOverloadFactor;
    private final double paramsUnderloadFactor;
    private final double paramsGammaExpOverload;
    private final double paramsGammaExpUnderload;
    private final double paramsPInitBt;
    private final double paramsBetaBt;
    private final double paramsGammaBt;
    private final double paramsPInitWifi;
    private final double paramsBetaWifi;
    private final double paramsGammaWifi;
    private final double paramsPInitBackhaul;
    private final double paramsBetaBackhaul;
    private final double paramsGammaBackhaul;
    private final double hybridWPressure;
    private final double hybridWPBase;
    private final double hybridWRole;
    private final double hybridWLoad;
    private final double netWMean;
    private final double netWPressure;
    private boolean activationNotified = false;
    private final List<ThresholdScheduleEntry> thresholdSchedule;
    private int scheduleCursor = 0;

    private final BufferOccupancyTracker tracker = new BufferOccupancyTracker();
    private final Map<Integer, double[]> lastParamsByHost = new HashMap<Integer, double[]>();

    public BufferLoadHeuristicReport() {
        super();
        final Settings s = getSettings();
        String m = s.getSetting(MODE_S, "category").trim().toLowerCase();
        if (!("category".equals(m) || "pressure".equals(m) || "net_pressure".equals(m) ||
                "pbase_pressure".equals(m) || "pbase_hybrid".equals(m) || "role_split".equals(m))) {
            m = "category";
        }
        this.mode = m;
        double configuredDelta = 0.05;
        if (s.contains(DELTA_VALUE_S)) {
            configuredDelta = s.getDouble(DELTA_VALUE_S);
        } else {
            Settings prophetSettings = new Settings(ProphetRouter.PROPHET_NS);
            configuredDelta = prophetSettings.getDouble(
                    ProphetRouter.HEURISTIC_DELTA_S, configuredDelta);
        }
        this.deltaValue = Math.max(0.0, configuredDelta);
        double low = s.getDouble(LOW_THRESHOLD_S, 0.33);
        double high = s.getDouble(HIGH_THRESHOLD_S, 0.66);
        if (low < 0.0) low = 0.0;
        if (low > 1.0) low = 1.0;
        if (high < 0.0) high = 0.0;
        if (high > 1.0) high = 1.0;
        if (low > high) {
            double tmp = low;
            low = high;
            high = tmp;
        }
        this.lowThreshold = low;
        this.highThreshold = high;
        int age = (int)Math.round(s.getDouble(BUF_OCC_MAX_AGE_S, getDefaultWindow()));
        if (age < 0) age = (int)getDefaultWindow();
        this.bufOccMaxAge = age;
        String logOpt = s.getSetting(LOG_ACTIONS_S, "false").toLowerCase();
        this.logActions = ("true".equals(logOpt) || "1".equals(logOpt) || "yes".equals(logOpt));
        this.logActionsMax = (int)Math.round(s.getDouble(LOG_ACTIONS_MAX_S, 20.0));
        this.activationTime = Math.max(0.0, s.getDouble(ACTIVATION_TIME_S, 0.0));
        double ccmax = s.getDouble(CONTACTS_CMAX_S, 10.0);
        if (!Double.isFinite(ccmax) || ccmax <= 0.0) { ccmax = 10.0; }
        this.contactsCmax = ccmax;
        double ps = s.getDouble(PRESSURE_SCALE_S, 0.15);
        if (!Double.isFinite(ps) || ps <= 1e-9) { ps = 0.15; }
        this.pressureScale = ps;
        double pb = s.getDouble(PBASE_THRESHOLD_S, 0.5);
        if (!Double.isFinite(pb)) { pb = 0.5; }
        this.pBaseThreshold = clamp01(pb);
        double rt = s.getDouble(ROLE_THRESHOLD_S, 0.6);
        if (!Double.isFinite(rt)) { rt = 0.6; }
        this.roleThreshold = clamp01(rt);
        double cot = s.getDouble(CARRIER_OFFLOAD_THRESHOLD_S, 0.85);
        if (!Double.isFinite(cot)) { cot = 0.85; }
        this.carrierOffloadThreshold = clamp01(cot);
        double cos = s.getDouble(CARRIER_OFFLOAD_SCALE_S, 1.0);
        if (!Double.isFinite(cos) || cos < 0.0) { cos = 1.0; }
        this.carrierOffloadScale = cos;
        double rw = s.getDouble(RANGE_WIFI_S, 50.0);
        if (!Double.isFinite(rw) || rw <= 0.0) { rw = 50.0; }
        this.rangeWifi = rw;
        double rb = s.getDouble(RANGE_BACKHAUL_S, 200.0);
        if (!Double.isFinite(rb) || rb <= 0.0) { rb = 200.0; }
        this.rangeBackhaul = rb;

        this.paramsEnable = s.getBoolean(PARAMS_ENABLE_S, false);
        double pe = s.getDouble(PARAMS_EMA_BETA_S, 0.85);
        if (!Double.isFinite(pe)) { pe = 0.85; }
        this.paramsEmaBeta = clamp01(pe);
        double ppb = s.getDouble(PARAMS_PRESSURE_BAND_S, 0.12);
        if (!Double.isFinite(ppb) || ppb < 0.0) { ppb = 0.12; }
        this.paramsPressureBand = clamp01(ppb);
        double pof = s.getDouble(PARAMS_OVERLOAD_FACTOR_S, 0.85);
        if (!Double.isFinite(pof) || pof <= 0.0) { pof = 0.85; }
        this.paramsOverloadFactor = pof;
        double puf = s.getDouble(PARAMS_UNDERLOAD_FACTOR_S, 1.05);
        if (!Double.isFinite(puf) || puf <= 0.0) { puf = 1.05; }
        this.paramsUnderloadFactor = puf;
        double go = s.getDouble(PARAMS_GAMMA_EXP_OVERLOAD_S, 1.0);
        if (!Double.isFinite(go) || go <= 0.0) { go = 1.0; }
        this.paramsGammaExpOverload = go;
        double gu = s.getDouble(PARAMS_GAMMA_EXP_UNDERLOAD_S, 1.0);
        if (!Double.isFinite(gu) || gu <= 0.0) { gu = 1.0; }
        this.paramsGammaExpUnderload = gu;
        this.paramsPInitBt = clamp01(s.getDouble(PARAMS_PINIT_BT_S, ProphetRouter.P_INIT));
        this.paramsBetaBt = clamp01(s.getDouble(PARAMS_BETA_BT_S, ProphetRouter.DEFAULT_BETA));
        this.paramsGammaBt = clampGamma(s.getDouble(PARAMS_GAMMA_BT_S, ProphetRouter.DEFAULT_GAMMA));
        this.paramsPInitWifi = clamp01(s.getDouble(PARAMS_PINIT_WIFI_S, 0.80));
        this.paramsBetaWifi = clamp01(s.getDouble(PARAMS_BETA_WIFI_S, 0.30));
        this.paramsGammaWifi = clampGamma(s.getDouble(PARAMS_GAMMA_WIFI_S, 0.985));
        this.paramsPInitBackhaul = clamp01(s.getDouble(PARAMS_PINIT_BACKHAUL_S, 0.90));
        this.paramsBetaBackhaul = clamp01(s.getDouble(PARAMS_BETA_BACKHAUL_S, 0.40));
        this.paramsGammaBackhaul = clampGamma(s.getDouble(PARAMS_GAMMA_BACKHAUL_S, 0.992));

        double wP = s.getDouble(HYBRID_W_PRESSURE_S, 0.8);
        if (!Double.isFinite(wP)) { wP = 0.8; }
        this.hybridWPressure = clamp(wP, -5.0, 5.0);
        double wB = s.getDouble(HYBRID_W_PBASE_S, 0.6);
        if (!Double.isFinite(wB)) { wB = 0.6; }
        this.hybridWPBase = clamp(wB, -5.0, 5.0);
        double wR = s.getDouble(HYBRID_W_ROLE_S, 0.2);
        if (!Double.isFinite(wR)) { wR = 0.2; }
        this.hybridWRole = clamp(wR, -5.0, 5.0);
        double wL = s.getDouble(HYBRID_W_LOAD_S, 0.8);
        if (!Double.isFinite(wL)) { wL = 0.8; }
        this.hybridWLoad = clamp(wL, -5.0, 5.0);

        double wNm = s.getDouble(NET_W_MEAN_S, 0.7);
        if (!Double.isFinite(wNm)) { wNm = 0.7; }
        this.netWMean = clamp(wNm, -5.0, 5.0);
        double wNp = s.getDouble(NET_W_PRESSURE_S, 0.3);
        if (!Double.isFinite(wNp)) { wNp = 0.3; }
        this.netWPressure = clamp(wNp, -5.0, 5.0);
        if (this.activationTime > 0.0) {
            write("# BufferLoadHeuristic inactive until t >= " + format(this.activationTime));
        }
        this.thresholdSchedule = parseThresholdSchedule(s);

        write("# BufferLoadHeuristic active mode=" + this.mode +
                " delta=" + format(this.deltaValue) +
                " low<= " + format(this.lowThreshold) +
                " high>= " + format(this.highThreshold) +
                " sampleInterval=" + format(super.interval) +
                " params=" + (this.paramsEnable ? "on" : "off"));
        if (!this.thresholdSchedule.isEmpty()) {
            write("# BufferLoadHeuristic threshold schedule entries=" + this.thresholdSchedule.size());
        }
    }

    @Override
    protected void sample(List<DTNHost> hosts) {
        if (hosts == null || hosts.isEmpty()) {
            return;
        }
        final int now = (int) SimClock.getTime();
        maybeAdvanceThresholdSchedule(now);
        try {
            tracker.update(hosts, now, this.bufOccMaxAge);
        } catch (Exception ignore) { /* best effort */ }

        final boolean active = (this.activationTime <= 0.0) || (now >= this.activationTime);
        if (!active) {
            return;
        }
        if (!this.activationNotified && this.activationTime > 0.0) {
            write("# BufferLoadHeuristic activation at t=" + now);
            this.activationNotified = true;
        }

        // Clear existing offsets before applying new ones
        for (DTNHost host : hosts) {
            MessageRouter router = host.getRouter();
            if (router instanceof ProphetRouter) {
                ((ProphetRouter) router).clearExternalOffsets();
            }
        }

        int appliedHosts = 0;
        int appliedTotal = 0;
        int logged = 0;
        long maxBufferSize = 0;
        double maxTransmitRange = 0.0;
        for (DTNHost host : hosts) {
            MessageRouter router = host.getRouter();
            if (router == null) { continue; }
            long size = router.getBufferSize();
            if (size > maxBufferSize) { maxBufferSize = size; }
            double r = getMaxTransmitRange(host);
            if (Double.isFinite(r) && r > maxTransmitRange) { maxTransmitRange = r; }
        }
        if (maxBufferSize <= 0) { maxBufferSize = 1; }
        if (!Double.isFinite(maxTransmitRange) || maxTransmitRange <= 0.0) { maxTransmitRange = 1.0; }

        for (DTNHost host : hosts) {
            MessageRouter router = host.getRouter();
            if (!(router instanceof ProphetRouter)) {
                continue; // heuristic only defined for Prophet variants
            }

            double occMean = tracker.getMeanOccupancy(host.getAddress());
            if (Double.isNaN(occMean)) {
                continue;
            }
            double selfOcc = computeOccupancy(host);
            if (Double.isNaN(selfOcc)) {
                selfOcc = occMean;
            }

            if (this.paramsEnable) {
                try {
                    maybeUpdateProphetParams(host, (ProphetRouter) router, selfOcc, occMean);
                } catch (Exception ignore) { /* best effort */ }
            }

            if ("pressure".equals(this.mode)) {
                double diff = selfOcc - occMean;
                double mag = Math.abs(diff);
                double scaled = mag / this.pressureScale;
                if (!Double.isFinite(scaled) || scaled < 0.0) { scaled = 0.0; }
                if (scaled > 1.0) { scaled = 1.0; }
                double delta = 0.0;
                if (mag > 1e-12 && scaled > 1e-12) {
                    delta = (diff > 0.0 ? -1.0 : 1.0) * this.deltaValue * scaled;
                }
                if (Math.abs(delta) <= 1e-12) {
                    continue;
                }

                Set<String> uniqueDests = new HashSet<String>();
                Map<String, DTNHost> destMap = new HashMap<String, DTNHost>();
                for (Message m : host.getMessageCollection()) {
                    String destStr = m.getTo().toString();
                    if (uniqueDests.add(destStr)) {
                        destMap.put(destStr, m.getTo());
                    }
                }
                if (uniqueDests.isEmpty()) {
                    continue;
                }
                for (String destStr : uniqueDests) {
                    DTNHost destHost = destMap.get(destStr);
                    if (destHost == null) { continue; }
                    ((ProphetRouter) router).setExternalOffset(destHost, delta);
                    appliedTotal++;
                    if (this.logActions && logged < this.logActionsMax) {
                        write(now + " H " + host + " -> " + destStr +
                                " mode=pressure occ=" + format(selfOcc) +
                                " mean=" + format(occMean) +
                                " delta=" + format(delta));
                        logged++;
                    }
                }
                appliedHosts++;
                continue;
            }

            if ("net_pressure".equals(this.mode)) {
                // network congestion term based on mean occupancy (global-ish)
                double mid = 0.5 * (this.lowThreshold + this.highThreshold);
                double halfRange = 0.5 * (this.highThreshold - this.lowThreshold);
                if (!Double.isFinite(halfRange) || halfRange < 1e-9) { halfRange = 0.33; }
                double netTerm = (occMean - mid) / halfRange;
                netTerm = clamp(netTerm, -1.0, 1.0);

                // self pressure term: + when underloaded, - when overloaded
                double pressureTerm = 0.0;
                if (this.pressureScale > 1e-9) {
                    pressureTerm = (occMean - selfOcc) / this.pressureScale;
                    pressureTerm = clamp(pressureTerm, -1.0, 1.0);
                }

                double score = this.netWMean * netTerm + this.netWPressure * pressureTerm;
                score = clamp(score, -1.0, 1.0);
                double delta = this.deltaValue * score;
                if (Math.abs(delta) <= 1e-12) {
                    continue;
                }

                Set<String> uniqueDests = new HashSet<String>();
                Map<String, DTNHost> destMap = new HashMap<String, DTNHost>();
                for (Message m2 : host.getMessageCollection()) {
                    String destStr = m2.getTo().toString();
                    if (uniqueDests.add(destStr)) {
                        destMap.put(destStr, m2.getTo());
                    }
                }
                if (uniqueDests.isEmpty()) {
                    continue;
                }

                for (String destStr : uniqueDests) {
                    DTNHost destHost = destMap.get(destStr);
                    if (destHost == null) { continue; }
                    ((ProphetRouter) router).setExternalOffset(destHost, delta);
                    appliedTotal++;
                    if (this.logActions && logged < this.logActionsMax) {
                        write(now + " H " + host + " -> " + destStr +
                                " mode=net_pressure occ=" + format(selfOcc) +
                                " mean=" + format(occMean) +
                                " delta=" + format(delta));
                        logged++;
                    }
                }
                appliedHosts++;
                continue;
            }

            if ("pbase_pressure".equals(this.mode)) {
                long bufferCapacity = router.getBufferSize();
                double capacityNorm = ((double) bufferCapacity) / (double) maxBufferSize;
                if (!Double.isFinite(capacityNorm) || capacityNorm < 0.0) { capacityNorm = 0.0; }
                if (capacityNorm > 1.0) { capacityNorm = 1.0; }

                int contactsNow = 0;
                try { contactsNow = host.getConnections().size(); } catch (Exception ignore) { contactsNow = 0; }
                if (contactsNow < 0) { contactsNow = 0; }
                double contactsNorm = ((double) contactsNow) / this.contactsCmax;
                if (!Double.isFinite(contactsNorm) || contactsNorm < 0.0) { contactsNorm = 0.0; }
                if (contactsNorm > 1.0) { contactsNorm = 1.0; }

                double role = 0.6 * capacityNorm + 0.4 * contactsNorm; // [0,1]
                double roleTerm = clamp(-1.0, 1.0, (role - 0.5) * 2.0); // [-1,1]
                double pressure = clamp01(Math.max(0.0, selfOcc - occMean)); // [0,1]

                int totalMsgs = 0;
                Map<String, Integer> msgCountByDest = new HashMap<String, Integer>();
                Map<String, Long> bytesSumByDest = new HashMap<String, Long>();
                Set<String> uniqueDests = new HashSet<String>();
                Map<String, DTNHost> destMap = new HashMap<String, DTNHost>();
                for (Message m : host.getMessageCollection()) {
                    totalMsgs++;
                    String destStr = m.getTo().toString();
                    uniqueDests.add(destStr);
                    if (!destMap.containsKey(destStr)) {
                        destMap.put(destStr, m.getTo());
                    }
                    Integer prevCnt = msgCountByDest.get(destStr);
                    msgCountByDest.put(destStr, (prevCnt == null ? 1 : prevCnt + 1));
                    long sz = 0L;
                    try { sz = (long) m.getSize(); } catch (Exception ignore) { sz = 0L; }
                    if (sz < 0L) { sz = 0L; }
                    Long prevSum = bytesSumByDest.get(destStr);
                    bytesSumByDest.put(destStr, (prevSum == null ? sz : prevSum.longValue() + sz));
                }
                if (uniqueDests.isEmpty()) {
                    continue;
                }
                double denomMsgs = (totalMsgs > 0 ? (double) totalMsgs : 1.0);
                double denomBytes = (bufferCapacity > 0 ? (double) bufferCapacity : 1.0);

                for (String destStr : uniqueDests) {
                    DTNHost destHost = destMap.get(destStr);
                    if (destHost == null) { continue; }
                    double pBase = ((ProphetRouter) router).getBasePredFor(destHost);
                    if (!Double.isFinite(pBase)) { pBase = 0.0; }
                    if (pBase < 0.0) { pBase = 0.0; }
                    if (pBase > 1.0) { pBase = 1.0; }
                    double pTerm = clamp(-1.0, 1.0, (pBase - this.pBaseThreshold) * 2.0); // [-1,1]

                    int msgCnt = 0;
                    Integer mc = msgCountByDest.get(destStr);
                    if (mc != null) { msgCnt = mc.intValue(); }
                    if (msgCnt < 0) { msgCnt = 0; }
                    double nMsgsNorm = ((double) msgCnt) / denomMsgs;
                    if (!Double.isFinite(nMsgsNorm) || nMsgsNorm < 0.0) { nMsgsNorm = 0.0; }
                    if (nMsgsNorm > 1.0) { nMsgsNorm = 1.0; }

                    long bSum = 0L;
                    Long bs = bytesSumByDest.get(destStr);
                    if (bs != null) { bSum = bs.longValue(); }
                    if (bSum < 0L) { bSum = 0L; }
                    double bytesNorm = ((double) bSum) / denomBytes;
                    if (!Double.isFinite(bytesNorm) || bytesNorm < 0.0) { bytesNorm = 0.0; }
                    if (bytesNorm > 1.0) { bytesNorm = 1.0; }

                    double load = clamp01(0.7 * bytesNorm + 0.3 * nMsgsNorm); // [0,1]

                    // Positive delta => selfPred↑ => forward less (carry).
                    // Negative delta => selfPred↓ => forward more (offload).
                    double score = 0.6 * pTerm + 0.3 * roleTerm - 0.8 * pressure - 0.8 * load;
                    score = clamp(-1.0, 1.0, score);
                    double delta = this.deltaValue * score;
                    if (Math.abs(delta) <= 1e-12) {
                        continue;
                    }

                    ((ProphetRouter) router).setExternalOffset(destHost, delta);
                    appliedTotal++;
                    if (this.logActions && logged < this.logActionsMax) {
                        write(now + " H " + host + " -> " + destStr +
                                " mode=pbase_pressure p=" + format(pBase) +
                                " occ=" + format(selfOcc) +
                                " mean=" + format(occMean) +
                                " load=" + format(load) +
                                " delta=" + format(delta));
                        logged++;
                    }
                }
                appliedHosts++;
                continue;
            }

            if ("pbase_hybrid".equals(this.mode)) {
                long bufferCapacity = router.getBufferSize();
                double capacityNorm = ((double) bufferCapacity) / (double) maxBufferSize;
                if (!Double.isFinite(capacityNorm) || capacityNorm < 0.0) { capacityNorm = 0.0; }
                if (capacityNorm > 1.0) { capacityNorm = 1.0; }

                int contactsNow = 0;
                try { contactsNow = host.getConnections().size(); } catch (Exception ignore) { contactsNow = 0; }
                if (contactsNow < 0) { contactsNow = 0; }
                double contactsNorm = ((double) contactsNow) / this.contactsCmax;
                if (!Double.isFinite(contactsNorm) || contactsNorm < 0.0) { contactsNorm = 0.0; }
                if (contactsNorm > 1.0) { contactsNorm = 1.0; }

                double role = 0.6 * capacityNorm + 0.4 * contactsNorm; // [0,1]
                double roleTerm = clamp((role - 0.5) * 2.0, -1.0, 1.0);  // [-1,1]

                // pressure term: + when underloaded (selfOcc < mean), - when overloaded
                double pressureTerm = 0.0;
                double diff = selfOcc - occMean;
                if (Double.isFinite(diff) && this.pressureScale > 1e-9) {
                    pressureTerm = clamp((occMean - selfOcc) / this.pressureScale, -1.0, 1.0);
                }

                int totalMsgs = 0;
                Map<String, Integer> msgCountByDest = new HashMap<String, Integer>();
                Map<String, Long> bytesSumByDest = new HashMap<String, Long>();
                Set<String> uniqueDests = new HashSet<String>();
                Map<String, DTNHost> destMap = new HashMap<String, DTNHost>();
                for (Message m : host.getMessageCollection()) {
                    totalMsgs++;
                    String destStr = m.getTo().toString();
                    uniqueDests.add(destStr);
                    if (!destMap.containsKey(destStr)) {
                        destMap.put(destStr, m.getTo());
                    }
                    Integer prevCnt = msgCountByDest.get(destStr);
                    msgCountByDest.put(destStr, (prevCnt == null ? 1 : prevCnt + 1));
                    long sz = 0L;
                    try { sz = (long) m.getSize(); } catch (Exception ignore) { sz = 0L; }
                    if (sz < 0L) { sz = 0L; }
                    Long prevSum = bytesSumByDest.get(destStr);
                    bytesSumByDest.put(destStr, (prevSum == null ? sz : prevSum.longValue() + sz));
                }
                if (uniqueDests.isEmpty()) {
                    continue;
                }

                double denomMsgs = (totalMsgs > 0 ? (double) totalMsgs : 1.0);
                double denomBytes = (bufferCapacity > 0 ? (double) bufferCapacity : 1.0);

                for (String destStr : uniqueDests) {
                    DTNHost destHost = destMap.get(destStr);
                    if (destHost == null) { continue; }
                    double pBase = ((ProphetRouter) router).getBasePredFor(destHost);
                    if (!Double.isFinite(pBase)) { pBase = 0.0; }
                    if (pBase < 0.0) { pBase = 0.0; }
                    if (pBase > 1.0) { pBase = 1.0; }
                    double pTerm = clamp((pBase - this.pBaseThreshold) * 2.0, -1.0, 1.0); // [-1,1]

                    int msgCnt = 0;
                    Integer mc = msgCountByDest.get(destStr);
                    if (mc != null) { msgCnt = mc.intValue(); }
                    if (msgCnt < 0) { msgCnt = 0; }
                    double nMsgsNorm = ((double) msgCnt) / denomMsgs;
                    if (!Double.isFinite(nMsgsNorm) || nMsgsNorm < 0.0) { nMsgsNorm = 0.0; }
                    if (nMsgsNorm > 1.0) { nMsgsNorm = 1.0; }

                    long bSum = 0L;
                    Long bs = bytesSumByDest.get(destStr);
                    if (bs != null) { bSum = bs.longValue(); }
                    if (bSum < 0L) { bSum = 0L; }
                    double bytesNorm = ((double) bSum) / denomBytes;
                    if (!Double.isFinite(bytesNorm) || bytesNorm < 0.0) { bytesNorm = 0.0; }
                    if (bytesNorm > 1.0) { bytesNorm = 1.0; }

                    double load = clamp01(0.7 * bytesNorm + 0.3 * nMsgsNorm); // [0,1]
                    double loadTerm = clamp((0.5 - load) * 2.0, -1.0, 1.0);    // [-1,1]

                    double score = this.hybridWPressure * pressureTerm +
                            this.hybridWPBase * pTerm +
                            this.hybridWRole * roleTerm +
                            this.hybridWLoad * loadTerm;
                    score = clamp(score, -1.0, 1.0);
                    double delta = this.deltaValue * score;
                    if (Math.abs(delta) <= 1e-12) {
                        continue;
                    }

                    ((ProphetRouter) router).setExternalOffset(destHost, delta);
                    appliedTotal++;
                    if (this.logActions && logged < this.logActionsMax) {
                        write(now + " H " + host + " -> " + destStr +
                                " mode=pbase_hybrid p=" + format(pBase) +
                                " occ=" + format(selfOcc) +
                                " mean=" + format(occMean) +
                                " load=" + format(load) +
                                " delta=" + format(delta));
                        logged++;
                    }
                }
                appliedHosts++;
                continue;
            }

            if ("role_split".equals(this.mode)) {
                long bufferCapacity = router.getBufferSize();
                double capacityNorm = ((double) bufferCapacity) / (double) maxBufferSize;
                if (!Double.isFinite(capacityNorm) || capacityNorm < 0.0) { capacityNorm = 0.0; }
                if (capacityNorm > 1.0) { capacityNorm = 1.0; }

                double hostRange = getMaxTransmitRange(host);
                if (!Double.isFinite(hostRange) || hostRange < 0.0) { hostRange = 0.0; }
                double rangeNorm = hostRange / maxTransmitRange;
                if (!Double.isFinite(rangeNorm) || rangeNorm < 0.0) { rangeNorm = 0.0; }
                if (rangeNorm > 1.0) { rangeNorm = 1.0; }

                // Role score favors interface reach and capacity.
                double role = 0.7 * rangeNorm + 0.3 * capacityNorm; // [0,1]
                boolean carrier = role >= this.roleThreshold;
                double delta = carrier ? this.deltaValue : -this.deltaValue;
                if (carrier && selfOcc >= this.carrierOffloadThreshold) {
                    delta = -this.deltaValue * this.carrierOffloadScale;
                }
                if (Math.abs(delta) <= 1e-12) {
                    continue;
                }

                Set<String> uniqueDests = new HashSet<String>();
                Map<String, DTNHost> destMap = new HashMap<String, DTNHost>();
                for (Message m : host.getMessageCollection()) {
                    String destStr = m.getTo().toString();
                    if (uniqueDests.add(destStr)) {
                        destMap.put(destStr, m.getTo());
                    }
                }
                if (uniqueDests.isEmpty()) {
                    continue;
                }
                for (String destStr : uniqueDests) {
                    DTNHost destHost = destMap.get(destStr);
                    if (destHost == null) { continue; }
                    ((ProphetRouter) router).setExternalOffset(destHost, delta);
                    appliedTotal++;
                    if (this.logActions && logged < this.logActionsMax) {
                        write(now + " H " + host + " -> " + destStr +
                                " mode=role_split role=" + format(role) +
                                " occ=" + format(selfOcc) +
                                " mean=" + format(occMean) +
                                " delta=" + format(delta));
                        logged++;
                    }
                }
                appliedHosts++;
                continue;
            }

            double delta;
            String category;
            if (occMean <= this.lowThreshold) {
                delta = -this.deltaValue;
                category = "low";
            } else if (occMean >= this.highThreshold) {
                delta = this.deltaValue;
                category = "high";
            } else {
                delta = 0.0;
                category = "medium";
            }

            if (delta == 0.0) {
                if (this.logActions && logged < this.logActionsMax) {
                    write(now + " H " + host + " cat=" + category + " occ=" + format(occMean) + " delta=0");
                    logged++;
                }
                continue; // nothing to apply
            }

            Set<String> uniqueDests = new HashSet<String>();
            Map<String, DTNHost> destMap = new HashMap<String, DTNHost>();
            for (Message m : host.getMessageCollection()) {
                String destStr = m.getTo().toString();
                if (uniqueDests.add(destStr)) {
                    destMap.put(destStr, m.getTo());
                }
            }

            if (uniqueDests.isEmpty()) {
                continue;
            }

            for (String destStr : uniqueDests) {
                DTNHost destHost = destMap.get(destStr);
                if (destHost == null) { continue; }
                if (router instanceof ProphetRouter) {
                    ((ProphetRouter) router).setExternalOffset(destHost, delta);
                }
                appliedTotal++;
                if (this.logActions && logged < this.logActionsMax) {
                    write(now + " H " + host + " -> " + destStr + " cat=" + category + " occ=" + format(occMean) + " delta=" + format(delta));
                    logged++;
                }
            }
            appliedHosts++;
        }

        write("# BufferLoadHeuristic t=" + now + " hosts=" + appliedHosts + " applied_offsets=" + appliedTotal);
    }

    private void maybeUpdateProphetParams(DTNHost host, ProphetRouter router, double selfOcc, double occMean) {
        if (host == null || router == null) {
            return;
        }
        double maxRange = getMaxTransmitRange(host);
        if (!Double.isFinite(maxRange)) { maxRange = 0.0; }

        double basePInit;
        double baseBeta;
        double baseGamma;
        if (maxRange >= this.rangeBackhaul) {
            basePInit = this.paramsPInitBackhaul;
            baseBeta = this.paramsBetaBackhaul;
            baseGamma = this.paramsGammaBackhaul;
        } else if (maxRange >= this.rangeWifi) {
            basePInit = this.paramsPInitWifi;
            baseBeta = this.paramsBetaWifi;
            baseGamma = this.paramsGammaWifi;
        } else {
            basePInit = this.paramsPInitBt;
            baseBeta = this.paramsBetaBt;
            baseGamma = this.paramsGammaBt;
        }

        double diff = selfOcc - occMean;
        if (!Double.isFinite(diff)) { diff = 0.0; }
        double factor = 1.0;
        if (diff >= this.paramsPressureBand) {
            factor = this.paramsOverloadFactor;
        } else if (diff <= -this.paramsPressureBand) {
            factor = this.paramsUnderloadFactor;
        }
        double targetPInit = clamp01(basePInit * factor);
        double targetBeta = clamp01(baseBeta * factor);
        double gammaExp = 1.0;
        if (diff >= this.paramsPressureBand) {
            gammaExp = this.paramsGammaExpOverload;
        } else if (diff <= -this.paramsPressureBand) {
            gammaExp = this.paramsGammaExpUnderload;
        }
        if (!Double.isFinite(gammaExp) || gammaExp <= 0.0) {
            gammaExp = 1.0;
        }
        double targetGamma = clampGamma(Math.pow(baseGamma, gammaExp));

        int addr = host.getAddress();
        double[] prev = this.lastParamsByHost.get(addr);
        double prevP = (prev != null ? prev[0] : router.getPInit());
        double prevB = (prev != null ? prev[1] : router.getBeta());
        double prevG = (prev != null ? prev[2] : router.getGamma());

        double pNew = ema(prevP, targetPInit, this.paramsEmaBeta);
        double bNew = ema(prevB, targetBeta, this.paramsEmaBeta);
        double gNew = ema(prevG, targetGamma, this.paramsEmaBeta);
        router.setPInit(pNew);
        router.setBeta(bNew);
        router.setGamma(gNew);
        this.lastParamsByHost.put(addr, new double[]{pNew, bNew, gNew});
    }

    private double getMaxTransmitRange(DTNHost host) {
        if (host == null) {
            return Double.NaN;
        }
        double max = 0.0;
        try {
            List<NetworkInterface> ifs = host.getInterfaces();
            if (ifs == null || ifs.isEmpty()) {
                return 0.0;
            }
            for (NetworkInterface ni : ifs) {
                if (ni == null) {
                    continue;
                }
                double r = ni.getTransmitRange();
                if (Double.isFinite(r) && r > max) {
                    max = r;
                }
            }
        } catch (Exception ignore) {
            return Double.NaN;
        }
        return max;
    }

    private double ema(double prev, double x, double beta) {
        double b = clamp01(beta);
        if (!Double.isFinite(prev)) { prev = 0.0; }
        if (!Double.isFinite(x)) { x = 0.0; }
        return prev * b + x * (1.0 - b);
    }

    private double computeOccupancy(DTNHost host) {
        if (host == null) {
            return Double.NaN;
        }
        MessageRouter router = host.getRouter();
        if (router == null) {
            return Double.NaN;
        }
        long size = router.getBufferSize();
        if (size <= 0 || size >= Integer.MAX_VALUE) {
            return Double.NaN;
        }
        long free = router.getFreeBufferSize();
        double occ = 1.0 - ((double) free / (double) size);
        return clamp01(occ);
    }

    private double clamp(double value, double lo, double hi) {
        if (Double.isNaN(value)) {
            return value;
        }
        if (value < lo) {
            return lo;
        }
        if (value > hi) {
            return hi;
        }
        return value;
    }

    private double getDefaultWindow() {
        // fall back to 600s like RL bridge/state reports if interval not exposed
        return 600.0;
    }

    private void maybeAdvanceThresholdSchedule(int now) {
        if (this.thresholdSchedule.isEmpty()) {
            return;
        }
        boolean updated = false;
        while (this.scheduleCursor < this.thresholdSchedule.size()) {
            ThresholdScheduleEntry entry = this.thresholdSchedule.get(this.scheduleCursor);
            if ((double) now + SCHEDULE_TIME_EPS < entry.activationTime) {
                break;
            }
            this.lowThreshold = entry.low;
            this.highThreshold = entry.high;
            this.scheduleCursor++;
            updated = true;
        }
        if (updated) {
            write("# BufferLoadHeuristic thresholds updated at t=" + now +
                    " low<= " + format(this.lowThreshold) +
                    " high>= " + format(this.highThreshold));
        }
    }

    private List<ThresholdScheduleEntry> parseThresholdSchedule(Settings s) {
        List<ThresholdScheduleEntry> entries = new ArrayList<ThresholdScheduleEntry>();
        if (!s.contains(THRESHOLD_SCHEDULE_S)) {
            return entries;
        }
        String[] tokens = s.getCsvSetting(THRESHOLD_SCHEDULE_S);
        for (String token : tokens) {
            if (token == null || token.isEmpty()) {
                continue;
            }
            String[] fields = token.split(":");
            if (fields.length != 3) {
                throw new SettingsError("Invalid thresholdSchedule entry '" + token +
                        "'. Expected format time:low:high");
            }
            double activation = parseDouble(fields[0], "activationTime");
            double low = parseDouble(fields[1], "lowThreshold");
            double high = parseDouble(fields[2], "highThreshold");
            double[] normalized = normalizeThresholds(low, high);
            entries.add(new ThresholdScheduleEntry(
                    Math.max(0.0, activation),
                    normalized[0],
                    normalized[1]));
        }
        Collections.sort(entries);
        return entries;
    }

    private double parseDouble(String raw, String name) {
        try {
            return Double.parseDouble(raw.trim());
        } catch (NumberFormatException nfe) {
            throw new SettingsError("Invalid numeric value '" + raw + "' for " + name, nfe);
        }
    }

    private double[] normalizeThresholds(double low, double high) {
        double normalizedLow = clamp01(low);
        double normalizedHigh = clamp01(high);
        if (normalizedLow > normalizedHigh) {
            double tmp = normalizedLow;
            normalizedLow = normalizedHigh;
            normalizedHigh = tmp;
        }
        return new double[]{normalizedLow, normalizedHigh};
    }

    private double clamp01(double value) {
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }

    private double clampGamma(double gamma) {
        if (!Double.isFinite(gamma)) {
            return 1e-6;
        }
        if (gamma <= 0.0) {
            return 1e-6;
        }
        if (gamma > 1.0) {
            return 1.0;
        }
        return gamma;
    }

    private static final class ThresholdScheduleEntry implements Comparable<ThresholdScheduleEntry> {
        private final double activationTime;
        private final double low;
        private final double high;

        private ThresholdScheduleEntry(double activationTime, double low, double high) {
            this.activationTime = activationTime;
            this.low = low;
            this.high = high;
        }

        @Override
        public int compareTo(ThresholdScheduleEntry other) {
            if (this.activationTime == other.activationTime) {
                return 0;
            }
            return this.activationTime < other.activationTime ? -1 : 1;
        }
    }
}
