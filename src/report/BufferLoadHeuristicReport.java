package report;


import core.DTNHost;
import core.Message;
import core.Settings;
import core.SimClock;
import core.UpdateListener;
import routing.MessageRouter;
import routing.ProphetRouter;





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
 * Categories:
 *  - low (<= lowThreshold): decrease own predictability by delta (encourage forwarding).
 *  - medium (between lowThreshold and highThreshold): no change.
 *  - high (>= highThreshold): increase own predictability by delta (discourage forwarding).
 */
public class BufferLoadHeuristicReport extends SamplingReport implements UpdateListener {

    public static final String DELTA_VALUE_S = "deltaValue";
    public static final String LOW_THRESHOLD_S = "lowThreshold";
    public static final String HIGH_THRESHOLD_S = "highThreshold";
    public static final String BUF_OCC_MAX_AGE_S = "bufOccMaxAge";
    public static final String LOG_ACTIONS_S = "logActions";
    public static final String LOG_ACTIONS_MAX_S = "logActionsMax";
    public static final String ACTIVATION_TIME_S = "activationTime";

    private final double deltaValue;
    private final double lowThreshold;
    private final double highThreshold;
    private final int bufOccMaxAge;
    private final boolean logActions;
    private final int logActionsMax;
    private final double activationTime;
    private boolean activationNotified = false;

    private final BufferOccupancyTracker tracker = new BufferOccupancyTracker();

    public BufferLoadHeuristicReport() {
        super();
        final Settings s = getSettings();
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
        if (this.activationTime > 0.0) {
            write("# BufferLoadHeuristic inactive until t >= " + format(this.activationTime));
        }

        write("# BufferLoadHeuristic active delta=" + format(this.deltaValue) +
                " low<= " + format(this.lowThreshold) +
                " high>= " + format(this.highThreshold) +
                " sampleInterval=" + format(super.interval));
    }

    @Override
    protected void sample(List<DTNHost> hosts) {
        if (hosts == null || hosts.isEmpty()) {
            return;
        }
        final int now = (int) SimClock.getTime();
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

        for (DTNHost host : hosts) {
            MessageRouter router = host.getRouter();
            if (!(router instanceof ProphetRouter)) {
                continue; // heuristic only defined for Prophet variants
            }

            double occMean = tracker.getMeanOccupancy(host.getAddress());
            if (Double.isNaN(occMean)) {
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

    private double getDefaultWindow() {
        // fall back to 600s like RL bridge/state reports if interval not exposed
        return 600.0;
    }
}

