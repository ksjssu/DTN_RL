package report;


import core.DTNHost;
import core.Message;
import core.Settings;
import core.SettingsError;
import core.SimClock;
import core.UpdateListener;
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
    public static final String THRESHOLD_SCHEDULE_S = "thresholdSchedule";
    private static final double SCHEDULE_TIME_EPS = 1e-7;

    private final double deltaValue;
    private double lowThreshold;
    private double highThreshold;
    private final int bufOccMaxAge;
    private final boolean logActions;
    private final int logActionsMax;
    private final double activationTime;
    private boolean activationNotified = false;
    private final List<ThresholdScheduleEntry> thresholdSchedule;
    private int scheduleCursor = 0;

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
        this.thresholdSchedule = parseThresholdSchedule(s);

        write("# BufferLoadHeuristic active delta=" + format(this.deltaValue) +
                " low<= " + format(this.lowThreshold) +
                " high>= " + format(this.highThreshold) +
                " sampleInterval=" + format(super.interval));
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
