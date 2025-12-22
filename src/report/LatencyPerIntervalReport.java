/*
 * Reports time-binned (interval-based) average message delivery latency.
 * For each interval, tracks messages created in that interval and calculates
 * average latency (delivery_time - creation_time) for delivered messages.
 */
package report;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import core.DTNHost;
import core.Message;
import core.MessageListener;
import core.Settings;

public class LatencyPerIntervalReport extends Report implements MessageListener {

    /** Interval (seconds) for time bins (default 3600 = 1 hour). */
    public static final String BIN_SIZE_S = "binSize";
    /** Enables cumulative aggregation across time bins. */
    public static final String CUMULATIVE_MODE_S = "cumulativeMode";

    private int binSize;
    private Map<String, Integer> msgBinById;
    private Map<Integer, List<Double>> latenciesByBin;
    private boolean cumulativeMode;

    public LatencyPerIntervalReport() {
        init();
    }

    @Override
    protected void init() {
        super.init();
        Settings s = getSettings();
        this.binSize = (int) Math.round(s.getDouble(BIN_SIZE_S, 3600));
        if (this.binSize <= 0) {
            this.binSize = 3600;
        }
        this.msgBinById = new HashMap<String, Integer>();
        this.latenciesByBin = new HashMap<Integer, List<Double>>();
        this.cumulativeMode = parseBooleanSetting(s.getSetting(CUMULATIVE_MODE_S, "false"));
        write("# start end delivered avg_latency min_latency max_latency");
        write("# cumulative_mode=" + this.cumulativeMode);
    }

    private int timeToBin(double t) {
        return (int) Math.floor(t / this.binSize);
    }

    private boolean parseBooleanSetting(String value) {
        String v = (value == null ? "" : value.trim().toLowerCase());
        return ("true".equals(v) || "1".equals(v) || "yes".equals(v));
    }

    @Override
    public void newMessage(Message m) {
        if (isWarmup()) {
            addWarmupID(m.getId());
            return;
        }
        int bin = timeToBin(m.getCreationTime());
        msgBinById.put(m.getId(), bin);
    }

    @Override
    public void messageTransferred(Message m, DTNHost from, DTNHost to, boolean firstDelivery) {
        if (isWarmupID(m.getId())) {
            return;
        }
        if (!firstDelivery) {
            return;
        }
        Integer bin = msgBinById.get(m.getId());
        if (bin == null) {
            // Fallback from creation time if mapping is missing
            bin = timeToBin(m.getCreationTime());
        }
        double latency = getSimTime() - m.getCreationTime();
        if (!latenciesByBin.containsKey(bin)) {
            latenciesByBin.put(bin, new ArrayList<Double>());
        }
        latenciesByBin.get(bin).add(latency);
    }

    // Unused callbacks
    public void messageDeleted(Message m, DTNHost where, boolean dropped) {}
    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) {}
    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) {}

    @Override
    public void done() {
        // Output bins in order
        Set<Integer> bins = new TreeSet<Integer>(latenciesByBin.keySet());
        List<Double> cumulativeLatencies = new ArrayList<Double>();

        for (Integer b : bins) {
            List<Double> latencies = latenciesByBin.get(b);
            List<Double> outputLatencies = latencies;

            if (this.cumulativeMode) {
                cumulativeLatencies.addAll(latencies);
                outputLatencies = cumulativeLatencies;
            }

            if (outputLatencies.isEmpty()) {
                continue;
            }

            // Calculate statistics
            double sum = 0.0;
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (Double lat : outputLatencies) {
                sum += lat;
                if (lat < min) min = lat;
                if (lat > max) max = lat;
            }
            double avg = sum / outputLatencies.size();

            int start = b * binSize;
            int end = start + binSize;
            write(start + " " + end + " " + outputLatencies.size() + " " +
                  format(avg) + " " + format(min) + " " + format(max));
        }
        super.done();
    }
}
