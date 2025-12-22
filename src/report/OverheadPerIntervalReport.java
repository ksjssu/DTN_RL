/*
 * Reports time-binned (interval-based) overhead ratio.
 * For each interval, tracks the total number of message transmissions
 * and the number of delivered messages, then outputs overhead = transmissions / delivered.
 * Lower overhead means more efficient routing.
 */
package report;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import core.DTNHost;
import core.Message;
import core.MessageListener;
import core.Settings;

public class OverheadPerIntervalReport extends Report implements MessageListener {

    /** Interval (seconds) for time bins (default 3600 = 1 hour). */
    public static final String BIN_SIZE_S = "binSize";
    /** Enables cumulative aggregation across time bins. */
    public static final String CUMULATIVE_MODE_S = "cumulativeMode";

    private int binSize;
    private Map<String, Integer> msgBinById;
    private Map<Integer, Integer> transmissionsByBin;
    private Map<Integer, Integer> deliveredByBin;
    private boolean cumulativeMode;

    public OverheadPerIntervalReport() {
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
        this.transmissionsByBin = new HashMap<Integer, Integer>();
        this.deliveredByBin = new HashMap<Integer, Integer>();
        this.cumulativeMode = parseBooleanSetting(s.getSetting(CUMULATIVE_MODE_S, "false"));
        write("# start end transmissions delivered overhead");
        write("# cumulative_mode=" + this.cumulativeMode);
    }

    private int timeToBin(double t) {
        return (int) Math.floor(t / this.binSize);
    }

    private boolean parseBooleanSetting(String value) {
        String v = (value == null ? "" : value.trim().toLowerCase());
        return ("true".equals(v) || "1".equals(v) || "yes".equals(v));
    }

    private void inc(Map<Integer, Integer> map, int key) {
        Integer v = map.get(key);
        map.put(key, (v == null ? 1 : v + 1));
    }

    @Override
    public void newMessage(Message m) {
        if (isWarmup()) {
            addWarmupID(m.getId());
            return;
        }
        int bin = timeToBin(getSimTime());
        msgBinById.put(m.getId(), bin);
    }

    @Override
    public void messageTransferred(Message m, DTNHost from, DTNHost to, boolean firstDelivery) {
        if (isWarmupID(m.getId())) {
            return;
        }
        // Count every transmission (relay or delivery)
        int bin = timeToBin(getSimTime());
        inc(transmissionsByBin, bin);

        // Count only first deliveries
        if (firstDelivery) {
            inc(deliveredByBin, bin);
        }
    }

    // Unused callbacks
    public void messageDeleted(Message m, DTNHost where, boolean dropped) {}
    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) {}
    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) {}

    @Override
    public void done() {
        // Output bins in order
        Set<Integer> bins = new TreeSet<Integer>();
        bins.addAll(transmissionsByBin.keySet());
        bins.addAll(deliveredByBin.keySet());
        int cumulativeTransmissions = 0;
        int cumulativeDelivered = 0;
        for (Integer b : bins) {
            int transmissions = transmissionsByBin.containsKey(b) ? transmissionsByBin.get(b) : 0;
            int delivered = deliveredByBin.containsKey(b) ? deliveredByBin.get(b) : 0;
            int outputTransmissions = transmissions;
            int outputDelivered = delivered;
            if (this.cumulativeMode) {
                cumulativeTransmissions += transmissions;
                cumulativeDelivered += delivered;
                outputTransmissions = cumulativeTransmissions;
                outputDelivered = cumulativeDelivered;
            }
            double overhead = (outputDelivered > 0) ? ((double) outputTransmissions) / outputDelivered : Double.NaN;
            int start = b * binSize;
            int end = start + binSize;
            write(start + " " + end + " " + outputTransmissions + " " + outputDelivered + " " + format(overhead));
        }
        super.done();
    }
}
