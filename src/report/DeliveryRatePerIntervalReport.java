/*
 * Reports time-binned (interval-based) delivery success rate.
 * For each interval, counts messages created in that interval and how many of
 * those were eventually delivered (first delivery to final destination),
 * then outputs delivered/created per interval.
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

public class DeliveryRatePerIntervalReport extends Report implements MessageListener {

    /** Interval (seconds) for time bins (default 3600 = 1 hour). */
    public static final String BIN_SIZE_S = "binSize";
    /** Enables cumulative aggregation across time bins. */
    public static final String CUMULATIVE_MODE_S = "cumulativeMode";

    private int binSize;
    private Map<String, Integer> msgBinById;
    private Map<Integer, Integer> createdByBin;
    private Map<Integer, Integer> deliveredByBin;
    private boolean cumulativeMode;

    public DeliveryRatePerIntervalReport() {
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
        this.createdByBin = new HashMap<Integer, Integer>();
        this.deliveredByBin = new HashMap<Integer, Integer>();
        this.cumulativeMode = parseBooleanSetting(s.getSetting(CUMULATIVE_MODE_S, "false"));
        write("# start end created delivered success_rate");
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
        inc(createdByBin, bin);
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
        inc(deliveredByBin, bin);
    }

    // Unused callbacks
    public void messageDeleted(Message m, DTNHost where, boolean dropped) {}
    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) {}
    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) {}

    @Override
    public void done() {
        // Output bins in order
        Set<Integer> bins = new TreeSet<Integer>();
        bins.addAll(createdByBin.keySet());
        bins.addAll(deliveredByBin.keySet());
        int cumulativeCreated = 0;
        int cumulativeDelivered = 0;
        for (Integer b : bins) {
            int created = createdByBin.containsKey(b) ? createdByBin.get(b) : 0;
            int delivered = deliveredByBin.containsKey(b) ? deliveredByBin.get(b) : 0;
            int outputCreated = created;
            int outputDelivered = delivered;
            if (this.cumulativeMode) {
                cumulativeCreated += created;
                cumulativeDelivered += delivered;
                outputCreated = cumulativeCreated;
                outputDelivered = cumulativeDelivered;
            }
            double rate = (outputCreated > 0) ? ((double) outputDelivered) / outputCreated : Double.NaN;
            int start = b * binSize;
            int end = start + binSize;
            write(start + " " + end + " " + outputCreated + " " + outputDelivered + " " + format(rate));
        }
        super.done();
    }
}

