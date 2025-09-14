package report;

import core.DTNHost;
import core.Message;
import core.MessageListener;
import core.Settings;
import core.SimClock;

import java.util.List;

/**
 * Step-binned metrics report (per N sampling steps).
 * Outputs, every binSteps samples, the average delivery success rate,
 * overhead, and average delivery delay over that bin.
 *
 * Definitions per bin [t0, t1]:
 *  - created: number of messages created in the bin
 *  - delivered: first deliveries in the bin
 *  - relayed: hop-level successful transfers in the bin (all messageTransferred events)
 *  - rate: delivered / created (NaN if created == 0)
 *  - overhead: (relayed - delivered) / delivered (NaN if delivered == 0)
 *  - avg_delay: mean of (deliveryTime - creationTime) for first deliveries in the bin (seconds)
 *
 * The sampling period is controlled by sampleInterval (seconds), and bin size by binSteps.
 */
public class StepBinMetricsReport extends SamplingReport implements MessageListener {

    public static final String BIN_STEPS_S = "binSteps";

    private final int binSteps;

    private int curSteps = 0;
    private double binStartTime = Double.NaN;

    // Cumulative counters (ignore warmup-created messages like MessageStatsReport)
    private int cumCreated = 0;
    private int cumDelivered = 0;
    private int cumRelayed = 0;
    private double cumDelaySum = 0.0;
    private int cumDelayCount = 0;

    public StepBinMetricsReport() {
        super();
        final Settings s = getSettings();
        int b = (int)Math.round(s.getDouble(BIN_STEPS_S, 100.0));
        if (b <= 0) b = 100;
        this.binSteps = b;
        write("# t0 t1 steps delivered created rate overhead avg_delay");
    }

    @Override
    protected void sample(List<DTNHost> hosts) {
        if (curSteps == 0) {
            binStartTime = SimClock.getTime();
        }
        curSteps += 1;
        if (curSteps >= binSteps) {
            flushBin();
        }
    }

    private void flushBin() {
        double t0 = (Double.isNaN(binStartTime) ? SimClock.getTime() : binStartTime);
        double t1 = SimClock.getTime();
        Double rate = null;
        Double overhead = null;
        Double avgDelay = null;
        if (cumCreated > 0) {
            rate = cumDelivered / (double) cumCreated;
        }
        if (cumDelivered > 0) {
            overhead = (cumRelayed - cumDelivered) / (double) cumDelivered;
            avgDelay = (cumDelayCount > 0 ? (cumDelaySum / cumDelayCount) : null);
        }
        write(format(t0) + " " + format(t1) + " " + curSteps + " " +
                cumDelivered + " " + cumCreated + " " +
                format(rate == null ? Double.NaN : rate.doubleValue()) + " " +
                format(overhead == null ? Double.NaN : overhead.doubleValue()) + " " +
                format(avgDelay == null ? Double.NaN : avgDelay.doubleValue()));

        // reset bin
        curSteps = 0;
        binStartTime = Double.NaN;
        // keep cumulative counters; only reset bin step tracking
    }

    @Override
    public void done() {
        if (curSteps > 0) {
            flushBin();
        }
        super.done();
    }

    // MessageListener
    public void newMessage(Message m) {
        if (isWarmup()) {
            addWarmupID(m.getId());
            return;
        }
        cumCreated += 1;
    }
    public void messageTransferStarted(Message m, DTNHost from, DTNHost to) { }
    public void messageDeleted(Message m, DTNHost where, boolean dropped) { }
    public void messageTransferAborted(Message m, DTNHost from, DTNHost to) { }
    public void messageTransferred(Message m, DTNHost from, DTNHost to, boolean firstDelivery) {
        // Ignore warmup-created messages
        if (isWarmupID(m.getId())) { return; }
        // Count all hop-level relays
        cumRelayed += 1;
        if (firstDelivery) {
            cumDelivered += 1;
            double delay = SimClock.getTime() - m.getCreationTime();
            if (delay >= 0) { cumDelaySum += delay; cumDelayCount += 1; }
        }
    }
}
