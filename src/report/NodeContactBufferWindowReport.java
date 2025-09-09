/*
 * Reports, for each node and over time, two 20s-windowed averages:
 * - Average number of concurrent contacts (unique peers connected)
 * - Average free buffer size among {self + currently contacted peers}
 *
 * Configurable via:
 * - sampleInterval (inherited from SamplingReport; default 60s)
 * - windowSize (seconds; default 20s)
 */
package report;

import core.Connection;
import core.DTNHost;
import core.Settings;
import routing.MessageRouter;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class NodeContactBufferWindowReport extends SamplingReport {

    public static final String WINDOW_SIZE_S = "windowSize"; // seconds
    public static final String CONTACTS_NORM_MODE_S = "contactsNormMode"; // cmax|saturate
    public static final String CONTACTS_CMAX_S = "contactsCmax"; // for cmax mode
    public static final String CONTACTS_TAU_S = "contactsTau";   // for saturate mode

    private final int windowSizeSeconds;
    private final Map<Integer, Deque<Double>> contactsHistory = new HashMap<Integer, Deque<Double>>();
    private final Map<Integer, Deque<Double>> freeBufHistory = new HashMap<Integer, Deque<Double>>();
    private final Map<Integer, Deque<Double>> freeFracHistory = new HashMap<Integer, Deque<Double>>();

    private final String contactsNormMode; // "cmax" or "saturate"
    private final double contactsCmax;
    private final double contactsTau;

    public NodeContactBufferWindowReport() {
        super();
        final Settings s = getSettings();
        int w = (int)Math.round(s.getDouble(WINDOW_SIZE_S, 20.0));
        if (w <= 0) { w = 20; }
        this.windowSizeSeconds = w;
        this.contactsNormMode = s.getSetting(CONTACTS_NORM_MODE_S, "cmax").toLowerCase();
        this.contactsCmax = s.getDouble(CONTACTS_CMAX_S, 10.0);
        this.contactsTau = s.getDouble(CONTACTS_TAU_S, 3.0);

        write("# time host avg_contacts_" + windowSizeSeconds + "s avg_freebuf_bytes_" + windowSizeSeconds + "s contacts_norm freebuf_norm");
    }

    @Override
    protected void sample(final List<DTNHost> hosts) {
        final int windowSamples = Math.max(1, (int)Math.round(this.windowSizeSeconds / super.interval));

        for (DTNHost h : hosts) {
            final int addr = h.getAddress();

            // Compute current contacts (unique peers)
            final Set<Integer> peers = new HashSet<Integer>();
            for (Connection c : h.getConnections()) {
                peers.add(c.getOtherNode(h).getAddress());
            }
            final double contactsNow = peers.size();

            // Compute current average free buffer of {self + peers}
            long totalFree = 0L;
            int count = 0;
            double sumFrac = 0.0;
            int countFrac = 0;

            // self
            final MessageRouter selfRouter = h.getRouter();
            if (selfRouter != null) {
                long free = selfRouter.getFreeBufferSize();
                long size = selfRouter.getBufferSize();
                totalFree += free;
                count += 1;
                if (size > 0 && size < Integer.MAX_VALUE) {
                    double frac = Math.max(0.0, Math.min(1.0, (free * 1.0) / size));
                    sumFrac += frac;
                    countFrac += 1;
                }
            }
            // peers
            for (Integer pid : peers) {
                // Find DTNHost by address; hosts list is small, linear scan ok
                DTNHost ph = null;
                for (DTNHost cand : hosts) {
                    if (cand.getAddress() == pid) { ph = cand; break; }
                }
                if (ph != null && ph.getRouter() != null) {
                    MessageRouter r = ph.getRouter();
                    long free = r.getFreeBufferSize();
                    long size = r.getBufferSize();
                    totalFree += free;
                    count += 1;
                    if (size > 0 && size < Integer.MAX_VALUE) {
                        double frac = Math.max(0.0, Math.min(1.0, (free * 1.0) / size));
                        sumFrac += frac;
                        countFrac += 1;
                    }
                }
            }
            final double avgFreeNow = (count > 0) ? (totalFree * 1.0) / count : Double.NaN;
            final double avgFreeFracNow = (countFrac > 0) ? (sumFrac / countFrac) : Double.NaN;

            // Update per-node histories
            Deque<Double> ch = contactsHistory.get(addr);
            if (ch == null) {
                ch = new ArrayDeque<Double>(windowSamples);
                contactsHistory.put(addr, ch);
            }
            Deque<Double> fh = freeBufHistory.get(addr);
            if (fh == null) {
                fh = new ArrayDeque<Double>(windowSamples);
                freeBufHistory.put(addr, fh);
            }
            Deque<Double> ffh = freeFracHistory.get(addr);
            if (ffh == null) {
                ffh = new ArrayDeque<Double>(windowSamples);
                freeFracHistory.put(addr, ffh);
            }

            if (ch.size() == windowSamples) { ch.removeFirst(); }
            if (fh.size() == windowSamples) { fh.removeFirst(); }
            if (ffh.size() == windowSamples) { ffh.removeFirst(); }
            ch.addLast(contactsNow);
            fh.addLast(avgFreeNow);
            ffh.addLast(avgFreeFracNow);

            // Skip until window is full (no output for initial window)
            if (ch.size() < windowSamples || fh.size() < windowSamples || ffh.size() < windowSamples) { continue; }

            // Compute window averages
            double sumC = 0.0;
            double sumF = 0.0;
            double sumFfrac = 0.0;
            for (double v : ch) { sumC += v; }
            for (double v : fh) { sumF += v; }
            for (double v : ffh) { sumFfrac += v; }
            final double avgC = sumC / windowSamples;
            final double avgF = sumF / windowSamples;
            final double avgFfrac = sumFfrac / windowSamples;

            // Normalize contacts
            double cNorm;
            if ("saturate".equals(this.contactsNormMode)) {
                double tau = (this.contactsTau > 0 ? this.contactsTau : 3.0);
                cNorm = 1.0 - Math.exp(-avgC / tau);
            } else {
                double cmax = (this.contactsCmax > 0 ? this.contactsCmax : 10.0);
                cNorm = avgC / cmax;
                if (cNorm > 1.0) cNorm = 1.0;
                if (cNorm < 0.0) cNorm = 0.0;
            }

            // Normalize free buffer as fraction already in [0,1]
            double fNorm = avgFfrac; // may be NaN if no valid sizes

            write(((int) getSimTime()) + " " + h.toString() + " " + format(avgC) + " " + format(avgF) + " " + format(cNorm) + " " + format(fNorm));
        }
    }
}
