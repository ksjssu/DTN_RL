/*
 * Snapshot report for per-node buffer occupancy, energy level,
 * and average PRoPHET predictability over all other nodes.
 */
package report;

import java.util.List;

import core.DTNHost;
import routing.MessageRouter;
import routing.ProphetRouter;
import routing.ProphetRouterWithEstimation;
import routing.ProphetV2Router;

public class NodeMetricsSnapshotReport extends SnapshotReport {

    private List<DTNHost> lastHosts;

    @Override
    protected void createSnapshot(List<DTNHost> hosts) {
        write("[" + (int) getSimTime() + "]");
        this.lastHosts = hosts;
        for (DTNHost h : hosts) {
            if (this.reportedNodes != null && !this.reportedNodes.contains(h.getAddress())) {
                continue;
            }
            writeSnapshot(h);
        }
    }

    @Override
    protected void writeSnapshot(DTNHost h) {
        double bufferOcc = h.getBufferOccupancy();
        if (bufferOcc > 100.0) {
            bufferOcc = 100.0;
        }

        Double energy = null;
        Object ev = h.getComBus().getProperty(routing.util.EnergyModel.ENERGY_VALUE_ID);
        if (ev instanceof Double) {
            energy = (Double) ev;
        }

        double avgPred = getAvgProphetPredictability(h);

        String energyStr = (energy == null ? "NA" : format(energy));
        write(h.toString() + " " + format(bufferOcc) + " " + energyStr + " " + format(avgPred));
    }

    private double getAvgProphetPredictability(DTNHost h) {
        MessageRouter r = h.getRouter();
        if (lastHosts == null || lastHosts.isEmpty()) {
            return Double.NaN;
        }

        double sum = 0.0;
        int count = 0;

        if (r instanceof ProphetRouter) {
            ProphetRouter pr = (ProphetRouter) r;
            for (DTNHost o : lastHosts) {
                if (o == h) { continue; }
                sum += pr.getPredFor(o);
                count++;
            }
        } else if (r instanceof ProphetV2Router) {
            ProphetV2Router pr = (ProphetV2Router) r;
            for (DTNHost o : lastHosts) {
                if (o == h) { continue; }
                sum += pr.getPredFor(o);
                count++;
            }
        } else if (r instanceof ProphetRouterWithEstimation) {
            ProphetRouterWithEstimation pr = (ProphetRouterWithEstimation) r;
            for (DTNHost o : lastHosts) {
                if (o == h) { continue; }
                sum += pr.getPredFor(o);
                count++;
            }
        } else {
            return Double.NaN;
        }

        return (count > 0) ? (sum / count) : Double.NaN;
    }
}

