package report;

import core.Connection;
import core.DTNHost;
import routing.MessageRouter;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Maintains per-host knowledge about recently observed buffer occupancies of peers.
 * Entries are updated from direct contacts and merged when tables are exchanged.
 */
class BufferOccupancyTracker {

    private static class Sample {
        double occupancy;
        int time;

        Sample copy() {
            Sample s = new Sample();
            s.occupancy = this.occupancy;
            s.time = this.time;
            return s;
        }
    }

    private final Map<Integer, Map<Integer, Sample>> tables = new HashMap<Integer, Map<Integer, Sample>>();

    void update(List<DTNHost> hosts, int now, int maxAgeSeconds) {
        if (hosts == null || hosts.isEmpty()) {
            return;
        }

        Map<Integer, Double> occupancyCache = new HashMap<Integer, Double>(hosts.size());

        // Ensure table for each host and record self occupancy
        for (DTNHost host : hosts) {
            final int addr = host.getAddress();
            Map<Integer, Sample> table = tables.get(addr);
            if (table == null) {
                table = new HashMap<Integer, Sample>();
                tables.put(addr, table);
            }
            double selfOcc = computeOccupancy(host);
            if (!Double.isNaN(selfOcc)) {
                putSample(table, addr, selfOcc, now);
            }
            occupancyCache.put(addr, selfOcc);
        }

        // Exchange knowledge between connected peers
        for (DTNHost host : hosts) {
            final int addr = host.getAddress();
            Map<Integer, Sample> table = tables.get(addr);
            double selfOcc = getCachedOrCompute(host, occupancyCache);

            for (Connection conn : host.getConnections()) {
                DTNHost peer = conn.getOtherNode(host);
                final int peerAddr = peer.getAddress();

                Map<Integer, Sample> peerTable = tables.get(peerAddr);
                if (peerTable == null) {
                    peerTable = new HashMap<Integer, Sample>();
                    tables.put(peerAddr, peerTable);
                }

                double peerOcc = getCachedOrCompute(peer, occupancyCache);
                if (!Double.isNaN(peerOcc)) {
                    putSample(table, peerAddr, peerOcc, now);
                }
                if (!Double.isNaN(selfOcc)) {
                    putSample(peerTable, addr, selfOcc, now);
                }

                exchangeTables(table, peerTable);
            }
        }

        if (maxAgeSeconds > 0) {
            for (Map<Integer, Sample> table : tables.values()) {
                prune(table, now, maxAgeSeconds);
            }
        }
    }

    double getMeanOccupancy(int hostAddr) {
        Map<Integer, Sample> table = tables.get(hostAddr);
        if (table == null || table.isEmpty()) {
            return Double.NaN;
        }
        double sum = 0.0;
        int count = 0;
        for (Sample sample : table.values()) {
            if (Double.isNaN(sample.occupancy)) {
                continue;
            }
            sum += sample.occupancy;
            count += 1;
        }
        if (count == 0) {
            return Double.NaN;
        }
        return sum / count;
    }

    private void putSample(Map<Integer, Sample> table, int nodeAddr, double occupancy, int now) {
        if (table == null || Double.isNaN(occupancy)) {
            return;
        }
        Sample sample = table.get(nodeAddr);
        if (sample == null) {
            sample = new Sample();
            table.put(nodeAddr, sample);
        }
        sample.occupancy = clamp01(occupancy);
        sample.time = now;
    }

    private void exchangeTables(Map<Integer, Sample> a, Map<Integer, Sample> b) {
        if (a == null || b == null) {
            return;
        }
        syncOneWay(a, b);
        syncOneWay(b, a);
    }

    private void syncOneWay(Map<Integer, Sample> src, Map<Integer, Sample> dst) {
        if (src == null || dst == null) {
            return;
        }
        for (Map.Entry<Integer, Sample> entry : src.entrySet()) {
            Sample destSample = dst.get(entry.getKey());
            Sample srcSample = entry.getValue();
            if (srcSample == null) {
                continue;
            }
            if (destSample == null || destSample.time < srcSample.time) {
                dst.put(entry.getKey(), srcSample.copy());
            }
        }
    }

    private void prune(Map<Integer, Sample> table, int now, int maxAgeSeconds) {
        if (table == null || table.isEmpty()) {
            return;
        }
        Iterator<Map.Entry<Integer, Sample>> it = table.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Integer, Sample> entry = it.next();
            Sample sample = entry.getValue();
            if (sample == null) {
                it.remove();
                continue;
            }
            if (now - sample.time > maxAgeSeconds) {
                it.remove();
            }
        }
    }

    private double getCachedOrCompute(DTNHost host, Map<Integer, Double> cache) {
        if (host == null) {
            return Double.NaN;
        }
        Double cached = cache.get(host.getAddress());
        if (cached != null) {
            return cached.doubleValue();
        }
        double occ = computeOccupancy(host);
        cache.put(host.getAddress(), occ);
        return occ;
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

    private double clamp01(double value) {
        if (Double.isNaN(value)) {
            return value;
        }
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }
}
