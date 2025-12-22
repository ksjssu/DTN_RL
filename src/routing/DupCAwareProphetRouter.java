/*
 * DupC-aware Prophet routing: extends ProphetRouter with DupC threshold and
 * AvgDupC-based relaxation. Implements:
 *  - DupC (replica counter) per message, incremented on successful transfer
 *  - Knowledge exchange of DupC values on contact (also for messages not held)
 *  - Forwarding rule:
 *      * If DupC >= T_DUP: do not forward
 *      * Else if DupC > AvgDupC: require PB > PA (Prophet original)
 *      * Else (DupC <= AvgDupC): require PB > max(0, PA - DELTA)
 */
package routing;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import core.Connection;
import core.DTNHost;
import core.Message;
import core.Settings;
import routing.util.RoutingInfo;
import util.Tuple;

public class DupCAwareProphetRouter extends ProphetRouter {
    public static final String DUPC_KEY = "DupC";

    // Settings namespace & keys
    public static final String NS = "DupCAwareProphetRouter";
    public static final String T_DUP_S = "tDup";   // int
    public static final String DELTA_S = "delta";  // double

    // Defaults
    public static final int DEF_T_DUP = 16;
    public static final double DEF_DELTA = 0.2;

    // Instance-configured parameters
    private int tDup;
    private double delta;

    // Knowledge of DupC for messages this node may or may not carry
    private Map<String, Integer> knownDupC = new HashMap<String, Integer>();

    public DupCAwareProphetRouter(Settings s) {
        super(s);
        Settings ds = new Settings(NS);
        this.tDup = ds.getInt(T_DUP_S, DEF_T_DUP);
        this.delta = ds.getDouble(DELTA_S, DEF_DELTA);
    }

    protected DupCAwareProphetRouter(DupCAwareProphetRouter r) {
        super(r);
        // shallow copy is fine; values are immutable Integers
        this.knownDupC = new HashMap<String, Integer>(r.knownDupC);
        this.tDup = r.tDup;
        this.delta = r.delta;
    }

    // Runtime parameter update (for DRL / external control)
    public void setParams(int newTDup, double newDelta) {
        if (newTDup > 0) { this.tDup = newTDup; }
        this.delta = newDelta;
    }
    public void setTDup(int newTDup) { if (newTDup > 0) this.tDup = newTDup; }
    public void setDelta(double newDelta) { this.delta = newDelta; }
    public int getTDup() { return this.tDup; }
    public double getDelta() { return this.delta; }

    // Utils for DupC
    private static int asInt(Object o, int def) {
        if (o == null) return def;
        if (o instanceof Integer) return ((Integer)o).intValue();
        if (o instanceof Number) return ((Number)o).intValue();
        try { return Integer.parseInt(String.valueOf(o)); } catch (Exception e) { return def; }
    }

    private int getLocalDupC(Message m) {
        return asInt(m.getProperty(DUPC_KEY), 1);
    }

    private void setLocalDupC(Message m, int v) {
        try {
            m.updateProperty(DUPC_KEY, Integer.valueOf(v));
        } catch (Exception e) {
            // should not happen; addProperty/updateProperty both allowed
            try { m.addProperty(DUPC_KEY, Integer.valueOf(v)); } catch (Exception ignored) {}
        }
    }

    private int getKnownDupC(String msgId) {
        Integer v = knownDupC.get(msgId);
        return v == null ? -1 : v.intValue();
    }

    private void putKnownDupC(String msgId, int v) {
        Integer cur = knownDupC.get(msgId);
        if (cur == null || v > cur.intValue()) {
            knownDupC.put(msgId, Integer.valueOf(v));
        }
    }

    private double getAvgDupCOverBuffer() {
        Collection<Message> msgs = getMessageCollection();
        if (msgs.isEmpty()) return 1.0; // default
        long sum = 0;
        int n = 0;
        for (Message m : msgs) {
            sum += getLocalDupC(m);
            n++;
        }
        return n == 0 ? 1.0 : ((double)sum) / n;
    }

    @Override
    public void changedConnection(Connection con) {
        super.changedConnection(con);
        if (!con.isUp()) { return; }
        // Exchange DupC knowledge with peer (SV+ style)
        DTNHost otherHost = con.getOtherNode(getHost());
        if (!(otherHost.getRouter() instanceof DupCAwareProphetRouter)) {
            return; // only supported with same-type routers
        }
        DupCAwareProphetRouter peer = (DupCAwareProphetRouter)otherHost.getRouter();

        // 1) Share local buffer DupC
        for (Message m : this.getMessageCollection()) {
            int dup = getLocalDupC(m);
            this.putKnownDupC(m.getId(), dup);
            peer.putKnownDupC(m.getId(), dup);
        }
        for (Message m : peer.getMessageCollection()) {
            int dup = peer.getLocalDupC(m);
            this.putKnownDupC(m.getId(), dup);
            peer.putKnownDupC(m.getId(), dup);
        }

        // 2) Share known (even for messages not held) by taking max across maps
        for (Map.Entry<String,Integer> e : this.knownDupC.entrySet()) {
            peer.putKnownDupC(e.getKey(), e.getValue());
        }
        for (Map.Entry<String,Integer> e : peer.knownDupC.entrySet()) {
            this.putKnownDupC(e.getKey(), e.getValue());
        }
        // 3) Optionally lift local copies' DupC to known max
        for (Message m : this.getMessageCollection()) {
            int known = getKnownDupC(m.getId());
            int local = getLocalDupC(m);
            if (known > local) { setLocalDupC(m, known); }
        }
        for (Message m : peer.getMessageCollection()) {
            int known = peer.getKnownDupC(m.getId());
            int local = peer.getLocalDupC(m);
            if (known > local) { peer.setLocalDupC(m, known); }
        }
    }

    @Override
    public void update() {
        super.update();
        // Routing decisions are handled in tryOtherMessages override in this subclass
        if (!canStartTransfer() || isTransferring()) { return; }
        if (exchangeDeliverableMessages() != null) { return; }
        tryOtherMessagesDupC();
    }

    private Tuple<Message, Connection> tryOtherMessagesDupC() {
        List<Tuple<Message, Connection>> messages = new ArrayList<Tuple<Message, Connection>>();
        Collection<Message> msgCollection = getMessageCollection();

        double avgDup = getAvgDupCOverBuffer();

        for (Connection con : getConnections()) {
            DTNHost other = con.getOtherNode(getHost());
            if (!(other.getRouter() instanceof ProphetRouter)) { continue; }
            ProphetRouter othRouter = (ProphetRouter)other.getRouter();
            if (othRouter.isTransferring()) { continue; }

            for (Message m : msgCollection) {
                if (othRouter.hasMessage(m.getId())) { continue; }

                int dup = getLocalDupC(m);
                if (dup < 0) { dup = 1; }
                // Paper rule: block only if DupC > tDup (strictly greater)
                if (dup > tDup) { continue; }

                double PA = this.getPredFor(m.getTo());
                double PB = othRouter.getPredFor(m.getTo());

                // Paper: relax only when DupC < AvgDupC; otherwise use standard Prophet
                boolean useRelax = (dup < avgDup);
                double thresh = useRelax ? (PA - this.delta) : PA; // no clamping in paper

                if (PB > thresh) {
                    messages.add(new Tuple<Message, Connection>(m, con));
                }
            }
        }

        if (messages.isEmpty()) { return null; }
        Collections.sort(messages, new DupCComparator());
        return tryMessagesForConnected(messages);
    }

    private class DupCComparator implements Comparator<Tuple<Message, Connection>> {
        public int compare(Tuple<Message, Connection> a, Tuple<Message, Connection> b) {
            Message m1 = a.getKey();
            Message m2 = b.getKey();
            // 1) lower DupC first
            int d1 = getLocalDupC(m1);
            int d2 = getLocalDupC(m2);
            if (d1 != d2) { return (d1 < d2) ? -1 : 1; }
            // 2) higher remote advantage next
            DTNHost o1 = a.getValue().getOtherNode(getHost());
            DTNHost o2 = b.getValue().getOtherNode(getHost());
            double p1 = ((ProphetRouter)o1.getRouter()).getPredFor(m1.getTo());
            double p1a = getPredFor(m1.getTo());
            double p2 = ((ProphetRouter)o2.getRouter()).getPredFor(m2.getTo());
            double p2a = getPredFor(m2.getTo());
            double adv1 = p1 - p1a;
            double adv2 = p2 - p2a;
            if (adv1 == adv2) { return compareByQueueMode(m1, m2); }
            return (adv2 > adv1) ? 1 : -1;
        }
    }

    @Override
    protected void transferDone(Connection con) {
        // Called on sender side when transfer finishes successfully
        if (con.getMessage() != null) {
            String id = con.getMessage().getId();
            Message local = this.getMessage(id);
            if (local != null) {
                int dup = getLocalDupC(local);
                int newDup = dup + 1;
                setLocalDupC(local, newDup);
                putKnownDupC(id, newDup);
            }
        }
    }

    @Override
    public Message messageTransferred(String id, DTNHost from) {
        // Receiver side: bump DupC to (old + 1)
        Message m = super.messageTransferred(id, from);
        if (m != null) {
            int dup = getLocalDupC(m);
            int newDup = dup + 1;
            setLocalDupC(m, newDup);
            putKnownDupC(id, newDup);
        }
        return m;
    }

    @Override
    public boolean createNewMessage(Message m) {
        // Initialize DupC to 1 at source
        boolean ok = super.createNewMessage(m);
        if (ok) {
            setLocalDupC(m, 1);
            putKnownDupC(m.getId(), 1);
        }
        return ok;
    }

    @Override
    public RoutingInfo getRoutingInfo() {
        RoutingInfo top = super.getRoutingInfo();
        top.addMoreInfo(new RoutingInfo("DupC-known: " + knownDupC.size()));
        top.addMoreInfo(new RoutingInfo("DupC-params: tDup=" + this.tDup + ", delta=" + String.format("%.4f", this.delta)));
        return top;
    }

    @Override
    public MessageRouter replicate() {
        return new DupCAwareProphetRouter(this);
    }
}
