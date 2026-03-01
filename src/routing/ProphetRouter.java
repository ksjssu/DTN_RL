/*
 * Copyright 2010 Aalto University, ComNet
 * Released under GPLv3. See LICENSE.txt for details.
 */
package routing;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import routing.util.RoutingInfo;

import util.Tuple;

import core.Connection;
import core.DTNHost;
import core.Message;
import core.Settings;
import core.SimClock;

/**
 * Implementation of PRoPHET router as described in
 * <I>Probabilistic routing in intermittently connected networks</I> by
 * Anders Lindgren et al.
 */
public class ProphetRouter extends ActiveRouter {
	/** delivery predictability initialization constant*/
	public static final double P_INIT = 0.75;
	/** Per-host configurable P_INIT setting key ({@value}). */
	public static final String P_INIT_S = "pInit";
	/** delivery predictability transitivity scaling constant default value */
	public static final double DEFAULT_BETA = 0.25;
	/** delivery predictability aging constant */
	public static final double DEFAULT_GAMMA = 0.98;

	/** Prophet router's setting namespace ({@value})*/
	public static final String PROPHET_NS = "ProphetRouter";
	/**
	 * Number of seconds in time unit -setting id ({@value}).
	 * How many seconds one time unit is when calculating aging of
	 * delivery predictions. Should be tweaked for the scenario.*/
	public static final String SECONDS_IN_UNIT_S ="secondsInTimeUnit";

	/**
	 * Transitivity scaling constant (beta) -setting id ({@value}).
	 * Default value for setting is {@link #DEFAULT_BETA}.
	 */
	public static final String BETA_S = "beta";

	/**
	 * Predictability aging constant (gamma) -setting id ({@value}).
	 * Default value for setting is {@link #DEFAULT_GAMMA}.
	 */
	public static final String GAMMA_S = "gamma";

	/** Setting key for heuristic delta adjustments. */
	public static final String HEURISTIC_DELTA_S = "heuristicDelta";

	/** the value of nrof seconds in time unit -setting */
	private int secondsInTimeUnit;
	/** value of beta setting */
	private double beta;
	/** value of gamma setting */
	private double gamma;
	/** value of pInit setting (direct update strength) */
	private double pInit;

    /** delivery predictabilities */
    private Map<DTNHost, Double> preds;
    /** external offset for predictabilities (e.g., DRL delta per-destination) */
    private Map<DTNHost, Double> extOffsets = new HashMap<DTNHost, Double>();
    /** counts of skipped relays due to lower peer predictability, per destination */
    private Map<DTNHost, Integer> lowPredSkips = new HashMap<DTNHost, Integer>();
    /** counts of opportunities where peer has higher predictability, per destination */
    private Map<DTNHost, Integer> highPredOpportunities = new HashMap<DTNHost, Integer>();
	/** last delivery predictability update (sim)time */
	private double lastAgeUpdate;

	/**
	 * Constructor. Creates a new message router based on the settings in
	 * the given Settings object.
	 * @param s The settings object
	 */
	public ProphetRouter(Settings s) {
		super(s);
		Settings prophetSettings = new Settings(PROPHET_NS);
		secondsInTimeUnit = prophetSettings.getInt(SECONDS_IN_UNIT_S);
		if (prophetSettings.contains(BETA_S)) {
			beta = prophetSettings.getDouble(BETA_S);
		}
		else {
			beta = DEFAULT_BETA;
		}

		if (prophetSettings.contains(GAMMA_S)) {
			gamma = prophetSettings.getDouble(GAMMA_S);
		}
		else {
			gamma = DEFAULT_GAMMA;
		}

		if (prophetSettings.contains(P_INIT_S)) {
			pInit = prophetSettings.getDouble(P_INIT_S);
		}
		else {
			pInit = P_INIT;
		}
		// Clamp to safe ranges
		setBeta(beta);
		setGamma(gamma);
		setPInit(pInit);

		initPreds();
	}

	/**
	 * Copyconstructor.
	 * @param r The router prototype where setting values are copied from
	 */
	protected ProphetRouter(ProphetRouter r) {
		super(r);
		this.secondsInTimeUnit = r.secondsInTimeUnit;
		this.beta = r.beta;
		this.gamma = r.gamma;
		this.pInit = r.pInit;
		initPreds();
	}

	/**
	 * Initializes predictability hash
	 */
	private void initPreds() {
		this.preds = new HashMap<DTNHost, Double>();
	}

	@Override
	public void changedConnection(Connection con) {
		super.changedConnection(con);

		if (con.isUp()) {
			DTNHost otherHost = con.getOtherNode(getHost());
			updateDeliveryPredFor(otherHost);
			updateTransitivePreds(otherHost);
		}
	}

	/**
	 * Updates delivery predictions for a host.
	 * <CODE>P(a,b) = P(a,b)_old + (1 - P(a,b)_old) * P_INIT</CODE>
	 * @param host The host we just met
	 */
	private void updateDeliveryPredFor(DTNHost host) {
		double oldValue = getPredFor(host);
		double newValue = oldValue + (1 - oldValue) * this.pInit;
		preds.put(host, newValue);
	}

	/**
	 * Sets per-host P_INIT (direct-update strength).
	 * @param pInit New P_INIT value (clamped to [0,1])
	 */
	public void setPInit(double pInit) {
		if (!Double.isFinite(pInit)) {
			return;
		}
		if (pInit < 0.0) { pInit = 0.0; }
		if (pInit > 1.0) { pInit = 1.0; }
		this.pInit = pInit;
	}

	/**
	 * @return Current per-host P_INIT value.
	 */
	public double getPInit() {
		return this.pInit;
	}

	/**
	 * Sets per-host beta (transitivity scaling).
	 * @param beta New beta value (clamped to [0,1])
	 */
	public void setBeta(double beta) {
		if (!Double.isFinite(beta)) {
			return;
		}
		if (beta < 0.0) { beta = 0.0; }
		if (beta > 1.0) { beta = 1.0; }
		this.beta = beta;
	}

	/**
	 * @return Current per-host beta value.
	 */
	public double getBeta() {
		return this.beta;
	}

	/**
	 * Sets per-host gamma (aging/forgetting).
	 * @param gamma New gamma value (clamped to (0,1])
	 */
	public void setGamma(double gamma) {
		if (!Double.isFinite(gamma)) {
			return;
		}
		if (gamma <= 0.0) { gamma = 1e-6; }
		if (gamma > 1.0) { gamma = 1.0; }
		this.gamma = gamma;
	}

	/**
	 * @return Current per-host gamma value.
	 */
	public double getGamma() {
		return this.gamma;
	}

	/**
	 * Returns the current prediction (P) value for a host or 0 if entry for
	 * the host doesn't exist.
	 * @param host The host to look the P for
	 * @return the current P value
	 */
    public double getPredFor(DTNHost host) {
        ageDeliveryPreds(); // make sure preds are updated before getting
        double base = preds.containsKey(host) ? preds.get(host) : 0.0;
        Double off = extOffsets.get(host);
        double val = base + (off == null ? 0.0 : off.doubleValue());
        if (val < 0) val = 0;
        if (val > 1) val = 1;
        return val;
    }

	/**
	 * Returns the base prediction (P) value for a host WITHOUT external offset.
	 * This is the pure PROPHET delivery predictability before DRL delta is applied.
	 * Used for reward calculation to prevent reward hacking.
	 * @param host The host to look the base P for
	 * @return the base P value (without external offset)
	 */
	public double getBasePredFor(DTNHost host) {
		ageDeliveryPreds(); // make sure preds are updated before getting
		return preds.containsKey(host) ? preds.get(host) : 0.0;
	}

	/**
	 * Updates transitive (A->B->C) delivery predictions.
	 * <CODE>P(a,c) = P(a,c)_old + (1 - P(a,c)_old) * P(a,b) * P(b,c) * BETA
	 * </CODE>
	 * @param host The B host who we just met
	 */
	private void updateTransitivePreds(DTNHost host) {
		MessageRouter otherRouter = host.getRouter();
		assert otherRouter instanceof ProphetRouter : "PRoPHET only works " +
			" with other routers of same type";

		double pForHost = getPredFor(host); // P(a,b)
		Map<DTNHost, Double> othersPreds =
			((ProphetRouter)otherRouter).getDeliveryPreds();

		for (Map.Entry<DTNHost, Double> e : othersPreds.entrySet()) {
			if (e.getKey() == getHost()) {
				continue; // don't add yourself
			}

			double pOld = getPredFor(e.getKey()); // P(a,c)_old
			double pNew = pOld + ( 1 - pOld) * pForHost * e.getValue() * beta;
			preds.put(e.getKey(), pNew);
		}
	}

	/**
	 * Ages all entries in the delivery predictions.
	 * <CODE>P(a,b) = P(a,b)_old * (GAMMA ^ k)</CODE>, where k is number of
	 * time units that have elapsed since the last time the metric was aged.
	 * @see #SECONDS_IN_UNIT_S
	 */
	private void ageDeliveryPreds() {
		double timeDiff = (SimClock.getTime() - this.lastAgeUpdate) /
			secondsInTimeUnit;

		if (timeDiff == 0) {
			return;
		}

		double mult = Math.pow(gamma, timeDiff);
		for (Map.Entry<DTNHost, Double> e : preds.entrySet()) {
			e.setValue(e.getValue()*mult);
		}

		this.lastAgeUpdate = SimClock.getTime();
	}

	/**
	 * Returns a map of this router's delivery predictions
	 * @return a map of this router's delivery predictions
	 */
	private Map<DTNHost, Double> getDeliveryPreds() {
		ageDeliveryPreds(); // make sure the aging is done
		return this.preds;
	}

	@Override
	public void update() {
		super.update();
		if (!canStartTransfer() ||isTransferring()) {
			return; // nothing to transfer or is currently transferring
		}

		// try messages that could be delivered to final recipient
		if (exchangeDeliverableMessages() != null) {
			return;
		}

		tryOtherMessages();
	}

	/**
	 * Tries to send all other messages to all connected hosts ordered by
	 * their delivery probability
	 * @return The return value of {@link #tryMessagesForConnected(List)}
	 */
	private Tuple<Message, Connection> tryOtherMessages() {
		List<Tuple<Message, Connection>> messages =
			new ArrayList<Tuple<Message, Connection>>();

		Collection<Message> msgCollection = getMessageCollection();

		/* for all connected hosts collect all messages that have a higher
		   probability of delivery by the other host */
		for (Connection con : getConnections()) {
			DTNHost other = con.getOtherNode(getHost());
			ProphetRouter othRouter = (ProphetRouter)other.getRouter();

			if (othRouter.isTransferring()) {
				continue; // skip hosts that are transferring
			}

			for (Message m : msgCollection) {
				if (othRouter.hasMessage(m.getId())) {
					continue; // skip messages that the other one has
				}
                double otherPred = othRouter.getPredFor(m.getTo());
                double selfPred = getPredFor(m.getTo());
				if (otherPred > selfPred) {
					// the other node has higher probability of delivery
					messages.add(new Tuple<Message, Connection>(m,con));
                    recordHighPredOpportunity(m.getTo());
				} else if (otherPred < selfPred) {
                    recordLowPredSkip(m.getTo());
                }
			}
		}

		if (messages.size() == 0) {
			return null;
		}

		// sort the message-connection tuples
		Collections.sort(messages, new TupleComparator());
		return tryMessagesForConnected(messages);	// try to send messages
	}

	/**
	 * Comparator for Message-Connection-Tuples that orders the tuples by
	 * their delivery probability by the host on the other side of the
	 * connection (GRTRMax)
	 */
	private class TupleComparator implements Comparator
		<Tuple<Message, Connection>> {

		public int compare(Tuple<Message, Connection> tuple1,
				Tuple<Message, Connection> tuple2) {
			// delivery probability of tuple1's message with tuple1's connection
			double p1 = ((ProphetRouter)tuple1.getValue().
					getOtherNode(getHost()).getRouter()).getPredFor(
					tuple1.getKey().getTo());
			// -"- tuple2...
			double p2 = ((ProphetRouter)tuple2.getValue().
					getOtherNode(getHost()).getRouter()).getPredFor(
					tuple2.getKey().getTo());

			// bigger probability should come first
			if (p2-p1 == 0) {
				/* equal probabilities -> let queue mode decide */
				return compareByQueueMode(tuple1.getKey(), tuple2.getKey());
			}
			else if (p2-p1 < 0) {
				return -1;
			}
			else {
				return 1;
			}
		}
	}

	@Override
    public RoutingInfo getRoutingInfo() {
        ageDeliveryPreds();
        RoutingInfo top = super.getRoutingInfo();
        RoutingInfo ri = new RoutingInfo(preds.size() +
                " delivery prediction(s)");

		for (Map.Entry<DTNHost, Double> e : preds.entrySet()) {
			DTNHost host = e.getKey();
			Double value = e.getValue();

			ri.addMoreInfo(new RoutingInfo(String.format("%s : %.6f",
					host, value)));
		}

        top.addMoreInfo(ri);
        return top;
    }

    /** Clears all external offsets (e.g., at the start of a new step). */
    public void clearExternalOffsets() {
        this.extOffsets.clear();
    }

    /** Sets external offset for destination host. Value is added to base pred and clamped to [0,1]. */
    public void setExternalOffset(DTNHost dest, double offset) {
        if (dest == null) return;
        this.extOffsets.put(dest, new Double(offset));
    }

    /** Returns and resets counts of low-predictability relay skips per destination. */
    public Map<DTNHost, Integer> drainLowPredSkips() {
        Map<DTNHost, Integer> out = new HashMap<DTNHost, Integer>(this.lowPredSkips);
        this.lowPredSkips.clear();
        return out;
    }

    /** Returns and resets counts of higher-predictability peer opportunities per destination. */
    public Map<DTNHost, Integer> drainHighPredOpportunities() {
        Map<DTNHost, Integer> out = new HashMap<DTNHost, Integer>(this.highPredOpportunities);
        this.highPredOpportunities.clear();
        return out;
    }

    private void recordLowPredSkip(DTNHost dest) {
        if (dest == null) {
            return;
        }
        Integer v = this.lowPredSkips.get(dest);
        this.lowPredSkips.put(dest, (v == null ? 1 : v + 1));
    }

    private void recordHighPredOpportunity(DTNHost dest) {
        if (dest == null) {
            return;
        }
        Integer v = this.highPredOpportunities.get(dest);
        this.highPredOpportunities.put(dest, (v == null ? 1 : v + 1));
    }

	@Override
	public MessageRouter replicate() {
		ProphetRouter r = new ProphetRouter(this);
		return r;
	}

}
