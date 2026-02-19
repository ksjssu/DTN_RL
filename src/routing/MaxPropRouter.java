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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import routing.maxprop.MaxPropDijkstra;
import routing.maxprop.MeetingProbabilitySet;
import routing.util.RoutingInfo;
import util.Tuple;
import core.Connection;
import core.DTNHost;
import core.Message;
import core.Settings;
import core.SimClock;
import routing.util.EnergyModel;

/**
 * Implementation of MaxProp router as described in
 * <I>MaxProp: Routing for Vehicle-Based Disruption-Tolerant Networks</I> by
 * John Burgess et al.
 * @version 1.0
 *
 * Extension of the protocol by adding a parameter alpha (default 1)
 * By new connection, the delivery likelihood is increased by alpha
 * and divided by 1+alpha.  Using the default results in the original
 * algorithm.  Refer to Karvo and Ott, <I>Time Scales and Delay-Tolerant Routing
 * Protocols</I> Chants, 2008
 */
public class MaxPropRouter extends ActiveRouter {
    /** Router's setting namespace ({@value})*/
	public static final String MAXPROP_NS = "MaxPropRouter";
	/**
	 * Meeting probability set maximum size -setting id ({@value}).
	 * The maximum amount of meeting probabilities to store.  */
	public static final String PROB_SET_MAX_SIZE_S = "probSetMaxSize";
    /** Default value for the meeting probability set maximum size ({@value}).*/
    public static final int DEFAULT_PROB_SET_MAX_SIZE = 50;
    private static int probSetMaxSize;

	/** probabilities of meeting hosts */
	private MeetingProbabilitySet probs;
	/** meeting probabilities of all hosts from this host's point of view
	 * mapped using host's network address */
	private Map<Integer, MeetingProbabilitySet> allProbs;
	/** the cost-to-node calculator */
	private MaxPropDijkstra dijkstra;
	/** IDs of the messages that are known to have reached the final dst */
	private Set<String> ackedMessageIds;
	/** mapping of the current costs for all messages. This should be set to
	 * null always when the costs should be updated (a host is met or a new
	 * message is received) */
	private Map<Integer, Double> costsForMessages; // legacy single-from cache (kept for compatibility)
	/** From host of the last cost calculation */
	private DTNHost lastCostFrom; // legacy single-from cache (kept for compatibility)

	/** Multi-from cost cache to avoid Dijkstra recomputation thrash. */
	private final Map<Integer, CostCacheEntry> costCacheByFromAddr = new HashMap<Integer, CostCacheEntry>();
	/** Cache generation for invalidating both cost maps and destination set. */
	private int costCacheGeneration = 0;
	/** Cached destination set for Dijkstra calculations (depends on message buffer). */
	private Set<Integer> costToSetCache = null;
	private int costToSetCacheGen = -1;

	private static class CostCacheEntry {
		final Map<Integer, Double> costs;
		final double computedAt;
		CostCacheEntry(Map<Integer, Double> costs, double computedAt) {
			this.costs = costs;
			this.computedAt = computedAt;
		}
	}

	/** Map of which messages have been sent to which hosts from this host */
	private Map<DTNHost, Set<String>> sentMessages;

	/** Over how many samples the "average number of bytes transferred per
	 * transfer opportunity" is taken */
	public static int BYTES_TRANSFERRED_AVG_SAMPLES = 10;
	private int[] avgSamples;
	private int nextSampleIndex = 0;
	/** current value for the "avg number of bytes transferred per transfer
	 * opportunity"  */
	private int avgTransferredBytes = 0;

	/** The alpha parameter string*/
	public static final String ALPHA_S = "alpha";

	/* ===== MaxProp++ (RL-controllable) parameters ===== */
	/** Energy-risk penalty strength added into Dijkstra edge costs. */
	public static final String RL_LAMBDA_COST_S = "rlLambdaCost";
	/** Aging time constant (seconds) for energy/rate information staleness. */
	public static final String RL_TAU_AGE_S = "rlTauAge";
	/** EWMA coefficient for avgTransferredBytes (x) estimation. NaN/negative disables EWMA. */
	public static final String RL_BETA_X_S = "rlBetaX";
	/** Threshold scaling: x_eff = x * rlKx. */
	public static final String RL_KX_S = "rlKx";
	/** Relay margin gating: allow relay only if Δcost >= rlRelayMargin. NaN disables. */
	public static final String RL_RELAY_MARGIN_S = "rlRelayMargin";
	/** Time scale (seconds) for converting time-to-empty to a risk penalty. */
	public static final String ENERGY_RISK_TIME_SCALE_S = "energyRiskTimeScale";
	/** EWMA beta for estimating local energy consumption rate (energy/sec). */
	public static final String ENERGY_RATE_EWMA_BETA_S = "energyRateEwmaBeta";
	/** Max age (seconds) of cached Dijkstra costs when energy-penalty is enabled. */
	public static final String RL_COST_CACHE_MAX_AGE_S = "rlCostCacheMaxAge";

	/** The alpha variable, default = 1;*/
	private double alpha;

	/** The default value for alpha */
	public static final double DEFAULT_ALPHA = 1.0;

	/** RL parameter: lambda_cost (>=0 recommended). */
	private double rlLambdaCost = 0.0;
	/** RL parameter: tau_age seconds (>=0). */
	private double rlTauAgeSeconds = 0.0;
	/** RL parameter: beta_x in [0,1] to enable EWMA, otherwise disabled. */
	private double rlBetaX = Double.NaN;
	/** RL parameter: kx (>0). */
	private double rlKx = 1.0;
	/** RL parameter: relay margin (cost units). */
	private double rlRelayMargin = Double.NaN;
	/** RL parameter: relay margin (cost units) for priority messages (hop < threshold). NaN uses rlRelayMargin. */
	private double rlRelayMarginPrio = Double.NaN;
	/** Per-destination relay margin overrides (host-dest control). Missing/NaN => use node-level margins. */
	private final Map<Integer, Double> rlRelayMarginByDest = new HashMap<Integer, Double>();

	/** EWMA state for avgTransferredBytes estimation when rlBetaX is enabled. */
	private double avgTransferredBytesEwma = 0.0;

	/** Cost cache age tracking (needed when cost is time-varying due to energy penalties). */
	private double costsLastComputedAt = Double.NaN;
	private double rlCostCacheMaxAgeSeconds = 30.0;

	/** Energy penalty config (used when rlLambdaCost > 0). */
	private double energyRiskTimeScaleSeconds = 3600.0;
	private double energyRateEwmaBeta = 0.2;

	private static class EnergyInfo {
		double energy;
		double rate;
		double time;

		EnergyInfo copy() {
			EnergyInfo e = new EnergyInfo();
			e.energy = this.energy;
			e.rate = this.rate;
			e.time = this.time;
			return e;
		}
	}

	private final Map<Integer, EnergyInfo> energyInfoByAddr = new HashMap<Integer, EnergyInfo>();
	private double lastEnergySampleTime = Double.NaN;
	private double lastEnergySample = Double.NaN;
	private double energyRateEwma = 0.0;

	/**
	 * Constructor. Creates a new prototype router based on the settings in
	 * the given Settings object.
	 * @param settings The settings object
	 */
	public MaxPropRouter(Settings settings) {
		super(settings);
		Settings maxPropSettings = new Settings(MAXPROP_NS);
		if (maxPropSettings.contains(ALPHA_S)) {
			alpha = maxPropSettings.getDouble(ALPHA_S);
		} else {
			alpha = DEFAULT_ALPHA;
		}

		// Optional static defaults for MaxProp++ parameters (RL controller can override at runtime).
		this.rlLambdaCost = maxPropSettings.getDouble(RL_LAMBDA_COST_S, 0.0);
		this.rlTauAgeSeconds = maxPropSettings.getDouble(RL_TAU_AGE_S, 0.0);
		this.rlBetaX = maxPropSettings.getDouble(RL_BETA_X_S, Double.NaN);
		this.rlKx = maxPropSettings.getDouble(RL_KX_S, 1.0);
		this.rlRelayMargin = maxPropSettings.getDouble(RL_RELAY_MARGIN_S, Double.NaN);

		this.energyRiskTimeScaleSeconds = maxPropSettings.getDouble(ENERGY_RISK_TIME_SCALE_S, 3600.0);
		if (!Double.isFinite(this.energyRiskTimeScaleSeconds) || this.energyRiskTimeScaleSeconds <= 1e-6) {
			this.energyRiskTimeScaleSeconds = 3600.0;
		}
		this.energyRateEwmaBeta = maxPropSettings.getDouble(ENERGY_RATE_EWMA_BETA_S, 0.2);
		if (!Double.isFinite(this.energyRateEwmaBeta) || this.energyRateEwmaBeta < 0.0) {
			this.energyRateEwmaBeta = 0.0;
		}
		if (this.energyRateEwmaBeta > 0.999) { this.energyRateEwmaBeta = 0.999; }
		this.rlCostCacheMaxAgeSeconds = maxPropSettings.getDouble(RL_COST_CACHE_MAX_AGE_S, 30.0);
		if (!Double.isFinite(this.rlCostCacheMaxAgeSeconds) || this.rlCostCacheMaxAgeSeconds < 0.0) {
			this.rlCostCacheMaxAgeSeconds = 30.0;
		}

	        Settings mpSettings = new Settings(MAXPROP_NS);
	        if (mpSettings.contains(PROB_SET_MAX_SIZE_S)) {
	            probSetMaxSize = mpSettings.getInt(PROB_SET_MAX_SIZE_S);
	        } else {
            probSetMaxSize = DEFAULT_PROB_SET_MAX_SIZE;
        }
	}

	/**
	 * Copy constructor. Creates a new router based on the given prototype.
	 * @param r The router prototype where setting values are copied from
	 */
	protected MaxPropRouter(MaxPropRouter r) {
		super(r);
		this.alpha = r.alpha;
		this.rlLambdaCost = r.rlLambdaCost;
		this.rlTauAgeSeconds = r.rlTauAgeSeconds;
		this.rlBetaX = r.rlBetaX;
		this.rlKx = r.rlKx;
		this.rlRelayMargin = r.rlRelayMargin;
		this.rlRelayMarginPrio = r.rlRelayMarginPrio;
		this.rlRelayMarginByDest.putAll(r.rlRelayMarginByDest);
		this.avgTransferredBytesEwma = r.avgTransferredBytesEwma;
		this.energyRiskTimeScaleSeconds = r.energyRiskTimeScaleSeconds;
		this.energyRateEwmaBeta = r.energyRateEwmaBeta;
		this.rlCostCacheMaxAgeSeconds = r.rlCostCacheMaxAgeSeconds;
		this.probs = new MeetingProbabilitySet(probSetMaxSize, this.alpha);
		this.allProbs = new HashMap<Integer, MeetingProbabilitySet>();
		this.dijkstra = new MaxPropDijkstra(this.allProbs, this::getEnergyPenaltyForNode);
		this.ackedMessageIds = new HashSet<String>();
		this.avgSamples = new int[BYTES_TRANSFERRED_AVG_SAMPLES];
		this.sentMessages = new HashMap<DTNHost, Set<String>>();
	}

	/**
	 * Applies MaxProp++ parameters (typically from an external controller).
	 * Values are stored as-is; caller is expected to clamp to safe ranges.
	 */
	public void setRlParams(double lambdaCost, double tauAgeSeconds, double betaX, double kx, double relayMargin) {
		this.rlLambdaCost = lambdaCost;
		this.rlTauAgeSeconds = tauAgeSeconds;
		this.rlBetaX = betaX;
		this.rlKx = kx;
		this.rlRelayMargin = relayMargin;
		this.rlRelayMarginPrio = Double.NaN;
		invalidateCostCaches(); // invalidate cached costs (cost metric may change)
	}

	/**
	 * Applies MaxProp++ parameters with an optional separate relay margin for priority messages
	 * (hop < threshold). If relayMarginPrio is NaN, rlRelayMargin is used for all messages.
	 */
	public void setRlParams(double lambdaCost, double tauAgeSeconds, double betaX, double kx,
	                        double relayMargin, double relayMarginPrio) {
		this.rlLambdaCost = lambdaCost;
		this.rlTauAgeSeconds = tauAgeSeconds;
		this.rlBetaX = betaX;
		this.rlKx = kx;
		this.rlRelayMargin = relayMargin;
		this.rlRelayMarginPrio = relayMarginPrio;
		invalidateCostCaches(); // invalidate cached costs (cost metric may change)
	}

	private void invalidateCostCaches() {
		this.costsForMessages = null;
		this.lastCostFrom = null;
		this.costCacheByFromAddr.clear();
		this.costToSetCache = null;
		this.costToSetCacheGen = -1;
		this.costCacheGeneration++;
	}

	public double getRlLambdaCost() { return this.rlLambdaCost; }
	public double getRlTauAgeSeconds() { return this.rlTauAgeSeconds; }
	public double getRlBetaX() { return this.rlBetaX; }
	public double getRlKx() { return this.rlKx; }
	public double getRlRelayMargin() { return this.rlRelayMargin; }
	public double getRlRelayMarginPrio() { return this.rlRelayMarginPrio; }

	/** Returns meeting probability to a destination address (0 if unknown). */
	public double getMeetingProbability(int destAddr) {
		try {
			Double v = this.probs.getAllProbs().get(Integer.valueOf(destAddr));
			if (v != null && Double.isFinite(v.doubleValue())) {
				return v.doubleValue();
			}
		} catch (Exception ignore) {}
		return 0.0;
	}

	/** Clears all per-destination relay margin overrides. */
	public void clearRlRelayMarginOverrides() {
		this.rlRelayMarginByDest.clear();
	}

	/** Sets per-destination relay margin override. Use NaN to remove override. */
	public void setRlRelayMarginOverride(int destAddr, double relayMargin) {
		Integer k = Integer.valueOf(destAddr);
		if (!Double.isFinite(relayMargin)) {
			this.rlRelayMarginByDest.remove(k);
			return;
		}
		this.rlRelayMarginByDest.put(k, relayMargin);
	}

	/** Returns the current x estimate used for thresholding (EWMA if enabled). */
	public double getAvgTransferredBytesEstimate() {
		if (Double.isFinite(this.rlBetaX) && this.rlBetaX >= 0.0 && this.rlBetaX <= 1.0) {
			return this.avgTransferredBytesEwma;
		}
		return this.avgTransferredBytes;
	}

	/** Returns this node's current energy value from the energy model (or NaN if unavailable). */
	public double getLocalEnergyValue() {
		return getEnergyValue();
	}

	/** Returns EWMA estimate of local energy consumption rate (energy/sec). */
	public double getLocalEnergyRateEstimate() {
		return this.energyRateEwma;
	}

	public double getKnownEnergyValue(int nodeAddr) {
		EnergyInfo info = this.energyInfoByAddr.get(Integer.valueOf(nodeAddr));
		return (info != null ? info.energy : Double.NaN);
	}

	public double getKnownEnergyRate(int nodeAddr) {
		EnergyInfo info = this.energyInfoByAddr.get(Integer.valueOf(nodeAddr));
		return (info != null ? info.rate : Double.NaN);
	}

	public double getKnownEnergyInfoTime(int nodeAddr) {
		EnergyInfo info = this.energyInfoByAddr.get(Integer.valueOf(nodeAddr));
		return (info != null ? info.time : Double.NaN);
	}

	public boolean hasSentMessageTo(DTNHost peer, String msgId) {
		if (peer == null || msgId == null) {
			return false;
		}
		Set<String> sent = this.sentMessages.get(peer);
		return sent != null && sent.contains(msgId);
	}

	private double getEnergyValue() {
		if (getHost() == null || getHost().getComBus() == null) {
			return Double.NaN;
		}
		Object v = getHost().getComBus().getProperty(EnergyModel.ENERGY_VALUE_ID);
		if (v instanceof Double) {
			return ((Double) v).doubleValue();
		}
		return Double.NaN;
	}

	private void updateLocalEnergyInfo() {
		double now = SimClock.getTime();
		double e = getEnergyValue();
		if (!Double.isFinite(now) || !Double.isFinite(e)) {
			return;
		}
		if (Double.isFinite(this.lastEnergySampleTime) && now > this.lastEnergySampleTime) {
			double dt = now - this.lastEnergySampleTime;
			if (dt > 1e-6 && Double.isFinite(this.lastEnergySample)) {
				double de = this.lastEnergySample - e;
				if (!Double.isFinite(de) || de < 0.0) { de = 0.0; }
				double instRate = de / dt;
				if (!Double.isFinite(instRate) || instRate < 0.0) { instRate = 0.0; }
				double b = this.energyRateEwmaBeta;
				if (!Double.isFinite(b) || b < 0.0) { b = 0.0; }
				if (b > 0.999) { b = 0.999; }
				this.energyRateEwma = (1.0 - b) * this.energyRateEwma + b * instRate;
				if (!Double.isFinite(this.energyRateEwma) || this.energyRateEwma < 0.0) {
					this.energyRateEwma = 0.0;
				}
			}
		}

		this.lastEnergySampleTime = now;
		this.lastEnergySample = e;
		putEnergyInfo(getHost().getAddress(), e, this.energyRateEwma, now);
	}

	private void putEnergyInfo(int addr, double energy, double rate, double now) {
		if (!Double.isFinite(now) || addr < 0) {
			return;
		}
		EnergyInfo cur = this.energyInfoByAddr.get(addr);
		if (cur == null) {
			cur = new EnergyInfo();
			this.energyInfoByAddr.put(addr, cur);
		}
		cur.energy = (Double.isFinite(energy) ? energy : cur.energy);
		cur.rate = (Double.isFinite(rate) ? Math.max(0.0, rate) : cur.rate);
		cur.time = now;
	}

	private void mergeEnergyInfoFrom(MaxPropRouter src) {
		if (src == null) {
			return;
		}
		for (Map.Entry<Integer, EnergyInfo> e : src.energyInfoByAddr.entrySet()) {
			Integer k = e.getKey();
			EnergyInfo theirs = e.getValue();
			if (k == null || theirs == null || !Double.isFinite(theirs.time)) {
				continue;
			}
			EnergyInfo mine = this.energyInfoByAddr.get(k);
			if (mine == null || !Double.isFinite(mine.time) || mine.time < theirs.time) {
				this.energyInfoByAddr.put(k, theirs.copy());
			}
		}
	}

	private void exchangeEnergyInfo(MaxPropRouter otherRouter) {
		if (otherRouter == null) {
			return;
		}
		updateLocalEnergyInfo();
		otherRouter.updateLocalEnergyInfo();
		mergeEnergyInfoFrom(otherRouter);
		otherRouter.mergeEnergyInfoFrom(this);
	}

	private double getEnergyPenaltyForNode(Integer nodeAddr) {
		if (nodeAddr == null) {
			return 0.0;
		}
		if (!Double.isFinite(this.rlLambdaCost) || this.rlLambdaCost <= 0.0) {
			return 0.0;
		}

		EnergyInfo info = this.energyInfoByAddr.get(nodeAddr);
		if (info == null || !Double.isFinite(info.energy) || !Double.isFinite(info.rate) || !Double.isFinite(info.time)) {
			return 0.0;
		}
		double now = SimClock.getTime();
		if (!Double.isFinite(now)) {
			return 0.0;
		}
		double age = now - info.time;
		if (!Double.isFinite(age) || age < 0.0) { age = 0.0; }

		double w = 1.0;
		if (Double.isFinite(this.rlTauAgeSeconds) && this.rlTauAgeSeconds > 1e-6) {
			w = Math.exp(-age / this.rlTauAgeSeconds);
		} else if (age > 0.0) {
			w = 0.0;
		}

		double r = Math.max(0.0, info.rate);
		if (r <= 1e-9) {
			return 0.0;
		}
		double tEmpty = info.energy / r;
		if (!Double.isFinite(tEmpty) || tEmpty < 0.0) { tEmpty = 0.0; }
		double scale = this.energyRiskTimeScaleSeconds;
		if (!Double.isFinite(scale) || scale <= 1e-6) { scale = 3600.0; }
		double risk = Math.exp(-tEmpty / scale);
		if (!Double.isFinite(risk) || risk < 0.0) { risk = 0.0; }

		double pen = this.rlLambdaCost * w * risk;
		if (!Double.isFinite(pen) || pen < 0.0) {
			return 0.0;
		}
		return pen;
	}

	@Override
		public void changedConnection(Connection con) {
			super.changedConnection(con);

			if (con.isUp()) { // new connection
				invalidateCostCaches(); // invalidate old cost estimates

				if (con.isInitiator(getHost())) {
					/* initiator performs all the actions on behalf of the
					 * other node too (so that the meeting probs are updated
				 * for both before exchanging them) */
				DTNHost otherHost = con.getOtherNode(getHost());
				MessageRouter mRouter = otherHost.getRouter();

				assert mRouter instanceof MaxPropRouter : "MaxProp only works "+
				" with other routers of same type";
				MaxPropRouter otherRouter = (MaxPropRouter)mRouter;

				/* exchange ACKed message data */
				this.ackedMessageIds.addAll(otherRouter.ackedMessageIds);
				otherRouter.ackedMessageIds.addAll(this.ackedMessageIds);
				deleteAckedMessages();
				otherRouter.deleteAckedMessages();

				/* update both meeting probabilities */
				probs.updateMeetingProbFor(otherHost.getAddress());
				otherRouter.probs.updateMeetingProbFor(getHost().getAddress());

				/* exchange the transitive probabilities */
				this.updateTransitiveProbs(otherRouter.allProbs);
				otherRouter.updateTransitiveProbs(this.allProbs);
					this.allProbs.put(otherHost.getAddress(),
							otherRouter.probs.replicate());
					otherRouter.allProbs.put(getHost().getAddress(),
							this.probs.replicate());

					// Exchange lightweight energy info table for energy-aware costs (MaxProp++).
					exchangeEnergyInfo(otherRouter);
				}
			}
			else {
			/* connection went down, update transferred bytes average */
			updateTransferredBytesAvg(con.getTotalBytesTransferred());
		}
	}

	/**
	 * Updates transitive probability values by replacing the current
	 * MeetingProbabilitySets with the values from the given mapping
	 * if the given sets have more recent updates.
	 * @param p Mapping of the values of the other host
	 */
	private void updateTransitiveProbs(Map<Integer, MeetingProbabilitySet> p) {
		for (Map.Entry<Integer, MeetingProbabilitySet> e : p.entrySet()) {
			MeetingProbabilitySet myMps = this.allProbs.get(e.getKey());
			if (myMps == null ||
				e.getValue().getLastUpdateTime() > myMps.getLastUpdateTime() ) {
				this.allProbs.put(e.getKey(), e.getValue().replicate());
			}
		}
	}

	/**
	 * Deletes the messages from the message buffer that are known to be ACKed
	 */
	private void deleteAckedMessages() {
		for (String id : this.ackedMessageIds) {
			if (this.hasMessage(id) && !isSending(id)) {
				this.deleteMessage(id, false);
			}
		}
	}

	@Override
	public Message messageTransferred(String id, DTNHost from) {
		invalidateCostCaches(); // new message -> invalidate costs
		Message m = super.messageTransferred(id, from);
		/* was this node the final recipient of the message? */
		if (isDeliveredMessage(m)) {
			this.ackedMessageIds.add(id);
		}
		return m;
	}

	/**
	 * Method is called just before a transfer is finalized
	 * at {@link ActiveRouter#update()}. MaxProp makes book keeping of the
	 * delivered messages so their IDs are stored.
	 * @param con The connection whose transfer was finalized
	 */
	@Override
	protected void transferDone(Connection con) {
		Message m = con.getMessage();
		String id = m.getId();
		DTNHost recipient = con.getOtherNode(getHost());
		Set<String> sentMsgIds = this.sentMessages.get(recipient);

		/* was the message delivered to the final recipient? */
		if (m.getTo() == recipient) {
			this.ackedMessageIds.add(m.getId()); // yes, add to ACKed messages
			this.deleteMessage(m.getId(), false); // delete from buffer
		}

		/* update the map of where each message is already sent */
		if (sentMsgIds == null) {
			sentMsgIds = new HashSet<String>();
			this.sentMessages.put(recipient, sentMsgIds);
		}
		sentMsgIds.add(id);
	}

	/**
	 * Updates the average estimate of the number of bytes transferred per
	 * transfer opportunity.
	 * @param newValue The new value to add to the estimate
	 */
	private void updateTransferredBytesAvg(int newValue) {
		int realCount = 0;
		int sum = 0;

		this.avgSamples[this.nextSampleIndex++] = newValue;
		if(this.nextSampleIndex >= BYTES_TRANSFERRED_AVG_SAMPLES) {
			this.nextSampleIndex = 0;
		}

		for (int i=0; i < BYTES_TRANSFERRED_AVG_SAMPLES; i++) {
			if (this.avgSamples[i] > 0) { // only values above zero count
				realCount++;
				sum += this.avgSamples[i];
			}
		}

		if (realCount > 0) {
			this.avgTransferredBytes = sum / realCount;
		}
		else { // no samples or all samples are zero
			this.avgTransferredBytes = 0;
		}

		// Optional EWMA tracking (for MaxProp++ threshold control).
		if (Double.isFinite(this.rlBetaX) && this.rlBetaX >= 0.0 && this.rlBetaX <= 1.0) {
			// Bootstrap EWMA from the current average estimate if needed.
			if (this.avgTransferredBytesEwma <= 0.0) {
				this.avgTransferredBytesEwma = (double) this.avgTransferredBytes;
			}
			final double b = this.rlBetaX;
			this.avgTransferredBytesEwma = (1.0 - b) * this.avgTransferredBytesEwma +
					b * Math.max(0.0, (double) newValue);
			if (!Double.isFinite(this.avgTransferredBytesEwma) || this.avgTransferredBytesEwma < 0.0) {
				this.avgTransferredBytesEwma = 0.0;
			}
		}
	}

	/**
	 * Returns the next message that should be dropped, according to MaxProp's
	 * message ordering scheme (see MaxPropTupleComparator).
	 * @param excludeMsgBeingSent If true, excludes message(s) that are
	 * being sent from the next-to-be-dropped check (i.e., if next message to
	 * drop is being sent, the following message is returned)
	 * @return The oldest message or null if no message could be returned
	 * (no messages in buffer or all messages in buffer are being sent and
	 * exludeMsgBeingSent is true)
	 */
    @Override
	protected Message getNextMessageToRemove(boolean excludeMsgBeingSent) {
		Collection<Message> messages = this.getMessageCollection();
		List<Message> validMessages = new ArrayList<Message>();

		for (Message m : messages) {
			if (excludeMsgBeingSent && isSending(m.getId())) {
				continue; // skip the message(s) that router is sending
			}
			validMessages.add(m);
		}

		Collections.sort(validMessages,
				new MaxPropComparator(this.calcThreshold()));

		if (validMessages.isEmpty()) {
			return null;
		}

		return validMessages.get(validMessages.size()-1); // return last message
	}

	@Override
	public void update() {
		updateLocalEnergyInfo();
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
	 * Returns the message delivery cost between two hosts from this host's
	 * point of view. If there is no path between "from" and "to" host,
	 * Double.MAX_VALUE is returned. Paths are calculated only to hosts
	 * that this host has messages to.
	 * @param from The host where a message is coming from
	 * @param to The host where a message would be destined to
	 * @return The cost of the cheapest path to the destination or
	 * Double.MAX_VALUE if such a path doesn't exist
	 */
	public double getCost(DTNHost from, DTNHost to) {
		double now = SimClock.getTime();
		if (from == null || to == null) {
			return Double.MAX_VALUE;
		}

		int fromAddr = from.getAddress();
		if (this.costToSetCache == null || this.costToSetCacheGen != this.costCacheGeneration) {
			Set<Integer> toSet = new HashSet<Integer>();
			for (Message m : getMessageCollection()) {
				if (m != null && m.getTo() != null) {
					toSet.add(m.getTo().getAddress());
				}
			}
			this.costToSetCache = toSet;
			this.costToSetCacheGen = this.costCacheGeneration;
		}

		CostCacheEntry entry = this.costCacheByFromAddr.get(Integer.valueOf(fromAddr));
		if (entry != null &&
				Double.isFinite(this.rlLambdaCost) &&
				this.rlLambdaCost > 0.0 &&
				Double.isFinite(now) &&
				Double.isFinite(entry.computedAt) &&
				(now - entry.computedAt) > this.rlCostCacheMaxAgeSeconds) {
			entry = null; // stale due to time-varying energy penalty
		}

		if (entry == null) {
			this.allProbs.put(getHost().getAddress(), this.probs);
			Map<Integer, Double> costs = dijkstra.getCosts(fromAddr, this.costToSetCache);
			entry = new CostCacheEntry(costs, now);
			this.costCacheByFromAddr.put(Integer.valueOf(fromAddr), entry);
		}

		Double c = (entry.costs != null ? entry.costs.get(Integer.valueOf(to.getAddress())) : null);
		if (c != null) {
			return c.doubleValue();
		}
		return Double.MAX_VALUE;
	}

	/**
	 * Tries to send all other messages to all connected hosts ordered by
	 * hop counts and their delivery probability
	 * @return The return value of {@link #tryMessagesForConnected(List)}
	 */
	private Tuple<Message, Connection> tryOtherMessages() {
		List<Tuple<Message, Connection>> messages =
			new ArrayList<Tuple<Message, Connection>>();

		Collection<Message> msgCollection = getMessageCollection();
		final int thresholdCur = calcThreshold();

		/* for all connected hosts that are not transferring at the moment,
		 * collect all the messages that could be sent */
			for (Connection con : getConnections()) {
					DTNHost other = con.getOtherNode(getHost());
					MaxPropRouter othRouter = (MaxPropRouter)other.getRouter();
					Set<String> sentMsgIds = this.sentMessages.get(other);

				if (othRouter.isTransferring()) {
					continue; // skip hosts that are transferring
				}

			for (Message m : msgCollection) {
				/* skip messages that the other host has or that have
				 * passed the other host */
				if (othRouter.hasMessage(m.getId()) ||
						m.getHops().contains(other)) {
					continue;
				}
				/* skip message if this host has already sent it to the other
				   host (regardless of if the other host still has it) */
					if (sentMsgIds != null && sentMsgIds.contains(m.getId())) {
						continue;
					}
					/* Optional relay margin gating (overhead control): relay only if cost improves enough */
					// Allow a separate margin for priority messages (hop < threshold).
					final boolean isPrio = m.getHopCount() < thresholdCur;
					double relayMarginEff = this.rlRelayMargin;
					boolean destOverrideUsed = false;
					// Per-destination override has highest priority (host-dest control).
					try {
						Double ov = this.rlRelayMarginByDest.get(Integer.valueOf(m.getTo().getAddress()));
						if (ov != null && Double.isFinite(ov.doubleValue())) {
							relayMarginEff = ov.doubleValue();
							destOverrideUsed = true;
						}
					} catch (Exception ignore) {}
					if (!destOverrideUsed && isPrio && Double.isFinite(this.rlRelayMarginPrio)) {
						relayMarginEff = this.rlRelayMarginPrio;
					}
					final boolean relayMarginEnabled = Double.isFinite(relayMarginEff) && relayMarginEff >= 0.0;
					if (relayMarginEnabled) {
						double selfCost = getCost(getHost(), m.getTo());
						double peerCost = getCost(other, m.getTo());
						double delta = selfCost - peerCost; // positive if peer is better (lower cost)
						if (!Double.isFinite(delta) || delta < relayMarginEff) {
							continue;
						}
					}
					/* message was a good candidate for sending */
					messages.add(new Tuple<Message, Connection>(m,con));
				}
			}

		if (messages.size() == 0) {
			return null;
		}

		/* sort the message-connection tuples according to the criteria
		 * defined in MaxPropTupleComparator */
		Collections.sort(messages, new MaxPropTupleComparator(thresholdCur));
		return tryMessagesForConnected(messages);
	}

	/**
	 * Calculates and returns the current threshold value for the buffer's split
	 * based on the average number of bytes transferred per transfer opportunity
	 * and the hop counts of the messages in the buffer. Method is public only
	 * to make testing easier.
	 * @return current threshold value (hop count) for the buffer's split
	 */
	public int calcThreshold() {
		/* b, x and p refer to respective variables in the paper's equations */
		long b = this.getBufferSize();
		double xEst = getAvgTransferredBytesEstimate();
		if (!Double.isFinite(xEst) || xEst < 0.0) { xEst = 0.0; }
		double kx = this.rlKx;
		if (!Double.isFinite(kx) || kx <= 0.0) { kx = 1.0; }
		long x = (long) Math.round(xEst * kx);
		long p;

		if (x == 0) {
			/* can't calc the threshold because there's no transfer data */
			return 0;
		}

		/* calculates the portion (bytes) of the buffer selected for priority */
		if (x < b/2) {
			p = x;
		}
		else if (b/2 <= x && x < b) {
			p = Math.min(x, b-x);
		}
		else {
			return 0; // no need for the threshold
		}

		/* creates a copy of the messages list, sorted by hop count */
		ArrayList<Message> msgs = new ArrayList<Message>();
		msgs.addAll(getMessageCollection());
		if (msgs.size() == 0) {
			return 0; // no messages -> no need for threshold
		}
		/* anonymous comparator class for hop count comparison */
		Comparator<Message> hopCountComparator = new Comparator<Message>() {
			public int compare(Message m1, Message m2) {
				return m1.getHopCount() - m2.getHopCount();
			}
		};
		Collections.sort(msgs, hopCountComparator);

		/* finds the first message that is beyond the calculated portion */
		int i=0;
		for (int n=msgs.size(); i<n && p>0; i++) {
			p -= msgs.get(i).getSize();
		}

		i--; // the last round moved i one index too far
		if (i < 0) {
			return 0;
		}

		/* now i points to the first packet that exceeds portion p;
		 * the threshold is that packet's hop count + 1 (so that packet and
		 * perhaps some more are included in the priority part) */
		return msgs.get(i).getHopCount() + 1;
	}

	/**
	 * Message comparator for the MaxProp routing module.
	 * Messages that have a hop count smaller than the given
	 * threshold are given priority and they are ordered by their hop count.
	 * Other messages are ordered by their delivery cost.
	 */
	private class MaxPropComparator implements Comparator<Message> {
		private int threshold;
		private DTNHost from1;
		private DTNHost from2;

		/**
		 * Constructor. Assumes that the host where all the costs are calculated
		 * from is this router's host.
		 * @param treshold Messages with the hop count smaller than this
		 * value are transferred first (and ordered by the hop count)
		 */
		public MaxPropComparator(int treshold) {
			this.threshold = treshold;
			this.from1 = this.from2 = getHost();
		}

		/**
		 * Constructor.
		 * @param treshold Messages with the hop count smaller than this
		 * value are transferred first (and ordered by the hop count)
		 * @param from1 The host where the cost of msg1 is calculated from
		 * @param from2 The host where the cost of msg2 is calculated from
		 */
		public MaxPropComparator(int treshold, DTNHost from1, DTNHost from2) {
			this.threshold = treshold;
			this.from1 = from1;
			this.from2 = from2;
		}

		/**
		 * Compares two messages and returns -1 if the first given message
		 * should be first in order, 1 if the second message should be first
		 * or 0 if message order can't be decided. If both messages' hop count
		 * is less than the threshold, messages are compared by their hop count
		 * (smaller is first). If only other's hop count is below the threshold,
		 * that comes first. If both messages are below the threshold, the one
		 * with smaller cost (determined by
		 * {@link MaxPropRouter#getCost(DTNHost, DTNHost)}) is first.
		 */
		public int compare(Message msg1, Message msg2) {
			double p1, p2;
			int hopc1 = msg1.getHopCount();
			int hopc2 = msg2.getHopCount();

			if (msg1 == msg2) {
				return 0;
			}

			/* if one message's hop count is above and the other one's below the
			 * threshold, the one below should be sent first */
			if (hopc1 < threshold && hopc2 >= threshold) {
				return -1; // message1 should be first
			}
			else if (hopc2 < threshold && hopc1 >= threshold) {
				return 1; // message2 -"-
			}

			/* if both are below the threshold, one with lower hop count should
			 * be sent first */
			if (hopc1 < threshold && hopc2 < threshold) {
				return hopc1 - hopc2;
			}

			/* both messages have more than threshold hops -> cost of the
			 * message path is used for ordering */
			p1 = getCost(from1, msg1.getTo());
			p2 = getCost(from2, msg2.getTo());

			/* the one with lower cost should be sent first */
			if (p1-p2 == 0) {
				/* if costs are equal, hop count breaks ties. If even hop counts
				   are equal, the queue ordering is used  */
				if (hopc1 == hopc2) {
					return compareByQueueMode(msg1, msg2);
				}
				else {
					return hopc1 - hopc2;
				}
			}
			else if (p1-p2 < 0) {
				return -1; // msg1 had the smaller cost
			}
			else {
				return 1; // msg2 had the smaller cost
			}
		}
	}

	/**
	 * Message-Connection tuple comparator for the MaxProp routing
	 * module. Uses {@link MaxPropComparator} on the messages of the tuples
	 * setting the "from" host for that message to be the one in the connection
	 * tuple (i.e., path is calculated starting from the host on the other end
	 * of the connection).
	 */
	private class MaxPropTupleComparator
			implements Comparator <Tuple<Message, Connection>>  {
		private int threshold;

		public MaxPropTupleComparator(int threshold) {
			this.threshold = threshold;
		}

		/**
		 * Compares two message-connection tuples using the
		 * {@link MaxPropComparator#compare(Message, Message)}.
		 *
		 * Note: We must provide a comparator that obeys the general contract
		 * (transitivity/antisymmetry). Some JVMs' TimSort will throw
		 * IllegalArgumentException if the comparator is inconsistent under load.
		 */
		public int compare(Tuple<Message, Connection> tuple1,
				Tuple<Message, Connection> tuple2) {
			Message m1 = tuple1.getKey();
			Message m2 = tuple2.getKey();
			DTNHost from1 = tuple1.getValue().getOtherNode(getHost());
			DTNHost from2 = tuple2.getValue().getOtherNode(getHost());

			if (m1 == m2 && from1 == from2) {
				return 0;
			}

			int hopc1 = m1.getHopCount();
			int hopc2 = m2.getHopCount();
			boolean prio1 = (hopc1 < threshold);
			boolean prio2 = (hopc2 < threshold);

			/* Priority region first (hop-based) */
			if (prio1 && !prio2) {
				return -1;
			}
			if (prio2 && !prio1) {
				return 1;
			}
			if (prio1 && prio2) {
				int d = hopc1 - hopc2;
				if (d != 0) {
					return d;
				}
			}

			/* Non-priority region: compare by per-tuple delivery cost */
			if (!prio1 && !prio2) {
				double c1 = getCost(from1, m1.getTo());
				double c2 = getCost(from2, m2.getTo());
				int d = Double.compare(c1, c2);
				if (d != 0) {
					return d;
				}
				/* Tie-breaks */
				d = hopc1 - hopc2;
				if (d != 0) {
					return d;
				}
			}

			/* Final deterministic tie-breaks to ensure a total order */
			int idCmp = m1.getId().compareTo(m2.getId());
			if (idCmp != 0) {
				return idCmp;
			}
			return Integer.compare(from1.getAddress(), from2.getAddress());
		}
	}


	@Override
	public RoutingInfo getRoutingInfo() {
		RoutingInfo top = super.getRoutingInfo();
		RoutingInfo ri = new RoutingInfo(probs.getAllProbs().size() +
				" meeting probabilities");

		/* show meeting probabilities for this host */
		for (Map.Entry<Integer, Double> e : probs.getAllProbs().entrySet()) {
			Integer host = e.getKey();
			Double value = e.getValue();
			ri.addMoreInfo(new RoutingInfo(String.format("host %d : %.6f",
					host, value)));
		}

		top.addMoreInfo(ri);
		top.addMoreInfo(new RoutingInfo("Avg transferred bytes: " +
				this.avgTransferredBytes));

		return top;
	}

	@Override
	public MessageRouter replicate() {
		MaxPropRouter r = new MaxPropRouter(this);
		return r;
	}
}
