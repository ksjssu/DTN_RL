package routing.maxprop;

/**
 * Optional node penalty callback for MaxPropDijkstra.
 * Implementations can add an additional non-negative cost when the shortest path
 * relaxes an edge that enters the given node.
 */
public interface NodePenaltyFunction {
    /**
     * @param nodeAddr node address (DTNHost address)
     * @return additional non-negative penalty cost (0 for none)
     */
    double getPenalty(Integer nodeAddr);
}

