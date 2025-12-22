package report;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * Lightweight GRU-based policy loader for locally executing the R-MAPPO actor.
 * <p>
 * The weight file is produced by toolkit/export_rmappo_actor.py and is a simple
 * key=value text file.
 * </p>
 */
class LocalRmappoPolicy {

    private static final String ARCH_TAG = "rmappo_gru";

    private final int obsDim;
    private final int hiddenSize;
    private final int numLayers;

    private final double[][] embedWeight; // [hidden][obs]
    private final double[] embedBias;     // [hidden]
    private final double[] headWeight;    // [hidden]
    private final double headBias;

    private final GruLayer[] layers;

    private final Map<String, double[][]> hiddenByKey = new HashMap<String, double[][]>();

    private static class GruLayer {
        double[][] weightIh; // [3*hidden][input]
        double[][] weightHh; // [3*hidden][hidden]
        double[] biasIh;     // [3*hidden]
        double[] biasHh;     // [3*hidden]
        int inputSize;
    }

    static boolean isRmappoFormat(String path) {
        File f = new File(path);
        if (!f.exists()) {
            return false;
        }
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(f));
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                if (line.startsWith("arch=")) {
                    return line.substring(5).trim().equalsIgnoreCase(ARCH_TAG);
                }
            }
        } catch (IOException ignored) {
            return false;
        } finally {
            if (br != null) {
                try { br.close(); } catch (IOException ignore) {}
            }
        }
        return false;
    }

    LocalRmappoPolicy(String path) throws IOException {
        Map<String, String> kv = loadKeyValues(path);
        String arch = kv.get("arch");
        if (arch == null || !arch.equalsIgnoreCase(ARCH_TAG)) {
            throw new IOException("Policy file is not rmappo_gru format: " + path);
        }
        this.obsDim = Integer.parseInt(require(kv, "obs_dim"));
        this.hiddenSize = Integer.parseInt(require(kv, "hidden_size"));
        this.numLayers = Integer.parseInt(require(kv, "num_layers"));

        this.embedWeight = parseMatrix(require(kv, "actor_embedding.weight"), hiddenSize, obsDim);
        this.embedBias = parseVector(require(kv, "actor_embedding.bias"), hiddenSize);
        double[][] headW = parseMatrix(require(kv, "actor_head.weight"), 1, hiddenSize);
        this.headWeight = headW[0];
        double[] headB = parseVector(require(kv, "actor_head.bias"), 1);
        this.headBias = headB[0];

        this.layers = new GruLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            GruLayer layer = new GruLayer();
            layer.inputSize = Integer.parseInt(require(kv, "gru.l" + i + ".input_size"));
            layer.weightIh = parseMatrix(require(kv, "gru.l" + i + ".weight_ih"), hiddenSize * 3, layer.inputSize);
            layer.weightHh = parseMatrix(require(kv, "gru.l" + i + ".weight_hh"), hiddenSize * 3, hiddenSize);
            layer.biasIh = parseVector(require(kv, "gru.l" + i + ".bias_ih"), hiddenSize * 3);
            layer.biasHh = parseVector(require(kv, "gru.l" + i + ".bias_hh"), hiddenSize * 3);
            this.layers[i] = layer;
        }
    }

    /**
     * Run policy inference for the given host-destination key.
     *
     * @param key        unique key per host#dest pair
     * @param obs        local observation (length = obsDim)
     * @param deltaLimit scaling factor for the final tanh output
     * @return scaled delta action
     */
    double infer(String key, double[] obs, double deltaLimit) {
        if (obs.length != this.obsDim) {
            throw new IllegalArgumentException("Expected obs_dim=" + this.obsDim + ", got " + obs.length);
        }

        double[] embedded = new double[this.hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = embedBias[j];
            for (int i = 0; i < obsDim; i++) {
                sum += embedWeight[j][i] * obs[i];
            }
            embedded[j] = Math.tanh(sum);
        }

        double[][] state = hiddenByKey.get(key);
        if (state == null) {
            state = new double[numLayers][hiddenSize];
            hiddenByKey.put(key, state);
        }

        double[] layerInput = embedded;
        for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            GruLayer layer = layers[layerIdx];
            double[] hPrev = state[layerIdx];
            double[] gatesI = matVec(layer.weightIh, layerInput);
            double[] gatesH = matVec(layer.weightHh, hPrev);
            addInPlace(gatesI, layer.biasIh);
            addInPlace(gatesH, layer.biasHh);

            double[] hNew = new double[hiddenSize];
            int offsetR = 0;
            int offsetZ = hiddenSize;
            int offsetN = hiddenSize * 2;
            for (int i = 0; i < hiddenSize; i++) {
                double r = sigmoid(gatesI[offsetR + i] + gatesH[offsetR + i]);
                double z = sigmoid(gatesI[offsetZ + i] + gatesH[offsetZ + i]);
                double n = Math.tanh(gatesI[offsetN + i] + r * gatesH[offsetN + i]);
                double candidate = (1.0 - z) * n + z * hPrev[i];
                hNew[i] = candidate;
            }
            state[layerIdx] = hNew;
            layerInput = hNew;
        }

        double mean = headBias;
        for (int i = 0; i < hiddenSize; i++) {
            mean += headWeight[i] * layerInput[i];
        }

        double action = Math.tanh(mean);
        double scale = (deltaLimit > 0) ? deltaLimit : 1.0;
        return action * scale;
    }

    void resetState(String key) {
        hiddenByKey.remove(key);
    }

    void clear() {
        hiddenByKey.clear();
    }

    private static double[] matVec(double[][] matrix, double[] vec) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] out = new double[rows];
        for (int r = 0; r < rows; r++) {
            double sum = 0.0;
            double[] row = matrix[r];
            for (int c = 0; c < cols; c++) {
                sum += row[c] * vec[c];
            }
            out[r] = sum;
        }
        return out;
    }

    private static void addInPlace(double[] target, double[] bias) {
        for (int i = 0; i < target.length; i++) {
            target[i] += bias[i];
        }
    }

    private static double sigmoid(double x) {
        if (x >= 0) {
            double z = Math.exp(-x);
            return 1.0 / (1.0 + z);
        } else {
            double z = Math.exp(x);
            return z / (1.0 + z);
        }
    }

    private static Map<String, String> loadKeyValues(String path) throws IOException {
        File f = new File(path);
        if (!f.exists()) {
            throw new IOException("Policy weight file not found: " + path);
        }
        Map<String, String> kv = new LinkedHashMap<String, String>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(f));
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                int idx = line.indexOf('=');
                if (idx < 0) {
                    continue;
                }
                String key = line.substring(0, idx).trim();
                String value = line.substring(idx + 1).trim();
                kv.put(key, value);
            }
        } finally {
            if (br != null) {
                try { br.close(); } catch (IOException ignore) {}
            }
        }
        return kv;
    }

    private static String require(Map<String, String> kv, String key) {
        String v = kv.get(key);
        if (v == null) {
            throw new IllegalArgumentException("Missing key " + key);
        }
        return v;
    }

    private static double[] parseVector(String value, int expected) {
        String[] tokens = splitTokens(value);
        if (tokens.length != expected) {
            throw new IllegalArgumentException("Expected " + expected + " values, got " + tokens.length);
        }
        double[] out = new double[expected];
        for (int i = 0; i < expected; i++) {
            out[i] = Double.parseDouble(tokens[i]);
        }
        return out;
    }

    private static double[][] parseMatrix(String value, int rows, int cols) {
        String[] tokens = splitTokens(value);
        if (tokens.length != rows * cols) {
            throw new IllegalArgumentException("Expected " + (rows * cols) + " values, got " + tokens.length);
        }
        double[][] out = new double[rows][cols];
        int idx = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = Double.parseDouble(tokens[idx++]);
            }
        }
        return out;
    }

    private static String[] splitTokens(String value) {
        StringTokenizer tok = new StringTokenizer(value, ", \t\r\n");
        List<String> out = new java.util.ArrayList<String>();
        while (tok.hasMoreTokens()) {
            out.add(tok.nextToken());
        }
        return out.toArray(new String[out.size()]);
    }
}
