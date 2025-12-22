package report;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * Lightweight PPO actor for decentralized execution. Loads weights exported
 * from the Python PPO server and performs deterministic forward inference.
 */
class LocalPpoPolicy {

    private final int inputDim;
    private final int hidden1;
    private final int hidden2;
    private final int outputDim;

    private final double[][] w1; // [inputDim][hidden1]
    private final double[] b1;   // [hidden1]
    private final double[][] w2; // [hidden1][hidden2]
    private final double[] b2;   // [hidden2]
    private final double[][] w3; // [hidden2][outputDim]
    private final double[] b3;   // [outputDim]
    private final double logStd;

    LocalPpoPolicy(String path) throws IOException {
        Map<String, String> kv = loadKeyValues(path);
        try {
            this.inputDim = Integer.parseInt(require(kv, "input_dim"));
            this.hidden1 = Integer.parseInt(require(kv, "hidden_dim1"));
            this.hidden2 = Integer.parseInt(require(kv, "hidden_dim2"));
            this.outputDim = Integer.parseInt(require(kv, "output_dim"));
            String logStdStr = kv.get("log_std");
            this.logStd = (logStdStr == null) ? 0.0 : Double.parseDouble(logStdStr);

            this.w1 = parseMatrix(require(kv, "W1"), inputDim, hidden1);
            this.b1 = parseVector(require(kv, "b1"), hidden1);
            this.w2 = parseMatrix(require(kv, "W2"), hidden1, hidden2);
            this.b2 = parseVector(require(kv, "b2"), hidden2);
            this.w3 = parseMatrix(require(kv, "W3"), hidden2, outputDim);
            this.b3 = parseVector(require(kv, "b3"), outputDim);
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid policy weight file: " + path + " (" + e.getMessage() + ")", e);
        }
    }

    /**
     * Deterministic action: tanh(mean) * deltaLimit. If deltaLimit <= 0 the
     * result is left unscaled.
     */
    double infer(double[] obs, double deltaLimit) {
        if (obs.length != inputDim) {
            throw new IllegalArgumentException("Observation dim mismatch: expected " + inputDim + " got " + obs.length);
        }

        double[] layer1 = new double[hidden1];
        for (int j = 0; j < hidden1; j++) {
            double sum = b1[j];
            for (int i = 0; i < inputDim; i++) {
                sum += obs[i] * w1[i][j];
            }
            layer1[j] = relu(sum);
        }

        double[] layer2 = new double[hidden2];
        for (int j = 0; j < hidden2; j++) {
            double sum = b2[j];
            for (int i = 0; i < hidden1; i++) {
                sum += layer1[i] * w2[i][j];
            }
            layer2[j] = relu(sum);
        }

        double mean = b3[0];
        for (int i = 0; i < hidden2; i++) {
            mean += layer2[i] * w3[i][0];
        }
        double action = Math.tanh(mean);
        double scale = (deltaLimit > 0) ? deltaLimit : 1.0;
        return action * scale;
    }

    double getLogStd() {
        return this.logStd;
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
        }
        finally {
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
        List<String> out = new ArrayList<String>();
        while (tok.hasMoreTokens()) {
            out.add(tok.nextToken());
        }
        return out.toArray(new String[out.size()]);
    }

    private static double relu(double x) {
        return (x > 0.0) ? x : 0.0;
    }
}
