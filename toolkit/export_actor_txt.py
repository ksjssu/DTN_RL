#!/usr/bin/env python3
"""
Export PPO actor weights from a .pt checkpoint (toolkit/drl_server.py format)
to LocalPpoPolicy.txt format used by Java (src/report/LocalPpoPolicy.java).

Assumptions:
- Actor is an MLP: input_dim -> hidden -> hidden -> 1
- Hidden nonlinearity is ReLU (but weight export is agnostic)
- Checkpoint contains keys: 'actor' (state_dict), 'log_std' (tensor)

Usage:
  python toolkit/export_actor_txt.py models/buf05/dtn_sim_t99900_ppo.pt out.txt
"""
import sys
import os


def load_ckpt(path):
    import torch
    return torch.load(path, map_location='cpu')


def find_linear_layers(sd):
    """Return list of (w,b) for first three Linear layers in order 0,2,4."""
    # typical keys: '0.weight','0.bias','2.weight','2.bias','4.weight','4.bias'
    pairs = []
    # sort keys numerically by layer index and pick .weight/.bias pairs
    items = sorted([(int(k.split('.')[0]), k) for k in sd.keys() if k.endswith('.weight')])
    for idx, wkey in items:
        bkey = f"{idx}.bias"
        if bkey in sd:
            pairs.append((sd[wkey], sd[bkey]))
    if len(pairs) < 3:
        raise RuntimeError(f"Expected at least 3 Linear layers, found {len(pairs)}")
    return pairs[:3]


def to_java_matrix(w):
    """PyTorch weight is [out, in]; Java expects W as [in][out] flattened row-major.
    Return list of floats in order matching rows=in, cols=out.
    """
    w = w.detach().cpu().float()
    out, in_ = w.shape
    # transpose to [in, out]
    wt = w.t().contiguous().view(-1)
    return wt.tolist()


def to_java_vector(b):
    b = b.detach().cpu().float().view(-1)
    return b.tolist()


def write_txt(path, input_dim, h1, h2, out_dim, log_std, W1, b1, W2, b2, W3, b3):
    def fmt(arr):
        return ' '.join(f"{x:.10g}" for x in arr)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# PPO actor weights exported for LocalPpoPolicy\n")
        f.write(f"input_dim={input_dim}\n")
        f.write(f"hidden_dim1={h1}\n")
        f.write(f"hidden_dim2={h2}\n")
        f.write(f"output_dim={out_dim}\n")
        f.write(f"log_std={float(log_std):.10g}\n")
        f.write(f"W1={fmt(W1)}\n")
        f.write(f"b1={fmt(b1)}\n")
        f.write(f"W2={fmt(W2)}\n")
        f.write(f"b2={fmt(b2)}\n")
        f.write(f"W3={fmt(W3)}\n")
        f.write(f"b3={fmt(b3)}\n")


def main():
    if len(sys.argv) < 3:
        print("Usage: export_actor_txt.py <in.pt> <out.txt>")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    ckpt = load_ckpt(in_path)
    actor_sd = ckpt.get('actor')
    if actor_sd is None:
        raise RuntimeError("Checkpoint missing 'actor' state_dict")
    log_std = ckpt.get('log_std', 0.0)
    try:
        import torch
        if isinstance(log_std, torch.Tensor):
            log_std = log_std.mean().item()
        else:
            log_std = float(log_std)
    except Exception:
        log_std = float(log_std)

    layers = find_linear_layers(actor_sd)
    (w1, b1), (w2, b2), (w3, b3) = layers
    out1, in1 = w1.shape
    out2, in2 = w2.shape
    out3, in3 = w3.shape
    # consistency checks
    input_dim = in1
    h1 = out1
    if in2 != h1:
        raise RuntimeError(f"Mismatch: second layer in={in2} != hidden_dim1={h1}")
    h2 = out2
    if in3 != h2 or out3 != 1:
        raise RuntimeError("Expected final layer to map hidden2 -> 1")
    out_dim = 1

    W1 = to_java_matrix(w1)
    B1 = to_java_vector(b1)
    W2 = to_java_matrix(w2)
    B2 = to_java_vector(b2)
    W3 = to_java_matrix(w3)
    B3 = to_java_vector(b3)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    write_txt(out_path, input_dim, h1, h2, out_dim, log_std, W1, B1, W2, B2, W3, B3)
    print(f"Exported actor weights to {out_path} (dims: {input_dim}->{h1}->{h2}->1, log_std={log_std:.4g})")


if __name__ == '__main__':
    main()

