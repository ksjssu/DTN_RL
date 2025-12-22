#!/usr/bin/env python3
"""
Export PPO actor weights for decentralized execution.
Usage:
    python toolkit/export_policy.py models/ppo_checkpoint.pt models/ppo_actor_weights.txt
The output file is a plain-text key=value format parsed by LocalPpoPolicy.
"""
import argparse
import pathlib

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required to export policies: {exc}")


def _flatten(tensor):
    return " ".join(f"{float(x):.10f}" for x in tensor.reshape(-1))


def main():
    parser = argparse.ArgumentParser(description="Export PPO actor network weights")
    parser.add_argument("checkpoint", help="Path to PPO checkpoint (.pt)")
    parser.add_argument("output", help="Destination txt file")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    actor_state = ckpt.get("actor")
    if actor_state is None:
        raise SystemExit("Checkpoint does not contain 'actor' state_dict")

    # Extract layers (Linear -> Tanh -> ...)
    w1 = actor_state.get("0.weight")
    b1 = actor_state.get("0.bias")
    w2 = actor_state.get("2.weight")
    b2 = actor_state.get("2.bias")
    w3 = actor_state.get("4.weight")
    b3 = actor_state.get("4.bias")
    if None in (w1, b1, w2, b2, w3, b3):
        raise SystemExit("Unsupported actor architecture; expected 3 linear layers")

    # Transpose to match Java expectation: [input][output]
    w1_t = w1.t().contiguous()
    w2_t = w2.t().contiguous()
    w3_t = w3.t().contiguous()

    input_dim = w1_t.shape[0]
    hidden_dim1 = w1_t.shape[1]
    hidden_dim2 = w2_t.shape[1]
    output_dim = w3_t.shape[1]

    log_std_tensor = ckpt.get("log_std")
    log_std = float(log_std_tensor.view(-1)[0]) if log_std_tensor is not None else 0.0

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# PPO actor weights exported for LocalPpoPolicy\n")
        f.write(f"input_dim={input_dim}\n")
        f.write(f"hidden_dim1={hidden_dim1}\n")
        f.write(f"hidden_dim2={hidden_dim2}\n")
        f.write(f"output_dim={output_dim}\n")
        f.write(f"log_std={log_std:.10f}\n")
        f.write("W1=" + _flatten(w1_t) + "\n")
        f.write("b1=" + _flatten(b1) + "\n")
        f.write("W2=" + _flatten(w2_t) + "\n")
        f.write("b2=" + _flatten(b2) + "\n")
        f.write("W3=" + _flatten(w3_t) + "\n")
        f.write("b3=" + _flatten(b3) + "\n")

    print(f"Exported actor weights to {out_path}")


if __name__ == "__main__":
    main()
