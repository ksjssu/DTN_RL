#!/usr/bin/env python3
"""
Export the recurrent R-MAPPO actor (GRU) weights to a plain-text format that the
Java simulator can load for local inference (see LocalRmappoPolicy).

Usage:
    python toolkit/export_rmappo_actor.py <checkpoint.pt> <out.txt>

The checkpoint must be produced by toolkit/drl_server_rmappo.py.
"""
import argparse
import os
import sys


def _flatten(tensor):
    import torch

    if tensor is None:
        return []
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu().reshape(-1).tolist()
        return [float(x) for x in t]
    # Already list-like
    return [float(x) for x in tensor]


def _fmt(values):
    return " ".join(f"{v:.10g}" for v in values)


def _write_matrix(f, key, tensor):
    values = _flatten(tensor)
    f.write(f"{key}={_fmt(values)}\n")


def _write_vector(f, key, tensor):
    values = _flatten(tensor)
    f.write(f"{key}={_fmt(values)}\n")


def export(checkpoint_path, out_path):
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    actor_emb = ckpt.get("actor_embedding")
    actor_gru = ckpt.get("actor_gru")
    actor_head = ckpt.get("actor_head")
    if actor_emb is None or actor_gru is None or actor_head is None:
        raise RuntimeError("Checkpoint missing actor weights (expected actor_embedding, actor_gru, actor_head)")

    meta = ckpt.get("meta", {})
    hidden_size = int(meta.get("hidden_size", actor_emb["weight"].shape[0]))
    obs_dim = int(meta.get("obs_dim", actor_emb["weight"].shape[1]))
    num_layers = int(meta.get("num_layers", 1))
    log_std = ckpt.get("log_std")
    if log_std is None:
        log_std_value = 0.0
    elif isinstance(log_std, torch.Tensor):
        log_std_value = float(log_std.detach().cpu().mean().item())
    else:
        log_std_value = float(log_std)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# R-MAPPO actor weights exported for LocalRmappoPolicy\n")
        f.write("arch=rmappo_gru\n")
        f.write(f"obs_dim={obs_dim}\n")
        f.write(f"hidden_size={hidden_size}\n")
        f.write(f"num_layers={num_layers}\n")
        f.write(f"log_std={log_std_value:.10g}\n")

        # Embedding layer
        f.write(f"actor_embedding.weight_rows={actor_emb['weight'].shape[0]}\n")
        f.write(f"actor_embedding.weight_cols={actor_emb['weight'].shape[1]}\n")
        _write_matrix(f, "actor_embedding.weight", actor_emb["weight"])
        _write_vector(f, "actor_embedding.bias", actor_emb["bias"])

        # Head layer
        f.write(f"actor_head.weight_rows={actor_head['weight'].shape[0]}\n")
        f.write(f"actor_head.weight_cols={actor_head['weight'].shape[1]}\n")
        _write_matrix(f, "actor_head.weight", actor_head["weight"])
        _write_vector(f, "actor_head.bias", actor_head["bias"])

        # GRU layers
        for layer in range(num_layers):
            prefix = f"gru.l{layer}"
            weight_ih = actor_gru.get(f"weight_ih_l{layer}")
            weight_hh = actor_gru.get(f"weight_hh_l{layer}")
            bias_ih = actor_gru.get(f"bias_ih_l{layer}")
            bias_hh = actor_gru.get(f"bias_hh_l{layer}")
            if weight_ih is None or weight_hh is None:
                raise RuntimeError(f"Checkpoint missing GRU weights for layer {layer}")
            f.write(f"{prefix}.input_size={weight_ih.shape[1]}\n")
            f.write(f"{prefix}.hidden_size={hidden_size}\n")
            _write_matrix(f, f"{prefix}.weight_ih", weight_ih)
            _write_matrix(f, f"{prefix}.weight_hh", weight_hh)
            if bias_ih is None:
                bias_ih = torch.zeros(weight_ih.shape[0])
            if bias_hh is None:
                bias_hh = torch.zeros(weight_hh.shape[0])
            _write_vector(f, f"{prefix}.bias_ih", bias_ih)
            _write_vector(f, f"{prefix}.bias_hh", bias_hh)

    print(f"[export_rmappo_actor] wrote {out_path} (hidden={hidden_size}, obs_dim={obs_dim}, layers={num_layers})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to R-MAPPO checkpoint (.pt)")
    parser.add_argument("output", help="Output text file for LocalRmappoPolicy")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
