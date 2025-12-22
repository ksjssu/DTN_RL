#!/usr/bin/env python3
import csv
import os
import sys

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "time": int(float(row.get("time", 0) or 0)),
                    "episode": int(float(row.get("episode", 0) or 0)),
                    "updates": int(float(row.get("updates", 0) or 0)),
                    "sim_id": row.get("sim_id", ""),
                    "buffer": int(float(row.get("buffer", 0) or 0)),
                    "trained_on": int(float(row.get("trained_on", 0) or 0)),
                    "a_loss": float(row.get("a_loss", 0) or 0),
                    "c_loss": float(row.get("c_loss", 0) or 0),
                    "cval_loss": float(row.get("cval_loss", 0) or 0),
                    "entropy": float(row.get("entropy", 0) or 0),
                    "clip_frac": float(row.get("clip_frac", 0) or 0),
                    "approx_kl": float(row.get("approx_kl", 0) or 0),
                })
            except Exception:
                continue
    return rows

def plot_file(path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available: {e}. Install it or open CSV in your tool.")
        return
    rows = read_csv(path)
    if not rows:
        print(f"No data in {path}")
        return
    x = [r["updates"] for r in rows]
    a = [r["a_loss"] for r in rows]
    c = [r["c_loss"] for r in rows]
    cv = [r["cval_loss"] for r in rows]
    ent = [r["entropy"] for r in rows]
    cf = [r["clip_frac"] for r in rows]
    kl = [r["approx_kl"] for r in rows]
    buf = rows[-1].get("buffer", 0)

    base = os.path.splitext(os.path.basename(path))[0]
    out1 = os.path.join(os.path.dirname(path), f"{base}_losses.png")
    out2 = os.path.join(os.path.dirname(path), f"{base}_regularization.png")

    plt.figure(figsize=(8,4))
    plt.plot(x, a, label='a_loss')
    plt.plot(x, c, label='c_loss')
    plt.plot(x, cv, label='cval_loss')
    plt.xlabel('update')
    plt.ylabel('loss')
    title = f"Losses ({base})" + (f" buf={buf}M" if buf else "")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(x, ent, label='entropy')
    plt.plot(x, cf, label='clip_frac')
    plt.plot(x, kl, label='approx_kl')
    plt.xlabel('update')
    plt.ylabel('metric')
    title = f"Reg Metrics ({base})" + (f" buf={buf}M" if buf else "")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Wrote {out1} and {out2}")

def main():
    paths = sys.argv[1:]
    if not paths:
        # scan default pattern
        base = os.path.join('reports')
        if not os.path.isdir(base):
            print("No reports directory found.")
            return
        for name in os.listdir(base):
            if name.startswith('drl_losses_port') and name.endswith('.csv'):
                paths.append(os.path.join(base, name))
    if not paths:
        print("No CSVs provided/found.")
        return
    for p in paths:
        try:
            plot_file(p)
        except Exception as e:
            print(f"Failed plotting {p}: {e}")

if __name__ == '__main__':
    main()

