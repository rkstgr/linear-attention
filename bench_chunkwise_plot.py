"""Plot the figures for the chunkwise-linear-attention scaling post.

Reads bench_chunkwise.csv (produced by bench_chunkwise.py), writes:
  - main figure: 2x2 panels (rows = {fwd, fwd+bwd}, cols = {CPU, MPS}),
    log-log T vs ms/step, one line per impl, crossover annotations.
  - C-sweep figure: log-log C vs ms/step at T=2048, one line per device/mode.

Both figures are written to a fresh tmp dir; absolute paths printed.
"""

import csv
import math
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).resolve().parent / "bench_chunkwise.csv"

IMPLS = ["parallel", "recurrent", "chunkwise"]
COLORS = {"parallel": "#d62728", "recurrent": "#1f77b4", "chunkwise": "#2ca02c"}
DEVICES = ["cpu", "mps"]
MODES = ["fwd", "bwd"]
C_FIXED = 64
T_FOR_C_SWEEP = 2048


def load_rows():
    if not CSV_PATH.exists():
        sys.exit(f"no CSV at {CSV_PATH}; run bench_chunkwise.py first.")
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def series(rows, impl, device, mode, c=None):
    pts = []
    for r in rows:
        if r["impl"] != impl or r["device"] != device or r["mode"] != mode:
            continue
        if c is not None and r["C"] != str(c):
            continue
        pts.append((int(r["T"]), float(r["median_ms"])))
    pts.sort()
    return pts


def crossover(pts_a, pts_b):
    """Smallest T at which series A overtakes B (a > b), interpolated log-log."""
    map_a = dict(pts_a)
    map_b = dict(pts_b)
    common = sorted(set(map_a) & set(map_b))
    if len(common) < 2:
        return None
    a = [map_a[x] for x in common]
    b = [map_b[x] for x in common]
    for i in range(1, len(common)):
        if a[i - 1] <= b[i - 1] and a[i] > b[i]:
            x0, x1 = common[i - 1], common[i]
            la0, la1 = math.log(a[i - 1]), math.log(a[i])
            lb0, lb1 = math.log(b[i - 1]), math.log(b[i])
            denom = (la1 - la0) - (lb1 - lb0)
            if abs(denom) < 1e-12:
                return x0
            alpha = (lb0 - la0) / denom
            lx0, lx1 = math.log(x0), math.log(x1)
            return math.exp(lx0 + alpha * (lx1 - lx0))
    return None


def plot_main(rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey="row")
    for i, mode in enumerate(MODES):
        for j, device in enumerate(DEVICES):
            ax = axes[i][j]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.2)
            if i == 0:
                ax.set_title(device.upper())
            if j == 0:
                row_label = "fwd" if mode == "fwd" else "fwd+bwd"
                ax.set_ylabel(f"{row_label}\nms / step")
            if i == 1:
                ax.set_xlabel("T (sequence length)")

            data = {
                impl: series(rows, impl, device, mode,
                             C_FIXED if impl == "chunkwise" else None)
                for impl in IMPLS
            }
            if not any(data.values()):
                ax.text(0.5, 0.5, f"no data\nrun --device {device}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=10, color="gray")
                continue

            for impl in IMPLS:
                pts = data[impl]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=COLORS[impl], label=impl, lw=2, ms=5)

            par, rec, chk = data["parallel"], data["recurrent"], data["chunkwise"]
            annots = []
            if par and chk:
                t = crossover(par, chk)
                if t is not None:
                    annots.append((t, "parallel > chunkwise", COLORS["parallel"]))
            if rec and par and chk:
                t_rp = crossover(rec, par)
                t_rc = crossover(rec, chk)
                if t_rp is not None and t_rc is not None:
                    annots.append((max(t_rp, t_rc), "recurrent > both", COLORS["recurrent"]))
            for t, label, color in annots:
                ax.axvline(t, color=color, ls="--", alpha=0.4, lw=1)
                y_top = ax.get_ylim()[1]
                y_bot = ax.get_ylim()[0]
                y_pos = math.exp(0.85 * math.log(y_top) + 0.15 * math.log(y_bot))
                ax.text(t * 1.05, y_pos, f"{label}\nT≈{int(round(t))}",
                        color=color, fontsize=8, va="top")

            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Linear attention scaling: parallel vs recurrent vs chunkwise "
                 f"(head_dim=16, C={C_FIXED})", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_csweep(rows, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlabel("chunk size C")
    ax.set_ylabel("ms / step")
    ax.set_title(f"Chunkwise sweet spot at T={T_FOR_C_SWEEP}")

    has_data = False
    style = {("cpu", "fwd"): ("C0", "-",  "o"),
             ("cpu", "bwd"): ("C0", "--", "s"),
             ("mps", "fwd"): ("C1", "-",  "o"),
             ("mps", "bwd"): ("C1", "--", "s"),
             ("cuda", "fwd"): ("C2", "-",  "o"),
             ("cuda", "bwd"): ("C2", "--", "s")}
    for device in DEVICES + ["cuda"]:
        for mode in MODES:
            pts = []
            for r in rows:
                if (r["impl"] == "chunkwise" and r["device"] == device
                        and r["mode"] == mode and r["T"] == str(T_FOR_C_SWEEP)
                        and r["C"]):
                    pts.append((int(r["C"]), float(r["median_ms"])))
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            color, ls, marker = style[(device, mode)]
            ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                    label=f"{device} {mode}", lw=2, ms=5)
            has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "no C-sweep data\nrun --c-sweep",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
    else:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_rows()
    out_dir = Path(tempfile.mkdtemp(prefix="bench_chunkwise_"))
    main_path = out_dir / "bench_chunkwise.png"
    csweep_path = out_dir / "bench_chunkwise_C_sweep.png"
    plot_main(rows, main_path)
    plot_csweep(rows, csweep_path)
    print(f"main figure:    {main_path}")
    print(f"C-sweep figure: {csweep_path}")


if __name__ == "__main__":
    main()
