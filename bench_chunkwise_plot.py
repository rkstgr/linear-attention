"""Plot the figures for the chunkwise-linear-attention scaling post.

Reads bench_chunkwise.csv (produced by bench_chunkwise.py), writes:
  - main figure: 2 x N panels (rows = {fwd, fwd+bwd}, columns = devices in CSV).
    Log-log T vs ms/step, one line per impl, crossover annotations per panel.
  - C-sweep figure: 1 x M panels (one per distinct T in CSV that has a sweep),
    log-log C vs ms/step, one line per (device, mode).

Both figures land in a fresh tmp dir; absolute paths are printed.
"""

import csv
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

CSV_PATH = Path(__file__).resolve().parent / "bench_chunkwise.csv"

IMPLS = ["parallel", "recurrent", "chunkwise"]
COLORS_IMPL = {"parallel": "#d62728", "recurrent": "#1f77b4", "chunkwise": "#2ca02c"}
ALL_DEVICES = ["cpu", "mps", "cuda"]              # canonical column order
COLORS_DEVICE = {"cpu": "C0", "mps": "C1", "cuda": "C2"}
MODES = ["fwd", "bwd"]
C_FIXED = 64                                       # used in main figure


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
    map_a, map_b = dict(pts_a), dict(pts_b)
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
            return math.exp(math.log(x0) + alpha * (math.log(x1) - math.log(x0)))
    return None


def devices_with_tsweep(rows):
    """Devices that have at least one non-chunkwise T-sweep row in the CSV."""
    have = {r["device"] for r in rows
            if r["impl"] in ("parallel", "recurrent")
            or (r["impl"] == "chunkwise" and r["C"] == str(C_FIXED))}
    return [d for d in ALL_DEVICES if d in have]


def plot_main(rows, out_path):
    devices = devices_with_tsweep(rows)
    if not devices:
        return None
    n = len(devices)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7),
                             sharex=True, sharey="row", squeeze=False)
    for i, mode in enumerate(MODES):
        for j, device in enumerate(devices):
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
            for impl in IMPLS:
                pts = data[impl]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", color=COLORS_IMPL[impl],
                        label=impl, lw=2, ms=5)

            par, rec, chk = data["parallel"], data["recurrent"], data["chunkwise"]
            annots = []
            if par and chk:
                t = crossover(par, chk)
                if t is not None:
                    annots.append((t, "parallel > chunkwise", COLORS_IMPL["parallel"]))
            if rec and par and chk:
                t_rp = crossover(rec, par)
                t_rc = crossover(rec, chk)
                if t_rp is not None and t_rc is not None:
                    annots.append((max(t_rp, t_rc), "recurrent > both",
                                   COLORS_IMPL["recurrent"]))
            for t, label, color in annots:
                ax.axvline(t, color=color, ls="--", alpha=0.4, lw=1)
                y_top, y_bot = ax.get_ylim()[1], ax.get_ylim()[0]
                y = math.exp(0.85 * math.log(y_top) + 0.15 * math.log(y_bot))
                ax.text(t * 1.05, y, f"{label}\nT≈{int(round(t))}",
                        color=color, fontsize=8, va="top")

            if i == 0 and j == 0:
                ax.legend(loc="upper left", fontsize=9)

    fig.suptitle("Linear attention scaling (head_dim=16, "
                 f"chunkwise C={C_FIXED})", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def csweep_groups(rows):
    """Map T -> {device -> [(C, ms_fwd, ms_bwd) sorted by C]}."""
    by = defaultdict(lambda: defaultdict(dict))  # by[T][device][(C, mode)] = ms
    for r in rows:
        if r["impl"] != "chunkwise" or not r["C"]:
            continue
        by[int(r["T"])][r["device"]][(int(r["C"]), r["mode"])] = float(r["median_ms"])
    # keep T values where some device has at least 2 distinct C values
    result = {}
    for T, dev_map in by.items():
        keep = {}
        for device, cm_map in dev_map.items():
            cs = sorted({c for (c, _) in cm_map})
            if len(cs) >= 2:
                keep[device] = cm_map
        if keep:
            result[T] = keep
    return result


def plot_csweep(rows, out_path):
    groups = csweep_groups(rows)
    if not groups:
        return None
    Ts = sorted(groups)
    n = len(Ts)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), squeeze=False)
    for j, T in enumerate(Ts):
        ax = axes[0][j]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlabel("chunk size C")
        if j == 0:
            ax.set_ylabel("ms / step")
        ax.set_title(f"T = {T}")

        for device in ALL_DEVICES:
            if device not in groups[T]:
                continue
            cm_map = groups[T][device]
            for mode, ls, marker in [("fwd", "-", "o"), ("bwd", "--", "s")]:
                pts = sorted((c, ms) for (c, m), ms in cm_map.items() if m == mode)
                if len(pts) < 2:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=marker, linestyle=ls,
                        color=COLORS_DEVICE[device],
                        label=f"{device} {mode}", lw=2, ms=5)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Chunkwise sweet spot (head_dim=16)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    rows = load_rows()
    out_dir = Path(tempfile.mkdtemp(prefix="bench_chunkwise_"))
    main_path = plot_main(rows, out_dir / "bench_chunkwise.png")
    csweep_path = plot_csweep(rows, out_dir / "bench_chunkwise_C_sweep.png")
    if main_path:
        print(f"main figure:    {main_path}")
    else:
        print("main figure:    skipped (no T-sweep data)")
    if csweep_path:
        print(f"C-sweep figure: {csweep_path}")
    else:
        print("C-sweep figure: skipped (no C-sweep data; run --c-sweep)")


if __name__ == "__main__":
    main()
