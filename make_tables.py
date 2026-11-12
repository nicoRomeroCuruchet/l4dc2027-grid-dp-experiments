"""Print Table 1, Table 2 and the data of Figure 1 of the paper from a results log.

Usage:
    python make_tables.py                       # uses results/nonlin.jsonl (the runs reported in the paper)
    python make_tables.py my_runs.jsonl         # a log produced by experiments/resolution/nonlin_sweep.py
    python make_tables.py --lqr                 # also the LQR columns of Table 1 (needs numpy and cupy)

Without --lqr it needs only the Python standard library.

Each line of the log is one grid: system, box size R (= ell in the paper), q, M, the axis gain in
multiples of kappa, rho(A + B K), and the median amplitude of the closed loop in cells.
"""
import argparse
import json
from pathlib import Path


def load(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if line.startswith("{") and '"M"' in line:
            r = json.loads(line)
            if r["system"] == "dpend":
                recs.append(r)
    return recs


def runs(recs, R, q):
    return sorted((r for r in recs if r["R"] == R and r["q"] == q), key=lambda r: r["M"])


def plateau(rs):
    """Last gain and the smallest computed M from which every computed gain equals it."""
    last = rs[-1]["K_hat_over_kappa"]
    start = rs[-1]["M"]
    for r in reversed(rs):
        if r["K_hat_over_kappa"] != last:
            break
        start = r["M"]
    return last, start, rs[-1]["rho_lin_Khat"]


def lqr_columns(qs, R=0.4):
    """K*/kappa, its rounding, and rho(A + B K) for both. In cells they do not depend on M."""
    import sys
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiments.resolution.selfsim import dlqr_discounted, jacobians
    from experiments.resolution.stabilization_sweep import SYSTEMS, TAU, box
    spec = SYSTEMS["dpend"]
    J, b = jacobians()
    A = np.eye(4) + TAU * J
    B = (TAU * b).reshape(4, 1)
    K, _ = dlqr_discounted(A, B, np.diag(spec["cost_w"]), spec["u_w"], 0.99)
    K = K.ravel()
    half, _ = box(spec, R)
    rho = lambda g: float(max(abs(np.linalg.eigvals(A + B @ g.reshape(1, 4)))))
    out = {}
    for q in qs:
        du_per_M = q * float(np.abs(K) @ half)          # action step times M
        k_cells = K * half / du_per_M                    # K*/kappa
        k_round = np.rint(k_cells)
        out[q] = (k_cells, rho(K), k_round, rho(k_round * du_per_M / half))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", default=str(Path(__file__).resolve().parent / "results" / "nonlin.jsonl"))
    p.add_argument("--lqr", action="store_true", help="add the LQR columns of Table 1 (needs cupy)")
    args = p.parse_args()
    recs = load(args.log)
    fmt = lambda v: "(" + ", ".join(str(int(x)) for x in v) + ")"

    print("Table 1. Double pendulum: observed axis gain and the M from which it is constant")
    qs = (0.2, 0.3, 0.5)
    lqr = lqr_columns(qs) if args.lqr else None
    for q in qs:
        cols = []
        k_obs = rho_obs = None
        for R in (0.4, 0.8):
            rs = runs(recs, R, q)
            if not rs:
                cols.append("ell=%g: no runs" % R)
                continue
            k, start, rho = plateau(rs)
            cols.append("ell=%g: constant from M=%d (computed up to M=%d)" % (R, start, rs[-1]["M"]))
            if R == 0.4:
                k_obs, rho_obs = k, rho
        line = "  q=%g  %s" % (q, "; ".join(cols))
        if k_obs is not None:
            line += "  K_obs/kappa=%s rho=%.2f" % (fmt(k_obs), rho_obs)
        print(line)
        if lqr:
            kc, rc, kr, rr = lqr[q]
            print("        K*/kappa=(%s) rho=%.2f   rounded K*/kappa=%s rho=%.2f"
                  % (", ".join("%.2f" % x for x in kc), rc, fmt(kr), rr))

    rs = runs(recs, 0.4, 0.2)
    print("\nTable 2. Double pendulum, q=0.2, ell=0.4: axis gain and spectral radius for each range of M")
    i = 0
    while i < len(rs):
        j = i
        while j + 1 < len(rs) and rs[j + 1]["K_hat_over_kappa"] == rs[i]["K_hat_over_kappa"]:
            j += 1
        rng = str(rs[i]["M"]) if i == j else "%d-%d" % (rs[i]["M"], rs[j]["M"])
        print("  M=%-6s K_M/kappa=%-16s rho=%.2f" % (rng, fmt(rs[i]["K_hat_over_kappa"]), rs[i]["rho_lin_Khat"]))
        i = j + 1

    print("\nFigure 1. Same runs: median amplitude in the last quarter of each run")
    print("  M    amplitude (cells)   amplitude x delta   pi(0)=0 and antisymmetric on the axes")
    for r in rs:
        amp = r["near"]["amp_med"]
        ok = r["symmetric_axes"] and r["pi0"] == 0.0
        print("  %-4d %-19.3f %-19.4f %s" % (r["M"], amp, amp * r["delta"], ok))


if __name__ == "__main__":
    main()
