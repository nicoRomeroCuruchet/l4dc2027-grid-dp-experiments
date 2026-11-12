"""
experiments/resolution/nonlin_sweep.py — refinement with a FIXED box on the NONLINEAR pendulums.

Same protocol as unit_sweep.py (fixed physical box, action step tied to the cell through q, exactly
quadratic stage cost with exit penalty), but with the true nonlinear dynamics:
    dpend — base-actuated double pendulum (4D, one torque), dynamics of stabilization_sweep.SYSTEMS
    pend  — simple pendulum θ̈ = λ² sin θ + b u (2D), λ = fastest unstable rate of dpend, b = 20
q = κ / Σ|K*_i| uses the LQR gain of the linearisation at the upright equilibrium (cell coordinates).

For every (system, R, q, bins) we record, in cell units:
    amplitude over the last quarter (median / max) from 'near' (±3 cells) and 'far' (±8 cells) starts,
    fraction converged (< 1e-6 cells), fraction that left the box, max excursion of the trajectories,
    axis gain K̂/κ, symmetry on the axes, π(0), and the smallest best-vs-second action gap among the
    axis neighbours and among all grid points within 3 cells (in units of c_box).
The action table is saved compressed (int16) for later table-vs-table comparisons.
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.resolution.selfsim import dlqr_discounted, jacobians
from experiments.resolution.stabilization_sweep import SYSTEMS, TAU, box, interp, make_solver, step_py
from src.cuda_policy_iteration import CudaPIConfig, CudaPolicyIteration2D


def nonlin_pend_spec(lam, b):
    spec = dict(names=["th", "w"], angle_dims=[0], vel_dims=[1], fixed={},
                acc_c=f"float a1 = ({lam * lam:.8f}f)*sinf(th) + ({b:.8f}f)*u;", deriv=["w", "a1"],
                acc_py=lambda s, u: [s[1], lam * lam * cp.sin(s[0]) + b * u],
                cost_w=[1.0, 0.1], u_w=0.01, base=CudaPolicyIteration2D)
    return spec, np.array([[0.0, 1.0], [lam * lam, 0.0]]), np.array([0.0, b])


class Grid:
    def __init__(self, pi, half, dvec):
        d = len(half)
        self.d, self.half, self.dvec = d, cp.asarray(half), cp.asarray(dvec)
        self.lo = cp.asarray(pi.bounds_low, cp.float64)
        self.hi = cp.asarray(pi.bounds_high, cp.float64)
        self.shape = cp.asarray(pi.grid_shape, cp.int64)
        self.strides = cp.asarray(pi.strides, cp.int64)
        self.corners = cp.asarray(list(product([0, 1], repeat=d)), dtype=cp.int64)
        self.M = (int(pi.grid_shape[0]) - 1) // 2

    def weights(self, X):
        return interp([X[:, i] for i in range(self.d)], self.lo, self.hi, self.shape, self.strides, self.corners)

    def flat(self, k):
        return int(((np.asarray(k) + self.M) * cp.asnumpy(self.strides)).sum())


def rollout(spec, g, u_node, X0, steps):
    X = X0.copy()
    alive = cp.ones(len(X), dtype=bool)
    tail = cp.zeros(len(X)); exc = cp.zeros(len(X))
    t_tail = int(0.75 * steps)
    for t in range(steps):
        idx, w = g.weights(X)
        u = (w * u_node[idx]).sum(1)
        Xn = step_py(spec, [X[:, i] for i in range(g.d)], u, [False] * g.d)
        X = cp.stack(Xn, 1)
        cells = cp.abs(X / g.dvec).max(1)
        alive &= ~(cp.abs(X) > g.half).any(1)
        exc = cp.maximum(exc, cp.where(alive, cells, 0.0))
        if t >= t_tail:
            tail = cp.maximum(tail, cells)
    tail = cp.where(alive, tail, cp.inf)
    fin = tail[alive]
    return {"exit": float((~alive).mean()),
            "conv": float((fin < 1e-6).mean()) if len(fin) else None,
            "amp_med": float(cp.median(fin)) if len(fin) else None,
            "amp_max": float(fin.max()) if len(fin) else None,
            "excursion_max": float(exc[alive].max()) if len(fin) else None}


def gaps(spec, g, pi, acts, c_box, gamma, radius):
    """Smallest (best − second best) action value over grid points within `radius` cells, / c_box."""
    k = cp.stack(cp.meshgrid(*[cp.arange(-radius, radius + 1)] * g.d, indexing="ij"), -1).reshape(-1, g.d)
    X = k * g.dvec
    V = cp.asarray(pi.value_function, dtype=cp.float64)
    Qd = cp.asarray(spec["cost_w"])
    best = cp.full(len(X), -cp.inf); second = cp.full(len(X), -cp.inf)
    for a in acts:
        u = cp.full(len(X), float(a))
        Y = cp.stack(step_py(spec, [X[:, i] for i in range(g.d)], u, [False] * g.d), 1)
        idx, w = g.weights(Y)
        out = (cp.abs(Y) > g.half).any(1)
        q = c_box - (X * X * Qd).sum(1) - spec["u_w"] * u * u + gamma * cp.where(out, 0.0, (w * V[idx]).sum(1))
        second = cp.maximum(second, cp.minimum(best, q)); best = cp.maximum(best, q)
    gap = cp.asnumpy((best - second) / c_box)
    on_axis = cp.asnumpy((cp.abs(k) > 0).sum(1) == 1) & (cp.asnumpy(cp.abs(k).max(1)) == 1)
    return float(gap[on_axis].min()), float(gap.min())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["pend", "dpend"])
    p.add_argument("--R", type=float, nargs="+", default=[0.4, 0.8])
    p.add_argument("--q", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--bins", type=int, nargs="+", default=[21, 31, 41, 51, 61, 71, 81, 91, 101, 111])
    p.add_argument("--pend-bins", type=int, nargs="+", default=[21, 31, 51, 71, 101, 151, 201])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max-actions", type=int, default=601)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out" / "nonlin")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from loguru import logger
    logger.remove(); logger.add(sys.stderr, level="WARNING")

    Jd, bd = jacobians()
    lam = float(np.max(np.linalg.eigvals(Jd).real))
    systems = {"dpend": (SYSTEMS["dpend"], Jd, bd), "pend": nonlin_pend_spec(lam, 20.0)}

    for name in args.systems:
        spec, J, b = systems[name]
        d = len(spec["names"])
        A = np.eye(d) + TAU * J
        B = (TAU * b).reshape(d, 1)
        K, _ = dlqr_discounted(A, B, np.diag(spec["cost_w"]), spec["u_w"], args.gamma)
        bins_list = args.pend_bins if name == "pend" else args.bins
        for R in args.R:
            half, periodic = box(spec, R)
            assert not any(periodic)
            u_box = float(np.abs(K.ravel()) @ half)
            rng = np.random.default_rng(0)
            Y_near = rng.uniform(-3, 3, (512, d)); Y_far = rng.uniform(-8, 8, (512, d))
            for q in args.q:
                for Bn in bins_list:
                    tag = f"{name}_R{R:g}_q{q:g}_B{Bn}"
                    done = args.out_dir / f"{tag}.npz"
                    if done.exists():
                        continue
                    assert Bn % 2 == 1, "origin must be a grid point"
                    M = (Bn - 1) // 2
                    dvec = 2 * half / (Bn - 1)
                    u_cell = float(np.abs(K.ravel()) @ dvec)
                    du = q * u_cell
                    n = int(np.ceil(1.2 * u_box / du))
                    capped = 2 * n + 1 > args.max_actions
                    n = min(n, (args.max_actions - 1) // 2)
                    acts = (np.arange(-n, n + 1) * du).astype(np.float32)
                    c_box = float(np.sum(np.array(spec["cost_w"]) * half ** 2) + spec["u_w"] * (n * du) ** 2)
                    Solver = make_solver(spec, half, periodic, reward_c=c_box)
                    bins = {nm: np.linspace(-half[i], half[i], Bn, dtype=np.float32)
                            for i, nm in enumerate(spec["names"])}
                    t0 = time.time()
                    pi = Solver(bins, acts, CudaPIConfig(gamma=args.gamma, theta=1e-6,
                                                         max_eval_iter=20000, max_pi_iter=200))
                    pi.run()
                    t_train = time.time() - t0
                    g = Grid(pi, half, dvec)
                    u_node = cp.asarray(pi.action_space[pi.policy], dtype=cp.float64)
                    rec = {"system": name, "R": R, "q": q, "bins": Bn, "M": M, "delta": float(dvec[0]),
                           "n_actions": 2 * n + 1, "capped": capped, "train_s": round(t_train, 1)}
                    for nm, Y in (("near", Y_near), ("far", Y_far)):
                        if np.abs(Y).max() < M:
                            rec[nm] = rollout(spec, g, u_node, cp.asarray(Y * dvec), args.steps)
                    eye = np.eye(d, dtype=int)
                    pol = lambda kk: float(pi.action_space[pi.policy[g.flat(kk)]])
                    up = np.array([pol(eye[i]) for i in range(d)]); dn = np.array([pol(-eye[i]) for i in range(d)])
                    rec["K_hat_over_kappa"] = np.rint(up / du).astype(int).tolist()
                    rec["symmetric_axes"] = bool(np.allclose(up, -dn) and pol(np.zeros(d, int)) == 0.0)
                    rec["pi0"] = pol(np.zeros(d, int))
                    Kh = (up / dvec).reshape(1, d)
                    rec["rho_lin_Khat"] = float(max(abs(np.linalg.eigvals(A + B @ Kh))))
                    if M >= 3:
                        rec["gap_axis_min"], rec["gap_3cells_min"] = gaps(spec, g, pi, pi.action_space, c_box,
                                                                         args.gamma, 3)
                    np.savez_compressed(done, policy=pi.policy.astype(np.int16), actions=pi.action_space,
                                        dvec=dvec, K_star=K)
                    print(json.dumps(rec), flush=True)
                    with open(args.out_dir / "nonlin.jsonl", "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    del pi, g, u_node
                    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
