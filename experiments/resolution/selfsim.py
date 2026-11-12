"""
experiments/resolution/selfsim.py — self-similarity of the DP policy near the
upright equilibrium, and its effective linear gain vs the LQR gain.

Claim under test (autosemejanza): near a node-equilibrium, V* ≈ quadratic and
multilinear interpolation of a quadratic satisfies I_δ[q](δz) = δ² I_1[q](z),
so in coordinates z = x/δ the node policy u(δz)/δ should be (nearly)
δ-independent, up to (i) action quantization Δu/δ and (ii) the error V_δ − V*.

Systems (base-actuated double pendulum, τ = 0.02):
    lin    — its linearisation at upright: the LQ model problem (V* exactly quadratic)
    nonlin — the full nonlinear dynamics
Both use the same exactly-quadratic stage cost r = c_box − (xᵀQx + ρu²), so the
discounted LQR gain K* is the common reference.

Per (system, n_actions, R, bins) we record:
    K_hat         least-squares gain fitted to node actions on the {-2..2}^4 stencil
    K_hat_q       same fit, but for the *ideal quantised* policy round_Δu(K* ξ)
                  → isolates action quantisation from value-function error
    rho_cl        spectral radius of A + B K_hat   (<1 ⇒ linearised closed loop stable)
    rho_cl_q      same for K_hat_q
    stab          rollout result of the interpolated policy from small perturbations
and save the stencil actions (scaled by 1/δ) for plotting.
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
from experiments.resolution.stabilization_sweep import (
    SYSTEMS, TAU, box, dpend_acc, make_solver, rollout)
from src.cuda_policy_iteration import CudaPIConfig


def jacobians():
    """Continuous-time Jacobians of the double pendulum at upright (central differences)."""
    f = lambda s, u: np.array([float(v) for v in dpend_acc([cp.float64(x) for x in s], u)])
    e = 1e-6
    J = np.zeros((4, 4))
    for i in range(4):
        d = np.zeros(4); d[i] = e
        J[:, i] = (f(d, 0.0) - f(-d, 0.0)) / (2 * e)
    b = (f(np.zeros(4), e) - f(np.zeros(4), -e)) / (2 * e)
    return J, b


def lin_spec(J, b):
    spec = dict(SYSTEMS["dpend"])
    row = lambda k: " + ".join(f"({J[k, j]:.8f}f)*{n}" for j, n in enumerate(spec["names"])) + f" + ({b[k]:.8f}f)*u"
    spec["acc_c"] = f"float a1 = {row(1)}; float a2 = {row(3)};"

    def acc_py(s, u):
        return [sum(J[k, j] * s[j] for j in range(4)) + b[k] * u for k in range(4)]
    spec["acc_py"] = acc_py
    return spec


def dlqr_discounted(A, B, Q, rho, gamma, iters=20000):
    """min Σ γ^k (xᵀQx + ρu²), u = Kx. Returns K (1×d) and P."""
    P = Q.copy()
    for _ in range(iters):
        S = rho + gamma * B.T @ P @ B
        K = -gamma * np.linalg.solve(S, B.T @ P @ A)
        Pn = Q + K.T * rho @ K + gamma * (A + B @ K).T @ P @ (A + B @ K)
        if np.max(np.abs(Pn - P)) < 1e-12:
            P = Pn; break
        P = Pn
    return K, P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--systems", nargs="+", default=["lin", "nonlin"])
    p.add_argument("--n-actions", type=int, nargs="+", default=[21, 81])
    p.add_argument("--R", type=float, nargs="+", default=[0.2, 0.4, 0.8])
    p.add_argument("--bins", type=int, nargs="+", default=[15, 21, 31, 41, 51, 61, 71])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "out" / "selfsim")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from loguru import logger
    logger.remove(); logger.add(sys.stderr, level="WARNING")

    J, b = jacobians()
    A = np.eye(4) + TAU * J
    B = (TAU * b).reshape(4, 1)
    base = SYSTEMS["dpend"]
    Q = np.diag(base["cost_w"]); rho = base["u_w"]
    K_star, P_star = dlqr_discounted(A, B, Q, rho, args.gamma)
    rho_star = max(abs(np.linalg.eigvals(A + B @ K_star)))
    print(json.dumps({"K_star": K_star.ravel().tolist(), "rho_cl_star": rho_star,
                      "eig_A": sorted(np.abs(np.linalg.eigvals(A)).tolist())}), flush=True)

    specs = {"lin": lin_spec(J, b), "nonlin": base}
    stencil = np.array(list(product(range(-2, 3), repeat=4)))
    rng = np.random.default_rng(0)
    s0 = rng.uniform(-1, 1, (256, 4)) * np.array([0.05, 0.2, 0.05, 0.2])

    for sys_name in args.systems:
        spec = specs[sys_name]
        for n_act in args.n_actions:
            actions = np.linspace(-base["u_max"], base["u_max"], n_act, dtype=np.float32)
            du = float(actions[1] - actions[0])
            for R in args.R:
                half, periodic = box(spec, R)
                c_box = float(np.sum(np.array(base["cost_w"]) * half ** 2) + rho * base["u_max"] ** 2)
                Solver = make_solver(spec, half, periodic, reward_c=c_box)
                for Bn in args.bins:
                    assert Bn % 2 == 1, "origin must be a node"
                    t0 = time.time()
                    bins = {n: np.linspace(-half[i], half[i], Bn, dtype=np.float32)
                            for i, n in enumerate(spec["names"])}
                    pi = Solver(bins, actions, CudaPIConfig(gamma=args.gamma, theta=1e-5,
                                                            max_eval_iter=10000, max_pi_iter=100))
                    pi.run()
                    t_train = time.time() - t0

                    dvec = 2 * half / (Bn - 1)
                    c = (Bn - 1) // 2
                    flat = ((stencil + c) * pi.strides).sum(1)
                    xi = stencil * dvec
                    u = pi.action_space[pi.policy[flat]].astype(np.float64)
                    K_hat = np.linalg.lstsq(xi, u, rcond=None)[0].reshape(1, 4)
                    u_q = np.clip(np.round((xi @ K_star.ravel()) / du) * du, -base["u_max"], base["u_max"])
                    K_hat_q = np.linalg.lstsq(xi, u_q, rcond=None)[0].reshape(1, 4)
                    rho_cl = max(abs(np.linalg.eigvals(A + B @ K_hat)))
                    rho_cl_q = max(abs(np.linalg.eigvals(A + B @ K_hat_q)))
                    ring1 = np.abs(stencil).max(1) == 1
                    m = rollout(spec, pi, periodic, half, "interp", s0, 1500, args.gamma)

                    tag = f"{sys_name}_a{n_act}_R{R:.2f}_B{Bn}"
                    np.savez(args.out_dir / f"{tag}.npz", stencil=stencil, u_over_delta=u / dvec[0],
                             u_q_over_delta=u_q / dvec[0], dvec=dvec, K_star=K_star, K_hat=K_hat)
                    rec = {"system": sys_name, "n_actions": n_act, "du": du, "R": R, "bins": Bn,
                           "delta_theta": float(dvec[0]), "du_over_delta": du / float(dvec[0]),
                           "K_hat": K_hat.ravel().tolist(), "K_hat_q": K_hat_q.ravel().tolist(),
                           "rel_err_K": float(np.linalg.norm(K_hat - K_star) / np.linalg.norm(K_star)),
                           "rel_err_K_q": float(np.linalg.norm(K_hat_q - K_star) / np.linalg.norm(K_star)),
                           "rho_cl": float(rho_cl), "rho_cl_q": float(rho_cl_q),
                           "frac_zero_ring1": float((u[ring1] == 0).mean()),
                           "train_s": round(t_train, 1), **m}
                    print(json.dumps(rec), flush=True)
                    with open(args.out_dir / "selfsim.jsonl", "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    del pi
                    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
