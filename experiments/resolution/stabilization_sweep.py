"""
experiments/resolution/stabilization_sweep.py — does the DP greedy policy
stabilise the upright equilibrium, as a function of grid spacing δ?

Systems (same physics as the repo runners, τ = 0.02, λ ≈ 8.18 s⁻¹ for both):
    dpend  — base-actuated double pendulum, 4D  (th1, w1, th2, w2)
    dcp    — double inverted pendulum on a cart, 6D (x, xd, th1, w1, th2, w2)

To isolate the grid from reward shaping, every run uses the same smooth
stage reward r = exp(-cost), cost quadratic in the state (fixed physical
scales, independent of the grid box) plus a small control term. Leaving the
box through a non-periodic dimension terminates (future reward 0).

The grid box has angle half-width R (and angular-velocity half-width
R·15/π); R = π means the angles are periodic, i.e. the global swing-up grid.
For each (R, bins) we solve PI and roll out the same Euler map from small
perturbations around upright, so every failure is a discretisation failure.
Policy extractions: interp (barycentric blend of grid actions, what the repo
uses) and greedy (one-step lookahead on the interpolated value function).
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
from src.cuda_policy_iteration import (CudaPIConfig, CudaPolicyIteration4D,
                                       CudaPolicyIteration6D)

TAU = 0.02
G = 9.8
W_PER_RAD = 15.0 / np.pi

# Accelerations as shared C / array code. `c` is the math namespace (cosf vs cp.cos).
DPEND_C = r'''
    float m12 = 0.2f, l = 0.5f, m2 = 0.1f;
    float c12 = cosf(th1 - th2), s12 = sinf(th1 - th2);
    float M11 = m12*l*l, M12 = m2*l*l*c12, M22 = m2*l*l;
    float b1 = u + m12*9.8f*l*sinf(th1) - m2*l*l*s12*w2*w2;
    float b2 = m2*9.8f*l*sinf(th2) + m2*l*l*s12*w1*w1;
    float det = M11*M22 - M12*M12;
    float a1 = (M22*b1 - M12*b2)/det, a2 = (M11*b2 - M12*b1)/det;
'''

DCP_C = r'''
    float M = 1.0f, m12 = 0.2f, m2 = 0.1f, l = 0.5f;
    float c1 = cosf(th1), s1 = sinf(th1), c2 = cosf(th2), s2 = sinf(th2);
    float c12 = cosf(th1 - th2), s12 = sinf(th1 - th2);
    float a = M + m12, b = m12*l*c1, c = m2*l*c2, d = m12*l*l, e = m2*l*l*c12, f = m2*l*l;
    float r1 = u + m12*l*w1*w1*s1 + m2*l*w2*w2*s2;
    float r2 = m12*9.8f*l*s1 - m2*l*l*w2*w2*s12;
    float r3 = m2*9.8f*l*s2 + m2*l*l*w1*w1*s12;
    float k11 = d*f - e*e, k12 = e*c - b*f, k13 = b*e - d*c;
    float k22 = a*f - c*c, k23 = b*c - a*e, k33 = a*d - b*b;
    float det = a*k11 + b*k12 + c*k13;
    float ax = (k11*r1 + k12*r2 + k13*r3)/det;
    float a1 = (k12*r1 + k22*r2 + k23*r3)/det;
    float a2 = (k13*r1 + k23*r2 + k33*r3)/det;
'''


def dpend_acc(s, u):
    th1, w1, th2, w2 = s
    m12, l, m2 = 0.2, 0.5, 0.1
    c12, s12 = cp.cos(th1 - th2), cp.sin(th1 - th2)
    M11, M12, M22 = m12 * l * l, m2 * l * l * c12, m2 * l * l
    b1 = u + m12 * G * l * cp.sin(th1) - m2 * l * l * s12 * w2 * w2
    b2 = m2 * G * l * cp.sin(th2) + m2 * l * l * s12 * w1 * w1
    det = M11 * M22 - M12 * M12
    return [w1, (M22 * b1 - M12 * b2) / det, w2, (M11 * b2 - M12 * b1) / det]


def dcp_acc(s, u):
    x, xd, th1, w1, th2, w2 = s
    M, m12, m2, l = 1.0, 0.2, 0.1, 0.5
    c1, s1, c2, s2 = cp.cos(th1), cp.sin(th1), cp.cos(th2), cp.sin(th2)
    c12, s12 = cp.cos(th1 - th2), cp.sin(th1 - th2)
    a, b, c, d, e, f = M + m12, m12 * l * c1, m2 * l * c2, m12 * l * l, m2 * l * l * c12, m2 * l * l
    r1 = u + m12 * l * w1 * w1 * s1 + m2 * l * w2 * w2 * s2
    r2 = m12 * G * l * s1 - m2 * l * l * w2 * w2 * s12
    r3 = m2 * G * l * s2 + m2 * l * l * w1 * w1 * s12
    k11, k12, k13 = d * f - e * e, e * c - b * f, b * e - d * c
    k22, k23, k33 = a * f - c * c, b * c - a * e, a * d - b * b
    det = a * k11 + b * k12 + c * k13
    return [xd, (k11 * r1 + k12 * r2 + k13 * r3) / det,
            w1, (k12 * r1 + k22 * r2 + k23 * r3) / det,
            w2, (k13 * r1 + k23 * r2 + k33 * r3) / det]


# Per-system spec. cost_w are weights on (state/scale)^2 with fixed physical scales.
SYSTEMS = {
    "dpend": dict(
        names=["th1", "w1", "th2", "w2"], angle_dims=[0, 2], vel_dims=[1, 3], fixed={},
        acc_c=DPEND_C, deriv=["w1", "a1", "w2", "a2"], acc_py=dpend_acc,
        cost_w=[1.0, 0.1, 1.0, 0.1], u_w=0.01, u_max=3.0, base=CudaPolicyIteration4D),
    "dcp": dict(
        names=["x", "xd", "th1", "w1", "th2", "w2"], angle_dims=[2, 4], vel_dims=[3, 5],
        fixed={0: 2.4, 1: 5.0},
        acc_c=DCP_C, deriv=["xd", "ax", "w1", "a1", "w2", "a2"], acc_py=dcp_acc,
        cost_w=[0.1, 0.01, 1.0, 0.1, 1.0, 0.1], u_w=1e-4, u_max=20.0, base=CudaPolicyIteration6D),
}


def box(spec, R):
    d = len(spec["names"])
    half = np.zeros(d)
    for i in spec["angle_dims"]:
        half[i] = min(R, np.pi)
    for i in spec["vel_dims"]:
        half[i] = min(R, np.pi) * W_PER_RAD
    for i, h in spec["fixed"].items():
        half[i] = h
    periodic = [i in spec["angle_dims"] and R >= np.pi - 1e-3 for i in range(d)]
    return half, periodic


def make_solver(spec, half, periodic, reward_c=None):
    """reward_c=None → r = exp(-cost); otherwise r = reward_c - cost (exactly quadratic)."""
    names, d = spec["names"], len(spec["names"])
    args_in = ", ".join(f"float {n}" for n in names)
    args_out = ", ".join(f"float* n_{n}" for n in names)
    lines = []
    for i, n in enumerate(names):
        nxt = f"{n} + {TAU}f * {spec['deriv'][i]}"
        if periodic[i]:
            nxt = f"wrapf({nxt})"
        lines.append(f"*n_{n} = {nxt};")
    cost = " + ".join(f"{spec['cost_w'][i]}f * wrapf({n}) * wrapf({n})" if i in spec["angle_dims"]
                      else f"{spec['cost_w'][i]}f * {n} * {n}" for i, n in enumerate(names))
    outside = " || ".join(f"fabsf(*n_{n}) > {half[i]:.6f}f"
                          for i, n in enumerate(names) if not periodic[i])
    stage = f"({cost} + {spec['u_w']}f * u * u)"
    reward_expr = f"expf(-{stage})" if reward_c is None else f"{reward_c:.6f}f - {stage}"
    src = rf'''
    __device__ float wrapf(float x) {{
        float r = fmodf(x + 3.14159265358979f, 6.28318530717959f);
        if (r < 0.0f) r += 6.28318530717959f;
        return r - 3.14159265358979f;
    }}
    __device__ void step_dynamics({args_in}, float u, {args_out},
                                  float* reward, bool* terminated) {{
        {spec["acc_c"]}
        {" ".join(lines)}
        *reward = {reward_expr};
        *terminated = {outside if outside else "false"};
    }}
    '''

    class Solver(spec["base"]):
        def _dynamics_cuda_src(self):
            return src

        def _terminal_fn(self, states):
            return np.zeros(len(states), dtype=bool), 0.0

    return Solver


# ── GPU rollouts (float64) ────────────────────────────────────────────────────

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def step_py(spec, s, u, periodic):
    der = spec["acc_py"](s, u)
    out = [s[i] + TAU * der[i] for i in range(len(s))]
    return [wrap(v) if periodic[i] else v for i, v in enumerate(out)]


def interp(s, lo, hi, shape, strides, corners):
    """corners: (C, d) int array; returns flat indices and weights, both (n, C)."""
    S = cp.stack(s, 1)
    n = cp.clip((S - lo) / (hi - lo) * (shape - 1), 0, shape - 1)
    base = cp.minimum(n.astype(cp.int64), shape - 2)
    frac = n - base
    idx = ((base[:, None, :] + corners[None]) * strides).sum(-1)
    wgt = cp.where(corners[None].astype(bool), frac[:, None, :], 1 - frac[:, None, :]).prod(-1)
    return idx, wgt


def rollout(spec, pi, periodic, half, mode, s0, n_steps, gamma):
    d = len(spec["names"])
    lo, hi = cp.asarray(pi.bounds_low, cp.float64), cp.asarray(pi.bounds_high, cp.float64)
    shape, strides = cp.asarray(pi.grid_shape, cp.int64), cp.asarray(pi.strides, cp.int64)
    corners = cp.asarray(list(product([0, 1], repeat=d)), dtype=cp.int64)
    acts = cp.asarray(pi.action_space, dtype=cp.float64)
    pol = cp.asarray(pi.policy)
    V = cp.asarray(pi.value_function, dtype=cp.float64)
    s = [cp.asarray(s0[:, k]) for k in range(d)]
    n = len(s0)
    ang_max = cp.zeros(n); tail_amp = cp.zeros(n)
    alive = cp.ones(n, dtype=bool)
    tail_start = int(0.75 * n_steps)
    for t in range(n_steps):
        if mode == "interp":
            idx, wg = interp(s, lo, hi, shape, strides, corners)
            u = (wg * acts[pol[idx]]).sum(1)
        else:
            # All actions at once: rows are (action, trajectory) pairs.
            A = len(acts)
            s_rep = [cp.tile(v, A) for v in s]
            ua = cp.repeat(acts, n)
            ns = step_py(spec, s_rep, ua, periodic)
            idx, wg = interp(ns, lo, hi, shape, strides, corners)
            cost = sum(spec["cost_w"][k] * (wrap(s_rep[k]) if k in spec["angle_dims"] else s_rep[k]) ** 2
                       for k in range(d)) + spec["u_w"] * ua * ua
            out = cp.zeros(A * n, dtype=bool)
            for k in range(d):
                if not periodic[k]:
                    out |= cp.abs(ns[k]) > half[k]
            q = cp.exp(-cost) + gamma * cp.where(out, 0.0, (wg * V[idx]).sum(1))
            u = acts[cp.argmax(q.reshape(A, n), 0)]
        s = step_py(spec, s, u, periodic)
        amp = cp.maximum(*[cp.abs(wrap(s[k])) for k in spec["angle_dims"]])
        ang_max = cp.maximum(ang_max, amp)
        for k in range(d):
            if not periodic[k]:
                alive &= cp.abs(s[k]) <= half[k]
        if t >= tail_start:
            tail_amp = cp.maximum(tail_amp, amp)
    tail_amp = cp.where(alive, tail_amp, cp.inf)
    return {
        "frac_stab_0.02": float((tail_amp < 0.02).mean()),
        "frac_stab_0.1": float((tail_amp < 0.1).mean()),
        "frac_fell": float(((ang_max > 0.5) | ~alive).mean()),
        "median_tail_amp": float(cp.median(tail_amp)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=list(SYSTEMS), required=True)
    p.add_argument("--R", type=float, nargs="+", default=[0.2, 0.4, 0.8, 1.6, np.pi])
    p.add_argument("--bins", type=int, nargs="+", required=True)
    p.add_argument("--n-actions", type=int, default=21)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-init", type=int, default=256)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--modes", nargs="+", default=["interp", "greedy"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from loguru import logger
    logger.remove(); logger.add(sys.stderr, level="WARNING")

    spec = SYSTEMS[args.system]
    d = len(spec["names"])
    rng = np.random.default_rng(0)
    # Small perturbations around upright: angles ±0.05 rad, angular rates ±0.2 rad/s, cart ±0.1 m.
    scale = np.zeros(d)
    scale[spec["angle_dims"]] = 0.05
    scale[spec["vel_dims"]] = 0.2
    if args.system == "dcp":
        scale[0] = 0.1
    s0 = rng.uniform(-1, 1, (args.n_init, d)) * scale
    actions = np.linspace(-spec["u_max"], spec["u_max"], args.n_actions, dtype=np.float32)

    for R in args.R:
        half, periodic = box(spec, R)
        Solver = make_solver(spec, half, periodic)
        for B in args.bins:
            t0 = time.time()
            bins = {n: np.linspace(-half[i], half[i], B, dtype=np.float32)
                    for i, n in enumerate(spec["names"])}
            cfg = CudaPIConfig(gamma=args.gamma, theta=1e-4, max_eval_iter=5000, max_pi_iter=60)
            pi = Solver(bins, actions, cfg)
            pi.run()
            t_train = time.time() - t0
            for mode in args.modes:
                m = rollout(spec, pi, periodic, half, mode, s0, args.steps, args.gamma)
                rec = {"system": args.system, "R": R, "bins": B,
                       "delta_theta": 2 * min(R, np.pi) / (B - 1),
                       "n_states": int(B) ** d, "mode": mode, "train_s": round(t_train, 1), **m}
                print(json.dumps(rec), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            del pi
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
