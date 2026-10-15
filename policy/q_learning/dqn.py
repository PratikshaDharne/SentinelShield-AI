"""Deep Q-Network (pure numpy, no GPU/torch needed) for the live Allow / Rewrite / Block decision.

  * Q-network: MLP  [features] -> 64 -> 64 -> 3 Q-values, Adam optimiser, Huber loss
  * Double-DQN targets with a slowly tracking target network (soft update)
  * Experience replay buffer (uniform sampling)
  * Full-information rewards: when a label is known (owner feedback / labelled simulator stream) the reward function of the
    PDF (policy/q_learning/env.py) can be evaluated for every action, so one event teaches about all three actions
  * Safety: this module only produces Q-values. Action masking (policy.hybrid.allowed_actions), hard rules, exploration limits
    and the promotion gate live in policy/q_learning/live.py - the network can never bypass them.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from policy.hybrid import ACTIONS, allowed_actions, risk_score

N_ACT = 3


def features(p, rule_sev: float = 0.0) -> np.ndarray:
    """State vector: supervised-policy evidence vector + decision history + baseline recommendation + risk + legal-action mask."""
    v = np.asarray(p.vec if p.vec is not None else np.zeros(22), dtype=np.float32).copy()
    # scale the unbounded columns to ~[0,1]: 18 log_len, 19 rewritability(0-2), 20 n_rule_codes, 21 n_model_flags
    if v.shape[0] >= 22:
        v[18] = v[18] / 8.0; v[19] = v[19] / 2.0; v[20] = v[20] / 4.0; v[21] = v[21] / 3.0
    sup = np.zeros(3, dtype=np.float32); sup[int(p.sup_action) % 3] = 1.0
    legal = np.zeros(3, dtype=np.float32)
    for a in allowed_actions(p): legal[ACTIONS.index(a)] = 1.0
    risk = np.float32(risk_score(p, rule_sev))
    return np.concatenate([v, [min(p.prev_blocks, 1), risk], sup, legal]).astype(np.float32)


def legal_mask(p) -> np.ndarray:
    m = np.zeros(3, dtype=bool)
    for a in allowed_actions(p): m[ACTIONS.index(a)] = True
    return m


class QNet:
    def __init__(self, d_in: int, hidden=(64, 64), seed: int = 0):
        rng = np.random.default_rng(seed); dims = [d_in, *hidden, N_ACT]
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in dims[1:]]
        self.W[-1] *= 0.1
        self.m = [np.zeros_like(x) for x in self.W + self.b]; self.v = [np.zeros_like(x) for x in self.W + self.b]; self.t = 0

    def forward(self, X: np.ndarray, keep: bool = False):
        acts = [X]; h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1: h = np.maximum(h, 0.0)
            acts.append(h)
        return (h, acts) if keep else h

    def step(self, X, T, M, lr=1e-3, delta=1.0, clip=5.0) -> float:
        """One Adam step on masked Huber loss between Q(X) and targets T (only entries where M==1 count)."""
        Q, acts = self.forward(X, keep=True)
        diff = (Q - T) * M; n = max(float(M.sum()), 1.0)
        g = np.clip(diff, -delta, delta) / n                                   # d(Huber)/dQ
        loss = float((np.where(np.abs(diff) <= delta, 0.5 * diff ** 2, delta * (np.abs(diff) - 0.5 * delta))).sum() / n)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = g
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ d; gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        params, grads = self.W + self.b, gW + gb
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
        for k, (p_, g_) in enumerate(zip(params, grads)):
            nrm = np.linalg.norm(g_)
            if nrm > clip: g_ = g_ * (clip / nrm)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g_; self.v[k] = b2 * self.v[k] + (1 - b2) * g_ * g_
            p_ -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)
        return loss

    def copy_from(self, o: "QNet", tau: float = 1.0):
        for a, b in zip(self.W + self.b, o.W + o.b): a *= (1 - tau); a += tau * b

    def state(self) -> dict:
        return {**{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)}}

    def load_state(self, z):
        for i in range(len(self.W)): self.W[i] = z[f"W{i}"].astype(np.float32); self.b[i] = z[f"b{i}"].astype(np.float32)
        self.m = [np.zeros_like(x) for x in self.W + self.b]; self.v = [np.zeros_like(x) for x in self.W + self.b]; self.t = 0


class DQNAgent:
    def __init__(self, d_in: int, *, gamma: float = 0.3, lr: float = 1e-3, batch: int = 64, capacity: int = 50000, tau: float = 0.02,
                 seed: int = 0, hidden=(64, 64)):
        self.d_in, self.gamma, self.lr, self.batch, self.tau, self.capacity = d_in, gamma, lr, batch, tau, capacity
        self.net, self.target = QNet(d_in, hidden, seed), QNet(d_in, hidden, seed); self.target.copy_from(self.net)
        self.rng = np.random.default_rng(seed + 1)
        self.X = np.zeros((capacity, d_in), np.float32); self.R = np.zeros((capacity, N_ACT), np.float32)
        self.M = np.zeros((capacity, N_ACT), np.float32); self.X2 = np.zeros((capacity, d_in), np.float32)
        self.L2 = np.zeros((capacity, N_ACT), np.float32)          # legal mask of next state (all-zero = terminal)
        self.n = 0; self.ptr = 0; self.updates = 0; self.loss_ema = None; self.seen = 0

    # ------------------------------------------------------------------ acting
    def q(self, x: np.ndarray) -> np.ndarray:
        return self.net.forward(x[None, :])[0]

    # ------------------------------------------------------------------ replay
    def add(self, x, R, M, x2=None, legal2=None):
        i = self.ptr
        self.X[i] = x; self.R[i] = R; self.M[i] = M
        if x2 is None: self.X2[i] = 0.0; self.L2[i] = 0.0
        else: self.X2[i] = x2; self.L2[i] = legal2
        self.ptr = (self.ptr + 1) % self.capacity; self.n = min(self.n + 1, self.capacity); self.seen += 1

    def train(self, steps: int = 1) -> float | None:
        if self.n < max(self.batch, 32): return None
        tot = 0.0
        for _ in range(steps):
            idx = self.rng.integers(0, self.n, size=min(self.batch, self.n))
            X, R, M, X2, L2 = self.X[idx], self.R[idx], self.M[idx], self.X2[idx], self.L2[idx]
            T = R.copy()
            has_next = L2.sum(1) > 0
            if self.gamma > 0 and has_next.any():
                qo = self.net.forward(X2[has_next]); qo = np.where(L2[has_next] > 0, qo, -1e9)
                a2 = qo.argmax(1); qt = self.target.forward(X2[has_next])
                T[has_next] = R[has_next] + self.gamma * qt[np.arange(a2.size), a2][:, None]
            loss = self.net.step(X, T, M, lr=self.lr)
            self.target.copy_from(self.net, self.tau); self.updates += 1; tot += loss
        loss = tot / steps
        self.loss_ema = loss if self.loss_ema is None else 0.98 * self.loss_ema + 0.02 * loss
        return loss

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path, meta: dict | None = None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, d_in=self.d_in, gamma=self.gamma, updates=self.updates, **self.net.state())
        if meta is not None: path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path, **kw) -> "DQNAgent | None":
        path = Path(path)
        if not path.exists(): return None
        z = np.load(path)
        a = cls(int(z["d_in"]), gamma=float(z["gamma"]), **kw); a.net.load_state(z); a.target.copy_from(a.net); a.updates = int(z["updates"])
        return a

    def reinit(self, seed: int = 0):
        self.net = QNet(self.d_in, seed=seed); self.target = QNet(self.d_in, seed=seed); self.target.copy_from(self.net)
        self.n = self.ptr = self.updates = self.seen = 0; self.loss_ema = None
