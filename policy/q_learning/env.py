"""Controlled offline simulator for Q-Learning. Reward (PDF section 6):
   R = task-utility - lam_fn*security-failure - lam_fp*false-positive - lam_br*unnecessary-block
       - lam_rw*unnecessary-rewrite - lam_lat*latency
Weights are chosen on the VALIDATION set (scripts/train_q_policy.py) and reported; test stays isolated."""
from __future__ import annotations
import random
from dataclasses import dataclass, asdict


@dataclass
class Lambdas:
    fn: float = 10.0     # security failure (attack passed)
    fp: float = 3.0      # benign request blocked
    br: float = 1.5      # legit task blocked instead of rewritten ("unnecessary blocking")
    rw: float = 1.0      # unnecessary rewrite of a benign request
    lat: float = 0.05    # per 100 ms of extra latency
    def as_dict(self): return asdict(self)


def reward(action: str, gt: str, rw: dict | None, lam: Lambdas, intent_floor: float) -> float:
    if action == "ALLOW":
        return 1.0 if gt == "ALLOW" else -lam.fn
    if action == "BLOCK":
        return {"BLOCK": 1.0, "REWRITE": -lam.br, "ALLOW": -lam.fp}[gt]
    # REWRITE: only counts if our validator releases it, else the request is blocked
    lat = lam.lat * (rw["ms"] / 100.0)
    if not rw["ok"]:
        return {"BLOCK": 0.8, "REWRITE": -lam.br, "ALLOW": -lam.fp}[gt] - lat
    if rw["unsafe"]:
        return -lam.fn - lat
    sim = max(0.0, min(1.0, rw["sim_after"] if rw["sim_after"] is not None else 0.5))
    if gt == "REWRITE":
        return (1.0 if sim >= intent_floor else 0.3) * (0.5 + 0.5 * sim) - lat
    if gt == "ALLOW":
        return -lam.rw + 0.3 * sim - lat
    return -0.5 * lam.fn - lat          # rewrote something that had no legitimate intent


class SessionSampler:
    """Draws short client sessions (1-5 requests); 30% of sessions are attack-heavy so decision history matters."""
    def __init__(self, recs, seed=0):
        self.rng = random.Random(seed)
        self.risky = [i for i, r in enumerate(recs) if r.gt != "ALLOW"]; self.benign = [i for i, r in enumerate(recs) if r.gt == "ALLOW"]
        self.n = len(recs)

    def session(self):
        L = self.rng.randint(1, 5); heavy = self.rng.random() < 0.3
        pool = self.risky if heavy else None
        return [self.rng.choice(pool) if (heavy and self.rng.random() < 0.7 and self.risky) else self.rng.randrange(self.n) for _ in range(L)]
