"""Compact discrete state for the Q-Learning policy (PDF section 7): risk bin x evidence agreement x retrieval
x task present x context risk x semantic cluster x rewritability x recent-block history x supervised-recommendation (3) = 31104 states."""
from __future__ import annotations
import numpy as np

N_CLUSTERS = 4
SHAPE = (9, 4, 3, 2, 2, N_CLUSTERS, 3, 2, 3)
N_STATES = int(np.prod(SHAPE))
_EDGES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90]


def make_state(p, risk: float, prev_blocks: int | None = None) -> int:
    rb = int(np.searchsorted(_EDGES, risk, side="right"))
    rule_att = bool(p.has_attack_clause or p.rule_codes)
    model_att = p.clf_attack_prob >= 0.5
    agree = (1 if rule_att else 0) + (2 if model_att else 0)          # 0 neither,1 rules only,2 model only,3 both
    v = p.ret_attack_vote; rt = 0 if v < 0.2 else 1 if v < 0.6 else 2
    pb = int(min(p.prev_blocks if prev_blocks is None else prev_blocks, 1))
    dims = (rb, agree, rt, int(p.has_task), int(p.context_risk), int(p.cluster) % N_CLUSTERS, int(p.rewritability), pb, int(p.sup_action) % 3)
    return int(np.ravel_multi_index(dims, SHAPE))
