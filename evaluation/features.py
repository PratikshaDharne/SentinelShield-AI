"""Feature records for evaluation and policy training.
For the TRAIN split, classifier + retrieval features are OUT-OF-FOLD (5-fold, folds keyed by text hash so
duplicates share a fold) so learned policies never see over-confident in-sample model outputs."""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field
from classifiers.threat_classifier import ThreatClassifier, prep_row
from retrieval.index import RiskIndex
from policy.rules.hard_rules import scan
from configs.settings import settings

DATA = settings.datasets_dir / "data"


@dataclass
class Rec:
    row: dict
    an: object                     # api.pipeline.Analysis
    _rw: dict | None = field(default=None, repr=False)

    @property
    def gt(self) -> str: return self.row["decision"]
    @property
    def category(self) -> str: return self.row["category"]


def load_rows(split: str) -> list[dict]:
    f = DATA / f"{split}.jsonl"
    return [json.loads(l) for l in f.open(encoding="utf-8")] if f.exists() else []


def fold_of(text: str, k: int = 5) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % k


def analyze_rows(rows: list[dict], g, *, clf=None, index=None) -> list[Rec]:
    return [Rec(r, g.analyze(r["text"], r.get("history", [])[-6:], clf=clf, index=index, prev_blocks=0)) for r in rows]


def oof_analyze(rows: list[dict], g, k: int = 5) -> list[Rec]:
    """Analyse TRAIN rows with models that never saw them."""
    backbone = g.clf.backbone if g.clf else "tfidf"
    texts = [prep_row(r) for r in rows]; labels = [r["category"] for r in rows]; folds = [fold_of(r.get("family") or r["text"], k) for r in rows]    # GROUPED by attack phrase / topic -> mimics unseen phrasing
    out: list[Rec | None] = [None] * len(rows)
    for f in range(k):
        tr = [i for i in range(len(rows)) if folds[i] != f]; te = [i for i in range(len(rows)) if folds[i] == f]
        clf = ThreatClassifier(backbone).fit([texts[i] for i in tr], [labels[i] for i in tr])
        idx = RiskIndex(g.enc).build([texts[i] for i in tr], [labels[i] for i in tr])
        for i in te:
            out[i] = Rec(rows[i], g.analyze(rows[i]["text"], rows[i].get("history", [])[-6:], clf=clf, index=idx, prev_blocks=0))
    return out  # type: ignore


def leaks_attack(rewrite: str, attack_text: str | None, frac: float = 0.6) -> bool:
    """Ground-truth leakage check: does the rewrite still carry (most of) the known attack fragment?"""
    import re
    if not attack_text:
        return False
    toks = {w for w in re.findall(r"[a-z0-9]+", attack_text.lower()) if len(w) > 2}
    have = set(re.findall(r"[a-z0-9]+", rewrite.lower()))
    return bool(toks) and len(toks & have) / len(toks) >= frac


def rewrite_outcome(g, rec: Rec) -> dict:
    """What would our rewrite agent do with this prompt? (cached per record)"""
    if rec._rw is not None:
        return rec._rw
    t = time.perf_counter()
    r = g.rewriter.rewrite(rec.an.norm, rec.an.tagged)
    ms = (time.perf_counter() - t) * 1000
    intent = rec.row.get("intent")
    ok = r.status == "REWRITE"
    unsafe = bool(ok and (scan(r.rewritten_prompt, allow_discussion_exemption=False, strip_negations=True)
                          or leaks_attack(r.rewritten_prompt, rec.row.get("attack_text"))))
    sim_after = g.enc.similarity(r.rewritten_prompt, intent) if (ok and intent) else None
    sim_before = g.enc.similarity(rec.row["text"], intent) if intent else None
    rec._rw = {"ok": ok, "unsafe": unsafe, "text": r.rewritten_prompt, "ms": round(ms, 2), "sim_after": sim_after,
               "sim_before": sim_before, "confidence": r.confidence, "failures": r.validation_failures}
    return rec._rw


def oof_supervised_actions(recs: list[Rec], k: int = 5) -> list[int]:
    """Cross-fitted supervised-policy recommendations for TRAIN records (grouped folds), so Q-Learning can learn when to trust it."""
    from policy.supervised.model import SupervisedPolicy
    from policy.hybrid import ACTIONS
    folds = [fold_of(r.row.get("family") or r.row["text"], k) for r in recs]; out = [0] * len(recs)
    for f in range(k):
        tr = [r for r, g in zip(recs, folds) if g != f]; sp = SupervisedPolicy().fit(tr)
        for i, (r, g) in enumerate(zip(recs, folds)):
            if g == f: out[i] = ACTIONS.index(sp.act_vec(__import__("policy.supervised.model", fromlist=["vector"]).vector(r.an)))
    return out
