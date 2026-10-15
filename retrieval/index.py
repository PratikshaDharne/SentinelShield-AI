"""MiniLM + FAISS semantic retrieval over labelled known examples.
Retrieved content is UNTRUSTED DATA: only labels and similarity scores are used as features.
Retrieved text is never inserted into any prompt and can never override safety policy."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from embeddings.encoder import Encoder
from configs.settings import settings

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

ATTACK_LABELS = {"prompt_injection", "jailbreak", "prompt_leak", "instruction_conflict", "tool_abuse", "harmful"}


class RiskIndex:
    def __init__(self, encoder: Encoder | None = None):
        self.enc = encoder or Encoder.get()
        self.labels: list[str] = []
        self.index = None
        self.vecs: np.ndarray | None = None

    def build(self, texts: list[str], labels: list[str]):
        v = self.enc.encode(texts).astype(np.float32)
        self.vecs, self.labels = v, list(labels)
        if faiss is not None:
            self.index = faiss.IndexFlatIP(v.shape[1])
            self.index.add(v)
        return self

    def save(self, d: Path | None = None):
        d = Path(d or settings.models_dir / "retrieval"); d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vecs.npy", self.vecs)
        (d / "labels.json").write_text(json.dumps({"labels": self.labels, "encoder": self.enc.kind}))

    def load(self, d: Path | None = None) -> bool:
        d = Path(d or settings.models_dir / "retrieval")
        if not (d / "vecs.npy").exists():
            return False
        meta = json.loads((d / "labels.json").read_text())
        if meta.get("encoder") != self.enc.kind:      # index built with a different encoder -> unusable
            return False
        self.vecs = np.load(d / "vecs.npy"); self.labels = meta["labels"]
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.vecs.shape[1]); self.index.add(self.vecs)
        return True

    @property
    def ready(self) -> bool:
        return self.vecs is not None and len(self.labels) > 0

    def search(self, text: str, k: int = 8, vec=None):
        q = (np.asarray(vec, dtype=np.float32)[None, :] if vec is not None else self.enc.encode([text]).astype(np.float32))
        if self.index is not None:
            sims, idx = self.index.search(q, min(k, len(self.labels)))
            return [(self.labels[i], float(s)) for s, i in zip(sims[0], idx[0]) if i >= 0]
        sims = (self.vecs @ q[0]); idx = np.argsort(-sims)[:k]
        return [(self.labels[i], float(sims[i])) for i in idx]

    def features(self, text: str, k: int = 8, vec=None) -> dict:
        """Numeric retrieval-risk features (no retrieved text is exposed)."""
        if not self.ready:
            return {"ret_ready": 0.0, "ret_top_sim": 0.0, "ret_attack_vote": 0.0, "ret_best_attack_sim": 0.0,
                    "ret_best_benign_sim": 0.0, "ret_top_label": "unknown"}
        hits = self.search(text, k, vec)
        s0 = 0.50 if self.enc.kind == "minilm" else 0.35          # similarity floor: weak neighbours carry no vote
        n_att = sum(l in ATTACK_LABELS for l in self.labels); n_ben = max(1, len(self.labels) - n_att)
        cw = {True: len(self.labels) / (2 * max(n_att, 1)), False: len(self.labels) / (2 * n_ben)}   # class balance
        w = np.array([max(s - s0, 0.0) * cw[l in ATTACK_LABELS] for l, s in hits])
        att = np.array([1.0 if l in ATTACK_LABELS else 0.0 for l, _ in hits])
        if w.sum() <= 0:
            return {"ret_ready": 1.0, "ret_top_sim": float(hits[0][1]), "ret_attack_vote": 0.0, "ret_best_attack_sim": 0.0,
                    "ret_best_benign_sim": 0.0, "ret_top_label": hits[0][0]}
        best_att = max([s for l, s in hits if l in ATTACK_LABELS], default=0.0)
        best_ben = max([s for l, s in hits if l not in ATTACK_LABELS], default=0.0)
        return {"ret_ready": 1.0, "ret_top_sim": float(hits[0][1]), "ret_attack_vote": float((w * att).sum() / w.sum()),
                "ret_best_attack_sim": float(best_att), "ret_best_benign_sim": float(best_ben), "ret_top_label": hits[0][0]}
