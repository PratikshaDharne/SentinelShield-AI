"""Threat-category classifier.
 backbone 'tfidf'  : word+char TF-IDF + logistic regression (CPU-light, works offline)
 backbone 'minilm' : MiniLM sentence embeddings + logistic regression (needs sentence-transformers)
Categories: benign, prompt_injection, jailbreak, prompt_leak, instruction_conflict, tool_abuse, harmful
It is trained on the NORMALIZED text (after deobfuscation), so obfuscation does not hide an attack.
Raw model confidence is never the only security signal - the policy combines it with rules + retrieval."""
from __future__ import annotations
from pathlib import Path
import joblib, numpy as np
from configs.settings import settings

CATEGORIES = ["benign", "prompt_injection", "jailbreak", "prompt_leak", "instruction_conflict", "tool_abuse", "harmful"]


from sklearn.base import BaseEstimator, TransformerMixin


class EncoderTransformer(BaseEstimator, TransformerMixin):
    """Sentence-embedding feature step. A real class (not a lambda) so the fitted pipeline can be saved with joblib."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        from embeddings.encoder import Encoder
        return Encoder.get().encode(list(X))


class ThreatClassifier:
    def __init__(self, backbone: str = "tfidf"):
        self.backbone = backbone
        self.pipe = None
        self.classes_: list[str] = []

    def _make(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline, make_union
        from sklearn.feature_extraction.text import TfidfVectorizer
        if self.backbone == "minilm":
            return make_pipeline(EncoderTransformer(), LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        feats = make_union(TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, lowercase=True),
                           TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))
        return make_pipeline(feats, LogisticRegression(max_iter=3000, C=6.0, class_weight="balanced"))

    def fit(self, texts: list[str], labels: list[str]):
        self.pipe = self._make().fit(texts, labels)
        self.classes_ = list(self.pipe.classes_)
        return self

    def proba(self, text: str) -> dict[str, float]:
        p = self.pipe.predict_proba([text])[0]
        return dict(zip(self.classes_, map(float, p)))

    def attack_prob(self, text: str) -> float:
        return 1.0 - self.proba(text).get("benign", 0.0)

    def predict(self, text: str) -> tuple[str, float]:
        pr = self.proba(text); k = max(pr, key=pr.get)
        return k, pr[k]

    def save(self, path: Path | None = None):
        path = Path(path or settings.models_dir / "threat_clf.joblib"); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"backbone": self.backbone, "pipe": self.pipe, "classes": self.classes_}, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "ThreatClassifier | None":
        path = Path(path or settings.models_dir / "threat_clf.joblib")
        if not path.exists():
            return None
        try:
            d = joblib.load(path)
            c = cls(d["backbone"]); c.pipe, c.classes_ = d["pipe"], d["classes"]
            return c
        except Exception:
            return None


def prep_row(r: dict) -> str:
    """Text the classifier/retrieval index are trained on: normalized prompt (+ last user turn for multi-turn)."""
    from api.normalizer import normalize
    t = normalize(r["text"]).normalized
    return (" ".join(r.get("history", [])[-1:]) + " || " + t).strip(" |") if r.get("history") else t
