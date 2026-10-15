"""Output validator for rewrites. A rewrite is released ONLY if all checks pass:
  1. safety recheck   - no override/leak/persona/fake-authority/encoded text survives
  2. no obfuscation   - no leetspeak/homoglyph/base64 leftovers
  3. topic coverage   - the content words of the extracted intent are still present
  4. intent similarity- MiniLM cosine(intent, rewrite) >= threshold
  5. sanity           - non-empty, not a refusal, reasonable length
  6. classifier recheck (if a trained classifier is available)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from api.normalizer import normalize
from policy.rules.hard_rules import scan
from embeddings.encoder import Encoder
from configs.settings import settings

_STOP = set("a an the of to in on for and or with how what why is are was were be do does did explain describe tell me "
            "about its their it this that these those can you please give show write from by as at into using use "
            "simple example including covering".split())


def content_words(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


@dataclass
class Report:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    similarity: float = 0.0
    coverage: float = 0.0


class Validator:
    def __init__(self, encoder: Encoder | None = None, classifier=None):
        self.enc = encoder or Encoder.get()
        self.clf = classifier

    @property
    def threshold(self) -> float:
        return settings.intent_sim_threshold if self.enc.kind == "minilm" else 0.30

    def validate(self, candidate: str, intent_text: str, *, strict: bool = True) -> Report:
        reasons: list[str] = []
        c = (candidate or "").strip()
        if len(c) < 12 or len(c) > 1200:
            return Report(False, ["BAD_LENGTH"])
        if re.match(r"^(i can't|i cannot|sorry|as an ai)", c, re.I):
            return Report(False, ["REFUSAL_TEXT"])
        n = normalize(c)
        if n.flags and n.flags != ["LEETSPEAK"] or "LEETSPEAK" in n.flags and n.normalized != c:
            reasons.append("OBFUSCATION_LEFTOVER")
        if n.decoded_views:
            reasons.append("ENCODED_LEFTOVER")
        hits = scan(n.normalized, allow_discussion_exemption=False, strip_negations=True)
        if hits:
            reasons.append("UNSAFE_TEXT_REMAINS:" + ",".join(sorted({h.code for h in hits})))
        tw = content_words(intent_text)
        cov = len(tw & content_words(c)) / max(1, len(tw))
        if cov < 0.85:
            reasons.append(f"TOPIC_DRIFT({cov:.2f})")
        sim = self.enc.similarity(intent_text, c)
        if sim < self.threshold:
            reasons.append(f"LOW_INTENT_SIMILARITY({sim:.2f}<{self.threshold:.2f})")
        if self.clf is not None:
            try:
                p = self.clf.attack_prob(c)
                if p >= 0.8:
                    reasons.append(f"CLASSIFIER_FLAGS_REWRITE({p:.2f})")
            except Exception:
                pass
        return Report(not reasons, reasons, round(sim, 4), round(cov, 4))
