"""Our Rewrite Agent. No external LLM is involved.

 deobfuscate -> tag clauses (attack / task / context) -> extract intent -> look up scope in
 our topic KB -> generate candidates (fine-tuned Flan-T5 if available, then template composer)
 -> validate each (safety recheck + topic coverage + MiniLM intent similarity)
 -> retry with the next candidate / variant -> else BLOCK.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from api.normalizer import normalize, NormResult
from rewriting.span_tagger import tag, Tagged
from rewriting.topics import TopicKB
from rewriting.generator import TemplateRewriter, T5Rewriter, normalize_task
from validation.validator import Validator
from embeddings.encoder import Encoder
from configs.settings import settings


_STRONG = re.compile(r"(?:^|,\s*)(?:explain|describe|write|summari[sz]e|list|teach|compare|define|outline|translate|solve|calculate|draft|"
                     r"give me|help me|what (?:is|are)|how (?:does|do|can|to)|why (?:is|are|do|does)|tell me (?:about|how|what))\b", re.I)


@dataclass
class RewriteResult:
    status: str                        # REWRITE | BLOCK
    rewritten_prompt: str | None = None
    intent: str = ""
    context: str = ""
    removed_spans: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    similarity: float = 0.0
    attempts: int = 0
    generator: str = ""
    validation_failures: list[list[str]] = field(default_factory=list)


class RewriteAgent:
    def __init__(self, encoder: Encoder | None = None, validator: Validator | None = None,
                 kb: TopicKB | None = None, classifier=None):
        self.enc = encoder or Encoder.get()
        self.validator = validator or Validator(self.enc, classifier)
        self.kb = kb or TopicKB()
        self.template = TemplateRewriter()
        self.t5 = T5Rewriter() if settings.rewriter_backend in ("auto", "t5") else T5Rewriter.__new__(T5Rewriter)
        if not hasattr(self.t5, "ok"):
            self.t5.ok = False

    @staticmethod
    def _constraints(tg: Tagged, task: str) -> list[str]:
        codes = {h.code for h in tg.attack_hits}
        cons = []
        if "PROMPT_LEAK" in codes and re.search(r"\b(assistants?|chatbots?|llms?|language models?|instruction hierarchy|system prompts?)\b", task, re.I):
            cons.append("leak")
        if codes & {"PERSONA_JAILBREAK"}:
            cons.append("safety")
        return cons

    def rewrite(self, norm: NormResult, tagged: Tagged | None = None, max_tries: int | None = None) -> RewriteResult:
        tg = tagged or tag(norm)
        removed = [h.span for h in tg.attack_hits]
        conservative = not tg.attack_hits
        if conservative:
            # Suspicious but no attack clause identified: keep ONLY clauses that are clearly tasks (our safest intent extraction)
            keep = [cl for cl in tg.clauses if cl.label == "TASK" and _STRONG.search(cl.text)]
            dropped = [cl.text for cl in tg.clauses if cl.label == "TASK" and not _STRONG.search(cl.text)]
            removed = removed + dropped
            tg = Tagged(tg.clauses, tg.attack_hits, "; and ".join(c.text.rstrip(" .!") for c in keep), tg.context_text)
        if not tg.has_task:
            return RewriteResult("BLOCK", reason_codes=["NO_LEGIT_INTENT"], removed_spans=removed)
        task = tg.task_text
        example_asked = bool(re.search(r"\s+with (?:a|an) (?:simple|short|basic|small|quick) example\b", task, re.I))
        task = re.sub(r"\s+with (?:a|an) (?:simple|short|basic|small|quick) example\b", "", task, flags=re.I)
        intent = normalize_task(task)
        topic, scope = self.kb.lookup(task)
        scope = scope or self.kb.generic_scope(task) or None
        if example_asked and scope and "example" not in scope:
            scope = scope.replace(", and ", ", ") + ", and a simple illustrative example"
        elif example_asked and not scope:
            scope = "a simple illustrative example"
        cons = self._constraints(tg, task)
        tries = max_tries or settings.rewrite_max_tries
        gen_t5 = self.t5.candidates(task, tg.context_text, scope, cons, n=tries) if self.t5.ok else []
        gen_tm = self.template.candidates(task, tg.context_text, scope, cons, n=3)
        queue = [("t5", c) for c in gen_t5[:1]] + [("template", gen_tm[0])] + \
                [("t5", c) for c in gen_t5[1:2]] + [("template", gen_tm[1])] + \
                [("t5", c) for c in gen_t5[2:]] + [("template", gen_tm[2])]
        failures, attempts = [], 0
        for name, cand in queue[: tries + 2]:
            attempts += 1
            rep = self.validator.validate(cand, intent)
            if rep.passed:
                conf = round(min(1.0, 0.5 * rep.similarity + 0.5 * rep.coverage), 3)
                return RewriteResult("REWRITE", cand, intent, tg.context_text, removed, ["REWRITE_VALIDATED"] + (["CONSERVATIVE_INTENT_ONLY"] if conservative else []),
                                     conf, rep.similarity, attempts, name, failures)
            failures.append(rep.reasons)
        return RewriteResult("BLOCK", None, intent, tg.context_text, removed,
                             ["REWRITE_VALIDATION_FAILED"], 0.0, 0.0, attempts, "", failures)
