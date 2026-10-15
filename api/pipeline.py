"""Guardrail pipeline: Request -> Normalization -> Encoder -> Retrieval -> Safety models -> Policy
-> (Rewrite agent) -> Output validation. All decisions here are made by OUR code/models; no external
LLM is contacted. Gemini/Groq are called later (api/llm.py) ONLY with the final approved prompt."""
from __future__ import annotations
import hashlib, re, time
from dataclasses import dataclass, field, asdict
from api.normalizer import normalize
from rewriting.span_tagger import tag, Clause
from rewriting.agent import RewriteAgent, RewriteResult
from policy.rules.hard_rules import scan, scan_context, hard_block, tool_sensitivity, Hit
from policy.hybrid import HybridPolicy, PolicyInput
from classifiers.threat_classifier import ThreatClassifier
from retrieval.index import RiskIndex
from embeddings.encoder import Encoder
from validation.validator import Validator
from configs.settings import settings

_SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|AIza[0-9A-Za-z_\-]{30,}|gsk_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|"
                     r"AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PASS = re.compile(r"(?i)(password|passwd|pwd|secret|token|api[_ -]?key)\s*[:=]\s*\S+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s-]?){10,14}(?!\d)")


def redact(text: str) -> str:
    t = _SECRET.sub("[REDACTED_KEY]", text or "")
    t = _PASS.sub(lambda m: m.group(1) + "=[REDACTED]", t)
    t = _EMAIL.sub("[REDACTED_EMAIL]", t)
    return _PHONE.sub("[REDACTED_PHONE]", t)


@dataclass
class GuardResult:
    decision: str
    reason_codes: list[str]
    category: str
    risk_score: float
    confidence: float
    rewritten_prompt: str | None
    flags: list[str]
    needs_review: bool
    policy_version: str
    latency_ms: float
    stage_latency_ms: dict
    prompt_hash: str
    prompt_len: int
    rewrite: dict | None = None
    final_prompt: str | None = None          # prompt that may be sent to the answer LLM (None if BLOCK)
    models: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self); d.pop("final_prompt", None); return d


def _sup_vector(an):
    from policy.supervised.model import vector
    return vector(an)


@dataclass
class Analysis:
    """Everything the policy needs, computed once (live API, training and every experiment share this)."""
    prompt: str
    norm: object
    tagged: object
    rule_codes: list
    rule_sev: float
    hard_block: bool
    ctx_risk: bool
    probs: dict
    attack_p: float
    top_cat: str
    ret: dict
    rewritability: int
    cluster: int
    prev_blocks: int
    category: str
    stage_ms: dict
    history: list

    def policy_input(self) -> PolicyInput:
        return PolicyInput(self.rule_codes, self.hard_block, bool(self.tagged.attack_hits), self.tagged.has_task, self.attack_p,
                           self.top_cat, self.ret.get("ret_attack_vote", 0.0), self.ret.get("ret_top_sim", 0.0), self.ctx_risk,
                           bool(self.norm.flags), "none", 0.0, self.rewritability, self.cluster, self.prev_blocks, _sup_vector(self))


class Guardrail:
    _inst = None

    def __init__(self):
        self.enc = Encoder.get()
        self.clf = ThreatClassifier.load()
        self.index = RiskIndex(self.enc); self.index_ready = self.index.load()
        self.validator = Validator(self.enc, self.clf)
        self.rewriter = RewriteAgent(self.enc, self.validator, classifier=self.clf)
        self.centroids = self._load_centroids()
        self.policy = HybridPolicy(q_policy=self._load_q(), sup=self._load_sup())
        self.rl = self._load_rl()
        self._recent: dict = {}

    @staticmethod
    def _load_q():
        try:
            from policy.q_learning.agent import QPolicy
            q = QPolicy.load()
            if q is not None and not q.meta.get("recommended", True) and not settings.force_q_policy:
                return None            # trained but did not beat the safe baseline on validation -> not used live
            return q
        except Exception:
            return None

    @staticmethod
    def _load_rl():
        try:
            from policy.q_learning.live import RLManager
            return RLManager.load()
        except Exception:
            return None

    @staticmethod
    def _load_sup():
        try:
            from policy.supervised.model import SupervisedPolicy
            return SupervisedPolicy.load()
        except Exception:
            return None

    def _load_centroids(self):
        try:
            import numpy as np
            f = settings.models_dir / "q_policy" / "centroids.npy"
            return np.load(f) if f.exists() else None
        except Exception:
            return None

    @classmethod
    def get(cls) -> "Guardrail":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def reload_policy(self):
        self.centroids = self._load_centroids(); self.policy = HybridPolicy(q_policy=self._load_q(), sup=self._load_sup()); self.rl = self._load_rl()

    def model_info(self) -> dict:
        import json
        f = settings.models_dir / "MODEL_INFO.json"
        info = json.loads(f.read_text()) if f.exists() else {}
        info["runtime_encoder"] = self.enc.kind
        info["needs_retrain"] = bool(info.get("encoder") and info["encoder"] != self.enc.kind)
        return info

    def health(self) -> dict:
        return {"models": self.model_info(), **{"encoder": self.enc.kind, "classifier_loaded": self.clf is not None, "retrieval_ready": self.index_ready,
                "retrieval_size": len(self.index.labels) if self.index_ready else 0,
                "rewriter_t5_loaded": bool(getattr(self.rewriter.t5, "ok", False)), "q_policy_loaded": self.policy.q is not None, "supervised_policy_loaded": self.policy.sup is not None,
                "policy_version": settings.policy_version}}

    # -- decision history (per client/session) --------------------------------------
    def prev_blocks(self, key) -> int:
        if key is None:
            return 0
        now = time.time(); q = self._recent.get(key, [])
        return int(sum(1 for t, a in q if a == "BLOCK" and now - t < 600) >= 1)

    def record(self, key, action: str):
        if key is None:
            return
        q = self._recent.setdefault(key, []); q.append((time.time(), action)); del q[:-8]

    # -- classifier refinement of clause labels (catches paraphrases the rules miss) --
    def _refine(self, tagged, clf):
        if clf is None:
            return
        for cl in tagged.clauses:
            if cl.label == "TASK" and len(cl.text.split()) >= 3:
                pr = clf.proba(cl.text); p = 1.0 - pr.get("benign", 1.0)
                cat = max((k for k in pr if k != "benign"), key=pr.get)
                # a model-only flag needs high total attack mass AND one dominant attack class (not a diffuse guess)
                if p >= 0.85 and pr[cat] >= 0.60:
                    cl.label = "ATTACK"; cl.hits = [Hit("MODEL_FLAG", cat, round(p, 2), False, cl.text[:80])]
        tagged.attack_hits = list(tagged.attack_hits) + [h for cl in tagged.clauses if cl.label == "ATTACK"
                                                          for h in cl.hits if h.code == "MODEL_FLAG"]
        tagged.task_text = "; and ".join(cl.text.rstrip(" .!") for cl in tagged.clauses if cl.label == "TASK")

    def analyze(self, prompt: str, history: list[str] | None = None, *, clf=None, index=None, session_key=None,
                prev_blocks: int | None = None) -> Analysis:
        clf = clf if clf is not None else self.clf
        index = index if index is not None else self.index
        idx_ready = index is not None and getattr(index, "ready", False)
        T = {}
        def lap(name, since):
            T[name] = round((time.perf_counter() - since) * 1000, 2); return time.perf_counter()
        s = time.perf_counter(); norm = normalize(prompt); s = lap("normalize", s)
        tagged = tag(norm); s = lap("rules_tagging", s)
        full_hits = scan(norm.normalized) + [h for d in norm.decoded_views for h in scan(d, allow_discussion_exemption=False)]
        ctx_risk = scan_context(history or [])
        clf_text = norm.normalized if not history else f"{history[-1]} || {norm.normalized}"
        probs = clf.proba(clf_text) if clf else {}
        attack_p = 1.0 - probs.get("benign", 1.0) if probs else 0.0
        top_cat = max(probs, key=probs.get) if probs else "benign"
        s = lap("classifier", s)
        vec = self.enc.encode([clf_text])[0]; s = lap("encode", s)
        rf = index.features(clf_text, vec=vec) if idx_ready else {"ret_attack_vote": 0.0, "ret_top_sim": 0.0, "ret_ready": 0.0}
        s = lap("retrieval", s)
        self._refine(tagged, clf); s = lap("clause_refine", s)
        cluster = int((self.centroids @ vec).argmax()) if self.centroids is not None and self.centroids.shape[1] == vec.shape[0] else 0
        topic = self.rewriter.kb.lookup(tagged.task_text)[0] if tagged.has_task else None
        rewritability = 0 if not tagged.has_task else (2 if topic else 1)
        allhits = tagged.attack_hits + full_hits
        category = next((h.category for h in allhits if h.category != "benign"), None) or (top_cat if attack_p >= 0.5 else "benign")
        return Analysis(prompt, norm, tagged, sorted({h.code for h in allhits}), max([h.severity for h in allhits] or [0.0]),
                        hard_block(full_hits), ctx_risk, probs, attack_p, top_cat, rf, rewritability, cluster,
                        self.prev_blocks(session_key) if prev_blocks is None else prev_blocks, category, T, history or [])

    def evaluate(self, prompt: str, history: list[str] | None = None, want_rewrite: bool = True, session_key=None, rl_out: dict | None = None) -> GuardResult:
        """rl_out: optional dict the live-RL layer fills with {"event", "an"} so callers can attach rewards/labels."""
        t0 = time.perf_counter()
        an = self.analyze(prompt, history, session_key=session_key)
        T = dict(an.stage_ms); s = time.perf_counter()
        pin = an.policy_input()
        out = self.policy.decide(pin, an.rule_sev); rl_ev = None
        if self.rl is not None and self.rl.mode != "off":
            try:
                ov, rl_ev = self.rl.decide(pin, out, an.rule_sev)       # shadow: proposes + logs; active: may override inside the safety envelope
                if ov is not None: out = ov
            except Exception:
                rl_ev = None
        T["policy"] = round((time.perf_counter() - s) * 1000, 2)
        norm, tagged = an.norm, an.tagged
        category = an.category if out.action != "ALLOW" or an.category != "benign" else "benign"
        if out.action != "ALLOW" and category == "benign":
            category = an.top_cat
        rewritten, final, rw, codes = None, None, None, list(out.reason_codes)
        if out.action == "REWRITE" and want_rewrite:
            s = time.perf_counter()
            r: RewriteResult = self.rewriter.rewrite(norm, tagged); T["rewrite_agent"] = round((time.perf_counter() - s) * 1000, 2)
            rw = {"status": r.status, "intent": r.intent, "removed_span_types": sorted({h.code for h in tagged.attack_hits}),
                  "removed_span_count": len(r.removed_spans), "similarity": r.similarity, "confidence": r.confidence,
                  "attempts": r.attempts, "generator": r.generator, "validation_failures": r.validation_failures}
            if r.status == "REWRITE":
                rewritten = final = r.rewritten_prompt
                out.confidence = round(max(out.confidence * 0.5 + r.confidence * 0.5, 0.0), 3)
                if r.confidence < 0.55:
                    out.needs_review = True
            else:
                out.action = "BLOCK"; codes = r.reason_codes + codes
        elif out.action == "ALLOW":
            final = prompt
        self.record(session_key, out.action)
        if rl_ev is not None:
            rw_info = None
            if rw is not None:
                rw_info = {"ok": rw["status"] == "REWRITE", "unsafe": False, "sim_after": rw.get("similarity"), "ms": T.get("rewrite_agent", 8.0)}
            self.rl.finalize(rl_ev, out.action, rw_info)
        if rl_out is not None:
            rl_out["event"] = rl_ev; rl_out["an"] = an
        total = round((time.perf_counter() - t0) * 1000, 2)
        return GuardResult(out.action, codes, category, out.risk, out.confidence, rewritten, norm.flags, out.needs_review,
                           settings.policy_version, total, T, hashlib.sha256(prompt.encode()).hexdigest(), len(prompt), rw, final,
                           {"encoder": self.enc.kind, "policy_source": out.source, "rl": (self.rl.public_event(rl_ev) if rl_ev is not None else None)})

    def guard_tool(self, tool_name: str, instruction: str, arguments: dict | None = None, sensitivity: str | None = None) -> dict:
        sens = sensitivity or tool_sensitivity(tool_name)
        blob = f"{instruction}\n{arguments or ''}"
        norm = normalize(blob)
        hits = scan(norm.normalized, allow_discussion_exemption=False) + [h for d in norm.decoded_views for h in scan(d, False)]
        p = self.clf.attack_prob(norm.normalized) if self.clf else 0.0
        codes = sorted({h.code for h in hits})
        if hard_block(hits):
            dec, why = "BLOCK", ["HARD_BLOCK"] + codes
        elif hits and sens in ("medium", "high"):
            dec, why = "BLOCK", ["INJECTED_INSTRUCTION_IN_TOOL_CALL"] + codes
        elif hits:
            dec, why = "REQUIRE_CONFIRMATION", ["SUSPICIOUS_LOW_SENSITIVITY"] + codes
        elif p >= 0.85 and sens in ("medium", "high"):
            dec, why = "BLOCK", ["MODEL_HIGH_RISK_ON_SENSITIVE_TOOL"]
        elif sens == "high":
            dec, why = "REQUIRE_CONFIRMATION", ["HIGH_SENSITIVITY_TOOL"]
        else:
            dec, why = "ALLOW", ["TOOL_OK"]
        return {"decision": dec, "tool": tool_name, "sensitivity": sens, "reason_codes": why, "risk_score": round(max(p, max([h.severity for h in hits] or [0])), 3)}
