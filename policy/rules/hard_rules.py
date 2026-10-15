"""Deterministic hard safety rules. These live OUTSIDE the learned (Q-Learning) policy:
the RL policy can never override a HARD_BLOCK hit. Rules are structural (imperative verb +
protected target), and discussion/education about attacks is explicitly exempted so that
security-related but legitimate prompts are not flagged."""
from __future__ import annotations
import re
from dataclasses import dataclass

I = re.I
_X = lambda p: re.compile(p, I | re.S)

RULES = [
    # code, category, severity(0-1), hard_block, pattern
    ("INSTR_OVERRIDE", "prompt_injection", 0.90, False, _X(
        r"\b(ignore|disregard|forget|override|bypass|discard|abandon|drop)\b[^.!?\n]{0,40}?"
        r"\b(previous|prior|above|earlier|original|system|developer|your|all|any|every|those|these)\b[^.!?\n]{0,30}?"
        r"\b(instructions?|rules?|prompts?|guidelines?|polic(?:y|ies)|directions?|directives?|restrictions?|programming|training)\b")),
    ("INSTR_OVERRIDE", "prompt_injection", 0.90, False, _X(
        r"\b(forget|ignore|disregard|erase|unlearn)\b[^.!?\n]{0,15}?\b(everything|anything|all)\b[^.!?\n]{0,25}?"
        r"\b(you (?:were|have been|had been|are) (?:told|given|taught|instructed)|you know|above|before|earlier|so far)\b")),
    ("PROMPT_LEAK", "prompt_leak", 0.90, False, _X(
        r"\b(reveal|disclose|show|print|display|output|repeat|leak|expose|dump|share|tell me|give me|spit out|recite|read out)\b"
        r"[^.!?\n]{0,40}?\b(hidden|secret|confidential|internal|system|developer|initial|original|private|underlying)\b"
        r"[^.!?\n]{0,25}?\b(prompts?|instructions?|config(?:uration)?|rules|messages?|settings|guidelines)\b")),
    ("PERSONA_JAILBREAK", "jailbreak", 0.88, False, _X(
        r"\b(act as|pretend (?:to be|you are|you're)|you are now|roleplay as|role-play as|behave as|simulate|become)\b"
        r"[^.!?\n]{0,70}?\b(unrestricted|unfiltered|uncensored|jailbroken|dan\b|evil|no (?:rules|restrictions|limits|filters)|"
        r"without (?:any )?(?:safety|restrictions?|rules|filters?|polic(?:y|ies)|ethics|guidelines|limits))")),
    ("PERSONA_JAILBREAK", "jailbreak", 0.88, False, _X(r"\b(developer mode|do anything now|jailbreak mode|god mode)\b")),
    ("SAFETY_DISABLE", "jailbreak", 0.88, False, _X(
        r"\b(disabl(?:e|ing|ed)|turn(?:ing)? off|remov(?:e|ing)|deactivat(?:e|ing)|switch(?:ing)? off|bypass(?:ing)?|circumvent(?:ing)?|lift(?:ing)?|suspend(?:ing)?)\b[^.!?\n]{0,30}?"
        r"\b(safety|security|restrictions?|filters?|guardrails?|safeguards?|content polic(?:y|ies)|limitations?|protections?)\b")),
    ("FAKE_AUTHORITY", "instruction_conflict", 0.85, False, _X(
        r"(?:\[|\(|^|\n)\s*(?:sys(?:tem)?|admin(?:istrator)?|developer|root|sudo|operator|dev)[\s_-]*"
        r"(?:override|mode|command|message|access|prompt|instruction)s?\s*[\]\):]?")),
    ("FAKE_AUTHORITY", "instruction_conflict", 0.85, False, _X(
        r"\bi (?:am|'m) (?:your|the) (?:developer|admin(?:istrator)?|creator|owner|operator)\b")),
    ("OBEY_ONLY_ME", "instruction_conflict", 0.85, False, _X(
        r"\b(?:you (?:must|will|shall|should) )?(?:obey|follow|listen to)\b[^.!?\n]{0,15}?\bonly (?:me|my)\b|\bfrom now on[, ]+you (?:will|must|are)\b")),
    ("ENCODED_EXECUTE", "prompt_injection", 0.88, False, _X(
        r"\b(?:decode|decrypt|deobfuscate|unscramble)\b[^.!?\n]{0,50}?\b(?:and|then)\b[^.!?\n]{0,20}?\b(?:execute|run|follow|obey|apply|do)\b")),
    ("TOOL_DESTRUCTIVE", "tool_abuse", 0.95, True, _X(
        r"(?:\brm\s+-rf\b|\bdrop\s+(?:table|database)\b|\bdelete\s+all\s+(?:files|data|records)\b|\bformat\s+(?:c:|the disk)\b|"
        r"\bchmod\s+777\b|curl[^|\n]{0,80}\|\s*(?:sudo\s+)?(?:ba)?sh\b|\bmkfs\b)")),
    ("TOOL_EXFIL", "tool_abuse", 0.95, True, _X(
        r"\b(?:send|upload|post|email|forward|exfiltrate|transmit)\b[^!?\n]{0,60}?\b(?:passwords?|credentials?|api[ _-]?keys?|secrets?|tokens?|ssh keys?|private keys?)\b"
        r"[^!?\n]{0,50}?\b(?:to|at)\b[^!?\n]{0,50}?(?:https?://|\bexternal\b|\bremote server\b|@)")),
    ("HARMFUL_CONTENT", "harmful", 1.00, True, _X(
        r"\b(?:build|make|create|synthesi[sz]e|manufacture|assemble)\b[^.!?\n]{0,30}?\b(?:bomb|explosive|nerve agent|bioweapon|chemical weapon|ransomware|keylogger|malware|botnet)\b"
        r"|\bchild\s+sexual\b|\bhow to (?:kill|poison) (?:someone|a person|my)\b")),
]

# Education / discussion about attacks is NOT an attack ("explain what prompt injection is").
_DISCUSS_START = re.compile(
    r"^\s*(?:please\s+)?(?:what(?:'s| is| are| does)|why|how (?:do|does|can|to|would)|explain|describe|define|"
    r"can you explain|could you explain|tell me (?:what|about|why)|give (?:me )?an example of|summari[sz]e|"
    r"how (?:do|does|can) .{0,40}(?:detect|prevent|defend|mitigate))", I)
_META = re.compile(r"\b(prompt injection|jailbreak(?:ing)?s?|injection attacks?|attacks?|guardrails?|defen[sc]e|defend|detect|"
                   r"mitigat\w+|phrase|term|meaning|example|red[- ]team\w*|security|research|awareness|training|"
                   r"classif\w+|dataset|ai safety|llm safety)\b", I)
_QUOTED = re.compile(r"[\"'“”‘’`][^\"'“”‘’`]{3,}[\"'“”‘’`]")
_NEGATED = re.compile(r"\b(?:without|never|not|don't|do not|avoid|refrain from)\s+(?:\w+ing\s+)?[^,.;]{0,60}", I)


@dataclass
class Hit:
    code: str
    category: str
    severity: float
    hard_block: bool
    span: str

    def as_dict(self):
        return {"code": self.code, "category": self.category, "severity": self.severity,
                "hard_block": self.hard_block}


def is_discussion(text: str) -> bool:
    """True if the text is *about* attacks (education/defense) rather than issuing one."""
    return bool(_DISCUSS_START.match(text) and _META.search(text))


def _strip_quotes(text: str) -> str:
    return _QUOTED.sub(" ", text)


def scan(text: str, allow_discussion_exemption: bool = True, strip_negations: bool = False) -> list[Hit]:
    """Return rule hits for `text`. Discussion/education framing and quoted examples are exempt
    (except destructive/exfil/harm rules, which are hard blocks no matter the framing)."""
    t = text or ""
    if strip_negations:
        t = _NEGATED.sub(" ", t)
    discussion = allow_discussion_exemption and is_discussion(t)
    probe = _strip_quotes(t) if allow_discussion_exemption else t
    hits: list[Hit] = []
    for code, cat, sev, hard, rx in RULES:
        m = rx.search(probe)
        if not m:
            continue
        if discussion and not hard:
            continue
        if code == "INSTR_OVERRIDE" and re.search(r"\bmy\b", m.group(0), I) and not re.search(r"\b(your|system|developer)\b", m.group(0), I):
            continue    # user amending their own earlier instruction ("ignore my previous instructions")
        hits.append(Hit(code, cat, sev, hard, m.group(0)[:120]))
    seen, uniq = set(), []
    for h in hits:
        if (h.code, h.span) not in seen:
            seen.add((h.code, h.span)); uniq.append(h)
    return uniq


def hard_block(hits: list[Hit]) -> bool:
    return any(h.hard_block for h in hits)


# ---- multi-turn context manipulation (scan previous USER turns) -----------------------
_CONTEXT_SETUP = [
    _X(r"\b(no|without any|zero)\s+(rules|restrictions|limits|filters|safety)\b.{0,40}\b(always|comply|obey|answer)\b"),
    _X(r"\b(safety|security)\s+(policies|rules|filters|guardrails)\s+(are\s+)?(turned off|disabled|off|removed|suspended)\b"),
    _X(r"\b(treat|consider)\s+me\s+as\s+(your|the)\s+(admin|administrator|developer|owner|root)\b"),
    _X(r"\byou\s+(have|are under)\s+no\s+(rules|restrictions|limits|guidelines)\b"),
    _X(r"\blet'?s\s+play\s+a\s+game\s+where\s+you\b.{0,50}\b(no rules|no restrictions|always comply|never refuse)\b"),
]


def scan_context(history: list[str]) -> bool:
    """True if earlier user turns tried to set up a jailbreak / authority framing."""
    for turn in history or []:
        if any(rx.search(turn) for rx in _CONTEXT_SETUP) or scan(turn, allow_discussion_exemption=True):
            return True
    return False


# ---- agent/tool sensitivity --------------------------------------------------------------
HIGH_TOOLS = re.compile(r"(delete|remove|drop|exec|shell|bash|payment|transfer|wire|admin|sudo|deploy|write_file|chmod|kill)", I)
MEDIUM_TOOLS = re.compile(r"(send|email|post|upload|update|create|write|commit|push|message|publish)", I)


def tool_sensitivity(tool_name: str | None) -> str:
    n = tool_name or ""
    return "high" if HIGH_TOOLS.search(n) else "medium" if MEDIUM_TOOLS.search(n) else "low"
