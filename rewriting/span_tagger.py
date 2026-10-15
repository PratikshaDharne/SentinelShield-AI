"""Span / clause tagger: splits a prompt into clauses and labels each as
ATTACK | TASK | CONTEXT | NOISE. Rule-structural by default; a trained clause classifier
(models/span_tagger.joblib) can refine ATTACK-vs-TASK for clauses the rules are unsure about."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from policy.rules.hard_rules import scan, is_discussion, Hit
from api.normalizer import NormResult

TASK_VERBS = (r"explain|describe|tell|show|write|summari[sz]e|list|teach|give|help|create|compare|define|outline|walk|"
              r"generate|draft|translate|solve|calculate|find|suggest|recommend|analy[sz]e|discuss|elaborate|"
              r"illustrate|demonstrate|provide|make|design|plan|review|debug|fix|convert|rewrite|improve|"
              r"how|what|why|when|where|who|which|can you|could you|please")
_TV = re.compile(rf"^\s*(?:please\s+)?(?:{TASK_VERBS})\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_SPLIT_MID = re.compile(
    rf"\s*,\s*(?:and\s+)?then\s+|\s+and\s+then\s+|\s*;\s*|\s+and\s+(?=(?:please\s+)?(?:{TASK_VERBS})\b)|\s*,\s*(?=(?:please\s+)?(?:explain|describe|tell|show|write|summari[sz]e|list|teach|compare|define|outline)\b)|\s*,?\s+and\s+(?=(?:also\s+)?(?:ignore|disregard|forget|reveal|disclose|disable|bypass|override|pretend|act as|print your|show your|you must)\b)",
    re.I)
_LEAD = re.compile(r"^\s*(?:and\s+)?(?:then|after that|afterwards|finally|first|next|also|now|lastly|second|third|"
                   r"before (?:answering|you answer|responding|replying)|once done|when done)\b\s*[,:]?\s*", re.I)
_TAG = re.compile(r"^\s*(?:\[[^\]]{2,40}\]|\([^)]{2,40}\))\s*[:\-]?\s*")
_AUTH_PREFIX = re.compile(r"^\s*(?:admin|system|developer|root|operator)(?:istrator)?\s*(?:override|mode|message|command)\s*:\s*", re.I)
_CTX = re.compile(r"^\s*(?:for (?:my|our|a|an|the) |as (?:a|an|the) |i(?:'m| am) (?:a|an) |in the context of |"
                  r"my (?:teacher|professor|manager|boss)|this is for )", re.I)


@dataclass
class Clause:
    text: str
    label: str                       # ATTACK | TASK | CONTEXT | NOISE
    hits: list[Hit] = field(default_factory=list)


@dataclass
class Tagged:
    clauses: list[Clause]
    attack_hits: list[Hit]
    task_text: str
    context_text: str

    @property
    def has_attack(self): return bool(self.attack_hits)
    @property
    def has_task(self): return bool(self.task_text.strip())


_CTX_PREFIX = re.compile(r"^((?:for (?:my|our|a|an|the) |as (?:a|an|the) |i(?:'m| am) (?:a|an) |in the context of |this is for )[^,:;]{2,60}),\s+(.+)$", re.I | re.S)


def split_clauses(text: str) -> list[str]:
    out = []
    for sent in _SENT.split(text or ""):
        sent = sent.strip()
        if not sent:
            continue
        for part in _SPLIT_MID.split(sent):
            part = part.strip(" \t,;:-")
            part = _LEAD.sub("", part).strip(" \t,;:-")
            m = _CTX_PREFIX.match(part)
            if m:                                   # "As a student, how does X work?" -> context + task
                out.append(m.group(1).strip()); part = m.group(2).strip()
            if part:
                out.append(part)
    return out


def tag(norm: NormResult) -> Tagged:
    text = norm.normalized
    decoded_attack = [h for d in norm.decoded_views for h in scan(d, allow_discussion_exemption=False)]
    # extract fake-authority tags such as "[SYS OVERRIDE]" / "ADMIN OVERRIDE:" as attack hits
    pre_hits: list[Hit] = []
    m = _TAG.match(text)
    if m:
        pre_hits += scan(m.group(0), allow_discussion_exemption=False)
        text = text[m.end():]
    m = _AUTH_PREFIX.match(text)
    if m:
        pre_hits += scan(m.group(0), allow_discussion_exemption=False) or []
        text = text[m.end():]
        if not pre_hits:
            from policy.rules.hard_rules import Hit as H
            pre_hits.append(H("FAKE_AUTHORITY", "instruction_conflict", 0.85, False, m.group(0).strip()))
    clauses: list[Clause] = []
    text = text.replace("[ENCODED_PAYLOAD]", "[ENCODED_PAYLOAD].")
    for c in split_clauses(text):
        placeholder = "[ENCODED_PAYLOAD]" in c
        c_clean = c.replace("[ENCODED_PAYLOAD]", " ").strip(" :")
        hits = scan(c)
        if placeholder and (decoded_attack or re.search(r"\b(execute|run|follow|obey)\b", c, re.I)):
            hits = hits or [Hit("ENCODED_EXECUTE", "prompt_injection", 0.88, False, c[:80])]
        if hits:
            clauses.append(Clause(c, "ATTACK", hits))
        elif not c_clean or not re.search(r"[A-Za-z]{2}", c_clean):
            clauses.append(Clause(c, "NOISE"))
        elif _CTX.match(c_clean):
            clauses.append(Clause(c_clean, "CONTEXT"))
        elif _TV.match(c_clean) or c_clean.endswith("?") or is_discussion(c_clean):
            clauses.append(Clause(c_clean, "TASK"))
        elif len(c_clean.split()) >= 4:
            clauses.append(Clause(c_clean, "TASK"))
        else:
            clauses.append(Clause(c_clean, "PREFIX"))      # short non-task lead-in such as "In Python"
    # re-attach a lead-in ("In Python") to the task that follows it so no meaning is lost
    merged: list[Clause] = []
    for i, cl in enumerate(clauses):
        if cl.label == "PREFIX":
            nxt = clauses[i + 1] if i + 1 < len(clauses) else None
            if nxt is not None and nxt.label == "TASK":
                nxt.text = f"{cl.text}, {nxt.text}"; continue
            cl.label = "CONTEXT"
        merged.append(cl)
    clauses = merged
    attack_hits = pre_hits + [h for cl in clauses if cl.label == "ATTACK" for h in cl.hits] + decoded_attack
    task = "; and ".join(cl.text.rstrip(" .!") for cl in clauses if cl.label == "TASK")
    ctx = "; ".join(cl.text.rstrip(" .!,") for cl in clauses if cl.label == "CONTEXT")
    return Tagged(clauses, attack_hits, task, ctx)
