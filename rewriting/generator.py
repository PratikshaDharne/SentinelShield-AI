"""Rewrite generators (both are OUR OWN models - no external LLM):
  * TemplateRewriter: deterministic composer (always available, used as the safe fallback)
  * T5Rewriter: fine-tuned Flan-T5 (models/rewriter_t5). Trained by scripts/train_rewriter.py.
"""
from __future__ import annotations
import re
from pathlib import Path
from configs.settings import settings

_SUFFIX = {
    "leak": "without disclosing confidential system instructions",
    "safety": "while following applicable safety guidelines",
}


def normalize_task(task: str) -> str:
    """'explain what is machine learning' -> 'Explain machine learning'."""
    t = re.sub(r"\s+", " ", task).strip(" .!?;,")
    t = re.sub(r"^(?:please\s+|can you\s+|could you\s+|would you\s+|kindly\s+)+", "", t, flags=re.I)
    t = re.sub(r"^(explain|describe|tell me|define)\s+(?:to me\s+)?(?:what|who)\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?", r"\1 ", t, flags=re.I)
    t = re.sub(r"^(?:what|who)\s+(?:is|are)\s+(?:an?\s+|the\s+)?(.+)$", r"Explain \1", t, flags=re.I)
    t = re.sub(r"^tell me (?:about|how)\s+", "Explain ", t, flags=re.I)
    m = re.match(r"^(explain|describe)\s+the working of\s+(.+)$", t, re.I)
    if m:
        subj = m.group(2)
        t = f"{m.group(1)} how {subj} " + ("work" if subj.lower().endswith("s") else "works")
    t = re.sub(r"^how (does|do|did|can)\s+(.+?)\s+(work|works|function|functions)$", r"Explain how \2 works", t, flags=re.I)
    t = re.sub(r"^how (?:to|do i|can i)\s+(.+)$", r"Explain how to \1", t, flags=re.I)
    return t[:1].upper() + t[1:] if t else t


def context_label(context: str) -> str | None:
    m = re.search(r"\b(?:for|as)\s+(?:my|our|a|an|the)\s+(.+?\b(?:assignment|project|class|course|report|presentation|essay|exam|interview|thesis|homework|lecture|workshop|paper))\b", context or "", re.I)
    return m.group(1).strip() if m else None


class TemplateRewriter:
    name = "template"

    def candidates(self, task: str, context: str, scope: str | None, constraints: list[str], n: int = 3) -> list[str]:
        base = normalize_task(task)
        label = context_label(context)
        suffix = [_SUFFIX[c] for c in constraints if c in _SUFFIX]
        out = []
        for v in range(n):
            core = base
            has_scope = bool(scope) and not re.search(r"\b(including|covering|with)\b", base, re.I)
            if v == 0:
                s = core + (f", including {scope}" if has_scope else "")
            elif v == 1:
                s = core + (f", covering {scope}" if has_scope else "")
            else:
                s = core            # minimal, un-enriched fallback (safest, most faithful)
            if label:
                s += f" in a {label.replace(' assignment', '-assignment').replace(' project', '-project')} format" \
                     if v < 2 else f" for my {label}"
            if suffix:
                s += (", " if v < 2 else " ") + " and ".join(suffix)
            out.append(s.rstrip(" .") + ".")
        return out


class T5Rewriter:
    name = "t5"

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.models_dir / "rewriter_t5")
        self.ok = False
        self.model = self.tok = None
        if settings.rewriter_backend == "template" or not self.path.exists():
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tok = AutoTokenizer.from_pretrained(self.path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path).eval()
            self.ok = True
        except Exception:
            self.ok = False

    @staticmethod
    def build_input(task: str, context: str, scope: str | None, constraints: list[str]) -> str:
        return (f"rewrite safely: task: {normalize_task(task)} | context: {context or 'none'} | "
                f"scope: {scope or 'none'} | constraints: {', '.join(constraints) or 'none'}")

    def candidates(self, task, context, scope, constraints, n: int = 3) -> list[str]:
        if not self.ok:
            return []
        import torch
        enc = self.tok(self.build_input(task, context, scope, constraints), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=128, num_beams=max(4, n), num_return_sequences=n, early_stopping=True)
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]
