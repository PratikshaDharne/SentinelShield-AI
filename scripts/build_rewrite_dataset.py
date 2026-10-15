
"""Build (attack-wrapped prompt -> clean enriched prompt) pairs to fine-tune OUR Flan-T5 rewriter.
No external LLM is used. Inputs are produced by the same tagger the live system uses, so train and
inference formats match. Targets use several phrasings so the model learns fluent composition.
  python scripts/build_rewrite_dataset.py"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.normalizer import normalize
from rewriting.span_tagger import tag
from rewriting.topics import TopicKB
from rewriting.generator import normalize_task, context_label, T5Rewriter, _SUFFIX
from rewriting.agent import RewriteAgent
from configs.settings import settings
from policy.rules.hard_rules import scan


rnd = random.Random(11)
kb = TopicKB()

STYLES = [
    lambda t, sc, cl, sf: f"{t}, including {sc}" if sc else t,
    lambda t, sc, cl, sf: f"{t}, covering {sc}" if sc else t,
    lambda t, sc, cl, sf: f"{t}, with a focus on {sc}" if sc else t,
    lambda t, sc, cl, sf: f"{t}. Include {sc}" if sc else t,
    lambda t, sc, cl, sf: (
        f"Please {t[0].lower() + t[1:]}, addressing {sc}"
        if sc
        else f"Please {t[0].lower() + t[1:]}"
    ),
]


def target(task, ctx, scope, cons, style):
    t = normalize_task(task)
    s = STYLES[style](t, scope, None, None)

    lab = context_label(ctx)

    if lab:
        s += f" in a {lab.replace(' ', '-', 1)} format"

    suf = [_SUFFIX[c] for c in cons if c in _SUFFIX]

    if suf:
        s += ", " + " and ".join(suf)

    return s.rstrip(" .") + "."


def main():
    out = {
        "train": [],
        "val": []
    }

    for sp in ("train", "val"):
        data_path = settings.datasets_dir / "data" / f"{sp}.jsonl"

        for line in data_path.open(encoding="utf-8"):
            r = json.loads(line)

            if r["decision"] != "REWRITE" or not r.get("has_task"):
                continue

            tg = tag(normalize(r["text"]))

            if not tg.has_task:
                continue

            task = tg.task_text

            _, scope = kb.lookup(task)
            scope = scope or kb.generic_scope(task) or None

            cons = RewriteAgent._constraints(tg, task)

            tgt = target(
                task,
                tg.context_text,
                scope,
                cons,
                rnd.randrange(len(STYLES))
            )

            if scan(task, False) or scan(
                tgt,
                False,
                strip_negations=True
            ):
                continue

            out[sp].append(
                {
                    "input": T5Rewriter.build_input(
                        task,
                        tg.context_text,
                        scope,
                        cons
                    ),
                    "target": tgt
                }
            )

    gold_path = settings.datasets_dir / "rewrite_gold.jsonl"

    for line in gold_path.open(encoding="utf-8"):
        g = json.loads(line)

        tg = tag(normalize(g["prompt"]))

        _, scope = kb.lookup(tg.task_text)

        cons = RewriteAgent._constraints(
            tg,
            tg.task_text
        )

        out["train"] += [
            {
                "input": T5Rewriter.build_input(
                    tg.task_text,
                    tg.context_text,
                    scope,
                    cons
                ),
                "target": g["expected"]
            }
        ] * 15

    rnd.shuffle(out["train"])

    for sp, rows in out.items():
        p = settings.datasets_dir / f"rewrite_pairs_{sp}.jsonl"

        p.write_text(
            "\n".join(
                json.dumps(
                    x,
                    ensure_ascii=False
                )
                for x in rows
            ),
            encoding="utf-8"
        )

        print(sp, len(rows), "->", p)


if __name__ == "__main__":
    main()
