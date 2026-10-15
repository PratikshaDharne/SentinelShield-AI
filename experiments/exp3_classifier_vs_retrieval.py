"""E3 - Does semantic retrieval improve context over the classifier alone?"""
from experiments.common import SETS

def run(c):
    pols = ("classifier_only", "retrieval_only", "retrieval_assisted")
    return {"question": "Does semantic retrieval improve context?", "encoder": c.g.enc.kind,
            "note": "With the 'hash' encoder retrieval is lexical; run with MiniLM (sentence-transformers) for the semantic result.",
            "results": {p: {s: c.public(c.run(p, s)) for s in SETS} for p in pols}}
