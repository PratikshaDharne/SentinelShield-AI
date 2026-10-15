"""Train the supervised Allow/Rewrite/Block baseline on OUT-OF-FOLD train features; report on val."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.pipeline import Guardrail
from evaluation.features import load_rows, oof_analyze, analyze_rows
from evaluation.metrics import evaluate
from policy.supervised.model import SupervisedPolicy
from configs.settings import settings

if __name__ == "__main__":
    g = Guardrail.get()
    tr = oof_analyze(load_rows("train"), g); va = analyze_rows(load_rows("val"), g)
    sp = SupervisedPolicy().fit(tr); sp.save()
    m = evaluate(g, va, [sp.act(r) for r in va], "supervised_arb")
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "supervised_val.json").write_text(json.dumps(m, indent=2))
    print({k: m[k] for k in ("decision_accuracy", "security_recall", "false_positive_rate", "safety_constraint_violation_rate")})
