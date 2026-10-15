"""Fill docs/TECHNICAL_REPORT.md with the numbers in evaluation/results (run after run_evaluation.py)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "evaluation" / "results"


def load(n):
    f = R / f"{n}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def pct(x): return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    fm = load("final_heldout_eval"); m = fm.get("metrics", {}); ft = {k: v.get("final_test", {}) for k, v in m.items()}
    e4 = load("exp_e4_baseline_vs_q"); e7 = load("exp_e7_latency"); e5 = load("exp_e5_intent_preservation"); fa = load("failure_analysis")
    rows = "\n".join(f"| {n} | {pct(v.get('decision_accuracy'))} | {pct(v.get('security_recall'))} | {pct(v.get('safety_constraint_violation_rate'))} | "
                     f"{pct(v.get('false_positive_rate'))} | {pct(v.get('legit_blocked_rate'))} | {pct(v.get('legit_utility'))} |" for n, v in ft.items())
    ov = e7.get("overall_ms", {})
    tpl = (ROOT / "docs" / "TECHNICAL_REPORT.template.md").read_text(encoding="utf-8")
    out = (tpl.replace("{{ENCODER}}", str(fm.get("encoder", "?"))).replace("{{POLICY_VERSION}}", str(fm.get("policy_version", "?")))
           .replace("{{SETS}}", ", ".join(f"{k} (n={v})" for k, v in fm.get("sets", {}).items())).replace("{{BASELINE_TABLE}}", rows)
           .replace("{{Q_VERDICT}}", "recommended for live use" if e4.get("q_recommended") else "NOT recommended - did not beat the supervised baseline on validation; disabled live")
           .replace("{{LATENCY}}", f"p50 {ov.get('p50')} ms, p95 {ov.get('p95')} ms, p99 {ov.get('p99')} ms; throughput {e7.get('throughput_rps_8_threads')} req/s (8 threads)")
           .replace("{{INTENT}}", "; ".join(f"{k}: preserved {pct(v.get('intent_preserved_rate'))}, released {pct(v.get('released_rate'))}" for k, v in e5.get("per_set", {}).items()))
           .replace("{{FAILURES}}", "; ".join(f"{k.replace('_', ' ')}: {v.get('count')}" for k, v in fa.items())))
    (ROOT / "docs" / "TECHNICAL_REPORT.md").write_text(out, encoding="utf-8")
    print("wrote docs/TECHNICAL_REPORT.md")


if __name__ == "__main__":
    sys.exit(main())
