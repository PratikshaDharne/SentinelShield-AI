"""Run the FINAL held-out evaluation: 7 baselines, 8 experiments, 10 visualization datasets, failure analysis, report.
  python scripts/run_evaluation.py                 # everything except Ollama
  python scripts/run_evaluation.py --with-ollama   # + Llama Guard 3 baseline and LLM-judge (Ollama OR EVAL_JUDGE=gemini). Run once, by hand.
Uses ONLY final_test (fresh seed), unseen_test and the shift sets - never train/val. Results go to evaluation/results/*.json."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.settings import settings
from experiments.common import build_ctx, SETS, MAIN_SET
from experiments import (exp1_rules_vs_learned as e1, exp2_allowblock_vs_arb as e2, exp3_classifier_vs_retrieval as e3,
                         exp4_baseline_vs_q as e4, exp5_intent_preservation as e5, exp6_distribution_shift as e6,
                         exp7_latency as e7, exp8_ablation as e8, exp9_live_rl as e9)
from evaluation.metrics import evaluate, effective
from policy.hybrid import HybridPolicy, risk_score
from policy.baselines import ClassifierOnly, RetrievalAssisted

R = settings.results_dir


def save(name, obj):
    R.mkdir(parents=True, exist_ok=True); (R / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str))


class ThresholdHybrid:
    """Hybrid whose ambiguity zone escalates at risk >= t (REWRITE if a task exists, else BLOCK): traces the security-utility curve."""
    def __init__(self, t): self.t, self.p = t, HybridPolicy()
    def act(self, rec):
        pin = rec.an.policy_input(); a = self.p.decide(pin, rec.an.rule_sev).action
        if a == "ALLOW" and risk_score(pin, rec.an.rule_sev) >= self.t:
            return "REWRITE" if pin.has_task else "BLOCK"
        return a


def sweep(c, make, ts, sname=MAIN_SET):
    pts = []
    for t in ts:
        pol = make(t); recs = c.recs[sname]; m = evaluate(c.g, recs, [pol.act(r) for r in recs])
        pts.append({"threshold": round(float(t), 3), "security_recall": m["security_recall"], "false_positive_rate": m["false_positive_rate"],
                    "legit_utility": m["legit_utility"], "violation_rate": m["safety_constraint_violation_rate"]})
    return pts


def q_stats(c):
    if not c.q: return None
    Q, V = c.q.Q, c.q.visits; seen = V.sum(1) > 0
    best = Q[seen].argmax(1); acts = ["ALLOW", "REWRITE", "BLOCK"]
    return {"visited_states": int(seen.sum()), "argmax_action_counts": {a: int((best == i).sum()) for i, a in enumerate(acts)},
            "mean_q_by_action_visited": {a: round(float(Q[seen][:, i].mean()), 3) for i, a in enumerate(acts)},
            "max_q_histogram": np.histogram(Q[seen].max(1), bins=12)[0].tolist(), "max_q_bin_edges": [round(x, 2) for x in np.histogram(Q[seen].max(1), bins=12)[1]],
            "visits_per_state_quantiles": {str(q): int(np.percentile(V[seen].sum(1), q)) for q in (10, 50, 90)},
            "chosen_actions_on_main_set": c.run("q_learning", MAIN_SET)["action_counts"], "meta": c.q.meta}


def failures(c):
    out = {"attack_passed": [], "legit_blocked": [], "rewrite_failed": []}
    for pname in ("hybrid_sup" if "hybrid_sup" in c.policies else "hybrid_base",):
        for s in SETS:
            m = c.run(pname, s)
            for rec, a in zip(c.recs[s], m["_actions"]):
                e = effective(c.g, rec, a)[0]; row = {"set": s, "category": rec.category, "gt": rec.gt, "action": a, "effective": e, "text": rec.row["text"][:120]}
                if rec.gt != "ALLOW" and e == "ALLOW": out["attack_passed"].append(row)
                elif rec.gt == "ALLOW" and e != "ALLOW": out["legit_blocked"].append(row)
                elif rec.gt == "REWRITE" and a == "REWRITE" and e == "BLOCK": out["rewrite_failed"].append(row)
    return {k: {"count": len(v), "examples": v[:15]} for k, v in out.items()}


def report(c, S):
    L = ["# SentinelShield AI - Final Held-out Evaluation", "",
         f"Encoder: **{c.g.enc.kind}** - policy version {settings.policy_version} - official test set: `final_test` (fresh seed, de-duplicated), plus `unseen_test` and shift sets.",
         "", "> Training mix: real public datasets (scripts/download_github_datasets.py) + synthetic data for classes without public data. `real_test` = held-out REAL prompts (family-disjoint). `final_test`/`unseen_test`/shift sets are synthetic.", "",
         "## 1. Baselines on final_test", "", "| policy | decision acc | security recall | violation rate | false-positive rate | legit blocked | legit utility | p95 ms |", "|---|---|---|---|---|---|---|---|"]
    for p in [x for x in ("keyword_rules", "classifier_only", "llama_guard3", "retrieval_assisted", "rules_3way", "supervised_arb", "hybrid_base", "hybrid_sup", "q_learning", "dqn_rl") if x in c.policies]:
        m = c.run(p, MAIN_SET)
        L.append(f"| {p} | {m['decision_accuracy']} | {m['security_recall']} | {m['safety_constraint_violation_rate']} | {m['false_positive_rate']} | {m['legit_blocked_rate']} | {m['legit_utility']} | {m['latency_ms']['p95']} |")
    if "real_test" in SETS:
        L += ["", f"## 1b. Baselines on real_test (REAL public-dataset prompts, n={len(c.recs['real_test'])}, family-disjoint from training)", "",
              "| policy | decision acc | security recall | violation rate | false-positive rate | legit blocked |", "|---|---|---|---|---|---|"]
        for p in [x for x in ("keyword_rules", "classifier_only", "retrieval_assisted", "rules_3way", "supervised_arb", "hybrid_base", "hybrid_sup", "q_learning", "dqn_rl") if x in c.policies]:
            m = c.run(p, "real_test")
            L.append(f"| {p} | {m['decision_accuracy']} | {m['security_recall']} | {m['safety_constraint_violation_rate']} | {m['false_positive_rate']} | {m['legit_blocked_rate']} |")
    L += ["", "## 2. Distribution shift (security recall / violation rate / false-positive rate)", "", "| policy | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for p, row in S["e6"]["grid"].items():
        L.append(f"| {p} | " + " | ".join(f"{row[s]['security_recall']} / {row[s]['safety_constraint_violation_rate']} / {row[s]['false_positive_rate']}" for s in SETS) + " |")
    if "paired_changes" in S["e4"]:
        qm = S["e4"].get("q_meta", {})
        L += ["", "## 3. Does RL help? (decisions changed by Q-Learning vs its safe baseline: hard rules + supervised policy)", "",
              f"Q-policy validation verdict: **{'RECOMMENDED' if qm.get('recommended') else 'NOT recommended - did not beat the baseline on validation; disabled in the live API unless FORCE_Q_POLICY=true'}** "
              f"(selected lambdas {qm.get('lambdas')}, margin {qm.get('margin')}).", "", "| set | changed | fixed | regressed |", "|---|---|---|---|"]
        for s, d in S["e4"]["paired_changes"].items(): L.append(f"| {s} | {d['changed']} | {d['fixed_by_q']} | {d['regressed_by_q']} |")
    if "paired_changes" in S.get("e9", {}):
        L += ["", "## 3b. Live deep-RL agent (DQN) vs the supervised baseline", "", "| set | baseline reward | DQN reward | baseline violations | DQN violations | changed | fixed | regressed |", "|---|---|---|---|---|---|---|---|"]
        for s_, d in S["e9"]["results"].items():
            b_, q_ = d[S["e9"]["reference_policy"]], d["dqn_rl"]; pc = S["e9"]["paired_changes"][s_]
            L.append(f"| {s_} | {b_['mean_reward']} | {q_['mean_reward']} | {b_['safety_constraint_violation_rate']} | {q_['safety_constraint_violation_rate']} | {pc['changed']} | {pc['fixed_by_rl']} | {pc['regressed_by_rl']} |")
    L += ["", "## 4. Intent preservation (rewrites on true REWRITE cases)", ""]
    for s, d in S["e5"]["per_set"].items(): L.append(f"- **{s}**: released {d['released_rate']}, intent preserved {d['intent_preserved_rate']}, unsafe among released {d['unsafe_rate_among_released']}, similarity {d['mean_similarity_before']} -> {d['mean_similarity_after']}")
    L += ["", "## 5. Latency", "", f"- overall {S['e7']['overall_ms']} ms, throughput {S['e7']['throughput_rps_8_threads']} req/s (8 threads). {S['e7']['note']}",
          "", "## 6. Ablation (accuracy drop when a signal group is removed)", ""]
    for d, v in S["e8"]["accuracy_drop_when_removed"].items(): L.append(f"- **{d}**: " + ", ".join(f"{s} {x:+}" for s, x in v.items()))
    L += ["", "## 7. Failure analysis", ""]
    for k, v in S["fail"].items():
        L.append(f"### {k} ({v['count']})")
        L += [f"- [{e['set']}/{e['category']}] gt={e['gt']} action={e['action']}: `{e['text']}`" for e in v["examples"][:8]]
    L += ["", "## 8. Limitations (PDF rule 20)", "",
          "- Protection is reported only for the tested categories (injection, jailbreak, leakage, instruction conflict, tool abuse, obfuscation, multi-turn manipulation). No guarantee against all attacks.",
          "- Security recall and safety-constraint violations are reported separately; hard blocks sit outside the Q-policy and cannot be overridden.",
          "- Synthetic benchmark: wrapper phrasing is templated. Real-traffic datasets + MiniLM are required for deployment-grade claims.",
          "- Decision history (prev_blocks) is not exercised in single-request evaluation; the Q-policy was trained with it in the simulator."]
    (R / "final_report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--with-ollama", action="store_true"); a = ap.parse_args()
    t0 = time.time(); R.mkdir(parents=True, exist_ok=True)
    if a.with_ollama:
        from evaluation import ollama_eval as oe
        if oe.ollama_up() or settings.eval_judge == "gemini":
            c0 = build_ctx(); rows = [r.row for s in SETS for r in c0.recs[s]]
            if oe.ollama_up():
                print(f"Llama Guard baseline on {len(rows)} rows ..."); oe.llama_guard_predictions(rows)
            e5.run(c0)                                                     # writes judge_pairs.json
            pairs = json.loads((R / "judge_pairs.json").read_text())
            save("judge_scores", oe.judge_rewrites(pairs[:200]))
        else:
            print("Ollama not reachable at", settings.ollama_url, "- skipping Llama Guard / judge (start Ollama and re-run).")
    c = build_ctx(); S = {}
    for name, mod in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5), ("e6", e6), ("e7", e7), ("e8", e8), ("e9", e9)):
        print("running", name); S[name] = mod.run(c); save(f"exp_{name}_{mod.__name__.split('.')[-1].split('_', 1)[1]}", S[name])
    S["fail"] = failures(c)
    ts = np.linspace(0.05, 0.95, 19)
    viz = {"1_confusion_matrix": {p: c.run(p, MAIN_SET)["confusion"] for p in ("hybrid_sup", "q_learning") if p in c.policies},
           "2_recall_vs_fpr": {"classifier_only": sweep(c, ClassifierOnly, ts), "retrieval_assisted": sweep(c, RetrievalAssisted, ts),
                               "points": {p: {"fpr": c.run(p, MAIN_SET)["false_positive_rate"], "recall": c.run(p, MAIN_SET)["security_recall"]} for p in c.policies if not p.startswith("abl_")}},
           "3_security_utility_curve": sweep(c, ThresholdHybrid, ts),
           "4_intent_preservation": S["e5"]["histogram"], "5_latency_distribution": {"histogram_ms": S["e7"]["histogram_ms"], "percentiles": S["e7"]["overall_ms"], "stages": S["e7"]["mean_stage_ms"]},
           "6_by_threat_category": {p: c.run(p, MAIN_SET)["breakdowns"]["by_category"] for p in ("hybrid_sup", "q_learning") if p in c.policies},
           "7_by_length": {p: {"prompt": c.run(p, MAIN_SET)["breakdowns"]["by_prompt_length"], "context": c.run(p, MAIN_SET)["breakdowns"]["by_context_length"]} for p in ("hybrid_sup", "q_learning") if p in c.policies},
           "8_distribution_shift": S["e6"]["grid"], "9_q_values": q_stats(c), "10_ablation": S["e8"]["variants"]}
    save("visualizations", viz); save("failure_analysis", S["fail"])
    save("final_heldout_eval", {"encoder": c.g.enc.kind, "policy_version": settings.policy_version, "sets": {s: len(c.recs[s]) for s in SETS},
                                "metrics": {p: {s: c.public(c.run(p, s)) for s in SETS} for p in c.policies if not p.startswith("abl_")}})
    report(c, S)
    print(f"done in {time.time() - t0:.0f}s -> {R}")


if __name__ == "__main__":
    main()
