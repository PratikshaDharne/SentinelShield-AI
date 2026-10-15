
"""Train the Q-Learning policy OFFLINE in the simulator.

1. Fit semantic-cluster centroids on training prompt embeddings.
2. Build out-of-fold training features and full-model validation features.
3. Search reward weights on validation under safety constraints.
4. Train and save the selected policy, training curve, and search results.

The test and unseen test splits are never used for training.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from api.pipeline import Guardrail
from classifiers.threat_classifier import prep_row
from evaluation.features import (
    load_rows,
    oof_analyze,
    analyze_rows,
    rewrite_outcome,
    oof_supervised_actions,
)
from evaluation.metrics import evaluate
from policy.baselines import Hybrid
from policy.q_learning.agent import train, QDIR
from policy.q_learning.env import Lambdas
from policy.q_learning.state import N_CLUSTERS
from policy.supervised.model import SupervisedPolicy
from configs.settings import settings


def timed(label, fn, *args, **kwargs):
    """Run a function and report its elapsed time."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start

    print(
        f"[TIME] {label}: {elapsed:.2f}s",
        flush=True,
    )
    return result


def build_rewrite_cache(guardrail, records):
    """Compute each record's rewrite outcome once."""
    cache = {}

    start = time.perf_counter()
    total = len(records)

    for i, rec in enumerate(records, start=1):
        key = id(rec)
        cache[key] = rewrite_outcome(guardrail, rec)

        if i % 100 == 0 or i == total:
            elapsed = time.perf_counter() - start

            print(
                f"Rewrite outcomes: {i}/{total} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    print(
        f"[TIME] Rewrite cache: "
        f"{time.perf_counter() - start:.2f}s",
        flush=True,
    )

    def cached_rewrite(rec):
        key = id(rec)

        if key not in cache:
            raise KeyError(
                "Rewrite cache received an unknown record."
            )

        return cache[key]

    return cached_rewrite


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
    )
    parser.add_argument(
        "--max-violation",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    print(
        "[1/6] Loading guardrail and supervised policy...",
        flush=True,
    )

    guardrail = timed(
        "Guardrail initialization",
        Guardrail.get,
    )

    QDIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sup = timed(
        "Supervised policy loading",
        SupervisedPolicy.load,
    )

    assert sup is not None, (
        "Run scripts/train_supervised.py first."
    )

    print(
        "[2/6] Loading train and validation data...",
        flush=True,
    )

    train_rows = timed(
        "Training data loading",
        load_rows,
        "train",
    )

    val_rows = (
        load_rows("val")
        + load_rows("val_unseen")
    )

    print(
        f"Loaded {len(train_rows)} training rows "
        f"and {len(val_rows)} validation rows.",
        flush=True,
    )

    # ---------------------------------------------------------
    # Step 3: Fit semantic cluster centroids.
    # ---------------------------------------------------------

    from sklearn.cluster import KMeans

    print(
        "[3/6] Encoding training prompts for KMeans...",
        flush=True,
    )

    train_texts = [
        prep_row(record)
        for record in train_rows
    ]

    X = timed(
        "Training prompt embeddings",
        guardrail.enc.encode,
        train_texts,
    )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    print(
        f"Embeddings ready: shape={X.shape}. "
        "Fitting KMeans...",
        flush=True,
    )

    km = timed(
        "KMeans fitting",
        lambda: KMeans(
            n_clusters=N_CLUSTERS,
            n_init=5,
            random_state=0,
        ).fit(X),
    )

    centers = km.cluster_centers_

    norms = np.maximum(
        np.linalg.norm(
            centers,
            axis=1,
            keepdims=True,
        ),
        1e-9,
    )

    centroids = (
        centers / norms
    ).astype(np.float32)

    np.save(
        QDIR / "centroids.npy",
        centroids,
    )

    guardrail.centroids = centroids

    # ---------------------------------------------------------
    # Step 4: Build training and validation features.
    # ---------------------------------------------------------

    print(
        "[4/6] Building OOF training features...",
        flush=True,
    )

    tr = timed(
        "OOF training feature analysis",
        oof_analyze,
        train_rows,
        guardrail,
    )

    print(
        f"OOF training features ready: {len(tr)} rows.",
        flush=True,
    )

    print(
        "[5/6] Analyzing validation features...",
        flush=True,
    )

    va = timed(
        "Validation feature analysis",
        analyze_rows,
        val_rows,
        guardrail,
    )

    print(
        f"Validation features ready: {len(va)} rows.",
        flush=True,
    )

    # ---------------------------------------------------------
    # Step 5: Precompute rewrite outcomes for training records.
    # Validation records are not sent through this cache because
    # the original script discarded those computed outcomes.
    # ---------------------------------------------------------

    rw = build_rewrite_cache(
        guardrail,
        tr,
    )

    sup_acts = timed(
        "OOF supervised actions",
        oof_supervised_actions,
        tr,
    )

    # ---------------------------------------------------------
    # Evaluate the supervised hybrid baseline.
    # ---------------------------------------------------------

    print(
        "Evaluating supervised hybrid baseline...",
        flush=True,
    )

    base_actions = [
        Hybrid(
            None,
            "hybrid_sup",
            sup,
        ).act(record)
        for record in va
    ]

    base_val = timed(
        "Baseline validation evaluation",
        evaluate,
        guardrail,
        va,
        base_actions,
        "hybrid_sup_baseline",
    )

    baseline_keys = (
        "decision_accuracy",
        "security_recall",
        "false_positive_rate",
        "safety_constraint_violation_rate",
    )

    print(
        "Baseline hybrid on validation:",
        {
            key: base_val[key]
            for key in baseline_keys
        },
        flush=True,
    )

    # ---------------------------------------------------------
    # Search lambda combinations.
    # ---------------------------------------------------------

    print(
        "Searching reward weights...",
        flush=True,
    )

    grid = [
        Lambdas(
            fn=fn,
            fp=fp,
            rw=rw_weight,
        )
        for fn, fp, rw_weight in itertools.product(
            (5, 10, 20),
            (1.5, 3, 6),
            (1.0,),
        )
    ]

    margins = (
        0.0,
        0.3,
        0.8,
        1.5,
    )

    if args.quick:
        grid = grid[::4]

    results = []
    search_start = time.perf_counter()

    for index, lam in enumerate(grid, start=1):
        print(
            f"[SEARCH {index}/{len(grid)}] "
            f"Training with lambdas={lam.as_dict()}",
            flush=True,
        )

        pol, curve = timed(
            f"Q-policy training {index}",
            train,
            tr,
            rw,
            lam,
            epochs=args.epochs,
            sup=sup,
            sup_actions=sup_acts,
        )

        # Reuse the trained policy across margins.
        for margin in margins:
            pol.margin = margin

            actions = [
                Hybrid(
                    pol,
                    "q",
                    sup,
                ).act(record)
                for record in va
            ]

            metrics = evaluate(
                guardrail,
                va,
                actions,
                "q",
            )

            result = {
                "lambdas": lam.as_dict(),
                "margin": margin,
                "decision_accuracy": metrics[
                    "decision_accuracy"
                ],
                "security_recall": metrics[
                    "security_recall"
                ],
                "violation_rate": metrics[
                    "safety_constraint_violation_rate"
                ],
                "false_positive_rate": metrics[
                    "false_positive_rate"
                ],
                "unnecessary_block_rate": metrics[
                    "unnecessary_block_rate"
                ],
                "legit_utility": metrics[
                    "legit_utility"
                ],
                "final_train_reward": (
                    curve[-1] if curve else None
                ),
            }

            results.append(result)

            print(
                f"  margin={margin} "
                f"acc={result['decision_accuracy']} "
                f"viol={result['violation_rate']} "
                f"fpr={result['false_positive_rate']}",
                flush=True,
            )

    print(
        f"[TIME] Complete lambda search: "
        f"{time.perf_counter() - search_start:.2f}s",
        flush=True,
    )

    # ---------------------------------------------------------
    # Select a candidate against the baseline constraints.
    # ---------------------------------------------------------

    baseline_violation = (
        base_val["safety_constraint_violation_rate"]
    )

    baseline_fpr = (
        base_val["false_positive_rate"] or 0.0
    )

    baseline_accuracy = (
        base_val["decision_accuracy"]
    )

    feasible = [
        result
        for result in results
        if (
            result["violation_rate"]
            <= max(
                baseline_violation,
                args.max_violation,
            )
            and (
                result["false_positive_rate"] or 0.0
            )
            <= baseline_fpr + 0.005
        )
    ]

    if feasible:
        best = max(
            feasible,
            key=lambda result: (
                result["decision_accuracy"],
                -result["violation_rate"],
                result["legit_utility"] or 0,
            ),
        )
    else:
        best = max(
            results,
            key=lambda result: (
                result["decision_accuracy"],
                -result["violation_rate"],
            ),
        )

        print(
            "WARNING: No Q configuration meets the "
            "baseline safety constraints. The selected "
            "candidate will not be recommended.",
            flush=True,
        )

    improves = (
        bool(feasible)
        and (
            best["decision_accuracy"] > baseline_accuracy
            or best["violation_rate"] < baseline_violation
        )
        and best["decision_accuracy"]
        >= baseline_accuracy - 0.002
    )

    print(
        "Selected candidate:",
        best,
        flush=True,
    )

    # ---------------------------------------------------------
    # Train the final policy using the selected configuration.
    # ---------------------------------------------------------

    print(
        "[6/6] Training final policy...",
        flush=True,
    )

    selected_lambdas = Lambdas(
        **best["lambdas"]
    )

    final_policy, final_curve = timed(
        "Final Q-policy training",
        train,
        tr,
        rw,
        selected_lambdas,
        epochs=args.epochs * 2,
        log_every=10,
        sup=sup,
        sup_actions=sup_acts,
    )

    final_policy.margin = best["margin"]

    final_policy.meta.update(
        {
            "recommended": improves,
            "baseline_val_accuracy": baseline_accuracy,
            "baseline_val_violation": baseline_violation,
            "baseline_val_fpr": baseline_fpr,
            "selected_on": "validation",
            "constraint_max_violation": args.max_violation,
            "encoder": guardrail.enc.kind,
        }
    )

    final_policy.save()

    settings.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    search_results = {
        "constraint_max_violation": args.max_violation,
        "selected": best,
        "baseline_hybrid_val": base_val,
        "all": results,
    }

    (
        settings.results_dir / "q_lambda_search.json"
    ).write_text(
        json.dumps(
            search_results,
            indent=2,
        )
    )

    training_curve = {
        "epoch_mean_reward": final_curve,
        "lambdas": selected_lambdas.as_dict(),
    }

    (
        settings.results_dir / "q_training_curve.json"
    ).write_text(
        json.dumps(
            training_curve,
            indent=2,
        )
    )

    print(
        "Selected lambdas:",
        selected_lambdas.as_dict(),
        "->",
        QDIR,
        flush=True,
    )

    print(
        f"Policy recommended: {improves}",
        flush=True,
    )


if __name__ == "__main__":
    main()
