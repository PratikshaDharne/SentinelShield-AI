"""Train the threat classifier + build the MiniLM/FAISS retrieval index. Uses ONLY the train split
for fitting and the val split for reporting; test/unseen_test are touched only by evaluation scripts.
  python scripts/train_classifier.py --backbone tfidf     (CPU-light default)
  python scripts/train_classifier.py --backbone minilm    (better; needs sentence-transformers)"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.normalizer import normalize
from classifiers.threat_classifier import ThreatClassifier
from retrieval.index import RiskIndex
from embeddings.encoder import Encoder
from configs.settings import settings

D = settings.datasets_dir / "data"


def load(sp): return [json.loads(l) for l in (D / f"{sp}.jsonl").open(encoding="utf-8")]


from classifiers.threat_classifier import prep_row as prep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", default="tfidf"); a = ap.parse_args()
    from sklearn.metrics import classification_report
    tr, va = load("train"), load("val")
    clf = ThreatClassifier(a.backbone).fit([prep(r) for r in tr], [r["category"] for r in tr])
    pred = [clf.predict(prep(r))[0] for r in va]
    rep = classification_report([r["category"] for r in va], pred, output_dict=True, zero_division=0)
    print(classification_report([r["category"] for r in va], pred, zero_division=0))
    clf.save()
    idx = RiskIndex(Encoder.get()).build([prep(r) for r in tr], [r["category"] for r in tr]); idx.save()
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "classifier_val_report.json").write_text(json.dumps({"backbone": a.backbone, "encoder": idx.enc.kind, "report": rep}, indent=2))
    import datetime
    (settings.models_dir / "MODEL_INFO.json").write_text(json.dumps({"encoder": idx.enc.kind, "backbone": a.backbone, "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
                                                                  "train_rows": len(tr)}, indent=2))
    print("saved models/threat_clf.joblib and models/retrieval/")


if __name__ == "__main__":
    main()
