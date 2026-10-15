"""Fine-tune Flan-T5 as OUR rewrite model (run once; CPU works for flan-t5-small, use a GPU/Colab for -base).
  python scripts/build_rewrite_dataset.py
  python scripts/train_rewriter.py --base google/flan-t5-small --epochs 4
Output: models/rewriter_t5/  (auto-loaded by the API when present; the validator still gates every output)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.settings import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/flan-t5-small"); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
    D = settings.datasets_dir
    load = lambda n: [json.loads(l) for l in (D / f"rewrite_pairs_{n}.jsonl").open(encoding="utf-8") if l.strip()]
    tr, va = load("train"), load("val")
    tok = AutoTokenizer.from_pretrained(a.base); model = AutoModelForSeq2SeqLM.from_pretrained(a.base)

    def enc(b):
        x = tok(b["input"], max_length=256, truncation=True)
        x["labels"] = tok(text_target=b["target"], max_length=160, truncation=True)["input_ids"]
        return x
    dtr = Dataset.from_list(tr).map(enc, batched=True, remove_columns=["input", "target"])
    dva = Dataset.from_list(va[:500]).map(enc, batched=True, remove_columns=["input", "target"])
    args = Seq2SeqTrainingArguments(output_dir=str(settings.models_dir / "_t5_runs"), per_device_train_batch_size=a.bs,
                                    per_device_eval_batch_size=a.bs, learning_rate=a.lr, num_train_epochs=a.epochs,
                                    eval_strategy="epoch", save_strategy="no", logging_steps=50, report_to=[],
                                    fp16=torch.cuda.is_available(), predict_with_generate=False)
    Seq2SeqTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva, data_collator=DataCollatorForSeq2Seq(tok, model=model)).train()
    out = settings.models_dir / "rewriter_t5"; model.save_pretrained(out); tok.save_pretrained(out)
    print("saved", out)
    # quick sanity: share of val rewrites that pass OUR validator
    from rewriting.generator import T5Rewriter
    from validation.validator import Validator
    t5, v, ok = T5Rewriter(out), Validator(), 0
    sample = va[:100]
    for r in sample:
        gen = t5.tok.decode(t5.model.generate(**t5.tok(r["input"], return_tensors="pt"), max_new_tokens=128, num_beams=4)[0], skip_special_tokens=True)
        task = r["input"].split("task: ", 1)[1].split(" | ", 1)[0]
        ok += v.validate(gen, task).passed
    print(f"validator pass-rate on {len(sample)} val rewrites: {ok / len(sample):.2%}")


if __name__ == "__main__":
    main()
