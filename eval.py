#!/usr/bin/env python
# policy_inference.py
# -- Inference & evaluation with multi-label mapping (prediction-rewrite style)

import setGPU            # optional: pins GPU usage
import os, time, argparse
import pandas as pd
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_recall_fscore_support

# ────────────────────────── CLI ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Inference for policy classifier")
parser.add_argument(
    "--batch-size", "-b",
    type=int,
    default=16,
    help="Batch size for inference"
)
args = parser.parse_args()

MODEL_DIR = "policy_classifier_lora"

# ─────────────────── Multi-label category→allowed-labels map ──────────────────
with open("multi_labels.json", "r") as f:
    mapping = json.load(f)
# Normalize keys to lower-case once
mapping = {k.lower(): v for k, v in mapping.items()}

# ─────────────────── Inference configuration ────────────────────────────────
BATCH_SIZE   = args.batch_size
MAX_GEN_TOKS = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.padding_side    = "left"
tokenizer.truncation_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto"
).eval()

# ─────────────────── Policy metadata & system prompt ─────────────────────────
doc_info = {
    "alan":    {"path": "policy/financial/alan.csv"},
    "bis":     {"path": "policy/financial/bis.csv"},
    "finra":   {"path": "policy/financial/finra.csv"},
    "oecd":    {"path": "policy/financial/oecd.csv"},
    "treasury":{"path": "policy/financial/treasury.csv"},
}
for info in doc_info.values():
    df = pd.read_csv(info["path"])
    info["df"]         = df
    info["categories"] = df["category_name"].unique().tolist()

system_prompt = (
    "You are a policy-compliance classifier.\n"
    "Identify which document index (1–5) a user request violates, or 'None' if none.\n\n"
    "Documents:\n"
)
for idx, (abbr, info) in enumerate(doc_info.items(), start=1):
    system_prompt += f"{idx}: {abbr}: {', '.join(info['categories'])}\n"

# ─────────────────── Build test set (examples 4 & 5) ─────────────────────────
tests = []  # (abbr, cat, true_label, text)
for idx_doc, (abbr, info) in enumerate(doc_info.items(), start=1):
    df = info["df"]
    for _, row in df.iterrows():
        cat = str(row["category_name"]).strip()
        # 4 and 5 are for evaluation
        for ex in [4, 5]:
            # malicious
            tests.append((abbr, cat, str(idx_doc), str(row[f"malicious_example_{ex}"])))
            # benign
            tests.append((abbr, None, "None",      str(row[f"benign_example_{ex}"])))

print(f"Total test requests: {len(tests)}")

# ─────────────────── Prepare prompts ───────────────────────────────────────────
prompts, docs, cats, true_labels = [], [], [], []
for abbr, cat, label, text in tests:
    conv = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text}
    ]
    prompts.append(
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True
        )
    )
    docs.append(abbr)
    cats.append(cat)
    true_labels.append(label)

# ─────────────────── Inference ───────────────────────────────────────────────
preds = []
device = model.device
start = time.time()
for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Inference"):
    batch_texts = prompts[i:i+BATCH_SIZE]
    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **enc,
            max_new_tokens=MAX_GEN_TOKS,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0
        )
    for out in out_ids:
        gen = out[enc.input_ids.shape[1]:]
        preds.append(tokenizer.decode(gen, skip_special_tokens=True).strip())

elapsed = time.time() - start
print(f"\nLatency: {len(prompts)/elapsed:.2f} queries/sec")

# ─────────────────── Prediction-rewrite ────────────────────────────────────
def rewrite_pred(pred, cat, gold):
    if cat is None or pd.isna(cat):
        return pred  # benign: leave untouched
    allowed = mapping.get(cat.lower(), [])
    if not allowed:
        print(f"Warning: no mapping for category '{cat}'")
    return gold if pred in allowed else pred

adj_preds = [
    rewrite_pred(p, c, t)
    for p, c, t in zip(preds, cats, true_labels)
]

# ─────────────────── Compute metrics ───────────────────────────────────────
results = pd.DataFrame({
    "doc":        docs,
    "category":   cats,
    "true_label": true_labels,
    "prediction": preds,
    "adj_pred":   adj_preds
})

all_tp = all_fp = all_fn = all_tn = 0
for idx_doc, abbr in enumerate(doc_info.keys(), start=1):
    sub = results[results.doc == abbr]
    y_true = sub.true_label == str(idx_doc)
    y_pred = sub.adj_pred   == str(idx_doc)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    fp = ((~y_true) & y_pred).sum()
    tn = ((~y_true) & (~y_pred)).sum()
    fpr = fp / (fp + tn) if fp + tn else 0.0

    tp = (y_true & y_pred).sum()
    fn = (y_true & ~y_pred).sum()
    all_tp += tp; all_fp += fp; all_fn += fn; all_tn += tn

overall_precision = (
    all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
)
overall_recall = (
    all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
)
overall_f1 = (
    2 * overall_precision * overall_recall /
    (overall_precision + overall_recall)
    if overall_precision + overall_recall else 0.0
)
overall_fpr = all_fp / (all_fp + all_tn) if all_fp + all_tn else 0.0

print(f"\nOverall Precision: {overall_precision:.2%}")
print(f"Overall Recall   : {overall_recall:.2%}")
print(f"Overall F1       : {overall_f1:.2%}")
print(f"Overall FPR      : {overall_fpr:.2%}")
