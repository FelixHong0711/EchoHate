#!/usr/bin/env python3
"""
Post-training final evaluation for mechanism-only EchoHate encoders.

Protocol (baseline-style, no leakage):
  1) Load saved ``encoder_model.pt``
  2) Tune threshold on validation (optional; default on)
  3) Evaluate once on test with frozen threshold
  4) Save predictions + metrics JSON in run_dir
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import AutoTokenizer

from pipeline.common.paths import DEFAULT_DATA_ROOT, SPLITTED_DATA_SUBDIR
from pipeline.stage2.mech_distill import (
    MechEncoder,
    DistillDataset,
    _collate_distill,
    checkpoint_mechanism_compatible,
)
from pipeline.stage2.roberta_baseline import (
    load_splits_baseline,
    selection_f1_from_metrics,
    tune_threshold_f1,
)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"failed to write non-empty file: {path}")


def _verify_non_empty_files(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"final eval outputs missing or empty: {missing}")


def _metrics_from_preds(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f1_micro": float(f1_micro),
        "precision_micro": float(prec_micro),
        "recall_micro": float(rec_micro),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "auc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _predict_probs_mech(
    run_dir: Path,
    model_name: str,
    tokenizer: Any,
    df: pd.DataFrame,
    max_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    ds = DistillDataset(
        texts=df["text"].fillna("").astype(str).tolist(),
        labels=df["binary_label"].astype(int).tolist(),
        mech_labels=[-100] * len(df),
        has_mech_list=[0] * len(df),
        mech_weight_list=[0.0] * len(df),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    model = MechEncoder(model_name, use_mechanism_head=False).to(device)
    ckpt = run_dir / "encoder_model.pt"
    if not ckpt.is_file():
        raise SystemExit(f"Missing mechanism checkpoint: {ckpt}")
    sd = torch.load(ckpt, map_location="cpu")
    cfg_path = run_dir / "encoder_distill_config.json"
    mech_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    if not mech_cfg and (run_dir / "metric_summary.json").is_file():
        mech_cfg = json.loads((run_dir / "metric_summary.json").read_text(encoding="utf-8"))
    ok, msg = checkpoint_mechanism_compatible(cfg=mech_cfg, state_dict=sd, expect_mech_head=False)
    if not ok:
        raise SystemExit(f"Stale mechanism checkpoint: {msg}")
    model.load_state_dict(sd, strict=False)
    model.eval()

    probs_all: List[float] = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = _collate_distill([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_all.extend(np.atleast_1d(probs).tolist())
    return np.asarray(probs_all, dtype=np.float64)


def _load_model_name(run_dir: Path, default_name: str) -> str:
    ms = run_dir / "metric_summary.json"
    if ms.is_file():
        j = json.loads(ms.read_text(encoding="utf-8"))
        tr = j.get("training_config", {}) or {}
        mi = j.get("model_info", {}) or {}
        return str(tr.get("model_name") or mi.get("model_name") or default_name)
    return default_name


def _resolve_splitted_subdir(
    run_dir: Path, cli_override: Optional[str], dataset_name: str
) -> Tuple[str, str]:
    if cli_override and str(cli_override).strip():
        return str(cli_override).strip(), "cli"
    ms = run_dir / "metric_summary.json"
    if ms.is_file():
        j = json.loads(ms.read_text(encoding="utf-8"))
        di = j.get("dataset_info") or {}
        s = di.get("splitted_data_subdir")
        if s:
            return str(s).strip(), "metric_summary.json:dataset_info.splitted_data_subdir"
        if str(dataset_name) == "IHC":
            n_train = di.get("train_combined_samples")
            if n_train == 17180:
                return (
                    "splitted_data_80_10_10",
                    "inferred:IHC train_combined_samples=17180 -> exp+imp split",
                )
            if n_train == 16311:
                return (
                    "splitted_data_80_10_10",
                    "inferred:IHC train_combined_samples=16311 -> implicit-only split",
                )
    return SPLITTED_DATA_SUBDIR, "echohate_split_layout(default/env)"


def _resolve_random_state(run_dir: Path) -> Tuple[Optional[int], str]:
    ms = run_dir / "metric_summary.json"
    if ms.is_file():
        try:
            j = json.loads(ms.read_text(encoding="utf-8"))
            tr = j.get("training_config") or {}
            rs = tr.get("random_state")
            if rs is not None:
                return int(rs), "metric_summary.json:training_config.random_state"
        except Exception:
            pass
    return None, "unavailable"


def main() -> None:
    p = argparse.ArgumentParser(description="Final one-time test evaluation (mechanism-only).")
    p.add_argument(
        "--approach",
        default="mech",
        choices=["mech"],
        help="Only mechanism distill checkpoints are supported.",
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--run_dir", required=True, help="Training run directory with encoder_model.pt + tokenizer/")
    p.add_argument(
        "--data_root",
        default=None,
        help="Default: DEFAULT_DATA_ROOT (data/<ECHOHATE_DATA_SEED_DIR>/).",
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tune_split", choices=["validation", "none"], default="validation")
    p.add_argument("--fixed_threshold", type=float, default=None, help="If set, skip tuning")
    p.add_argument("--output_prefix", default="final_test", help="Output prefix for csv/json files")
    p.add_argument(
        "--overwrite_outputs",
        action="store_true",
        help="Allow overwriting existing final-eval output files.",
    )
    p.add_argument(
        "--selection_metric",
        choices=["macro", "binary"],
        default="macro",
        help="Validation metric for τ sweep (macro=macro-F1; binary=positive-class F1).",
    )
    p.add_argument(
        "--splitted_data_subdir",
        default=None,
        help="Override split folder under data/<dataset>/ (e.g. splitted_data_80_10_10).",
    )
    args = p.parse_args()
    selection_metric = str(args.selection_metric).strip().lower()
    threshold_selection = f"{selection_metric}_f1"

    pipeline_started_at_utc = datetime.now(timezone.utc).isoformat()
    pipeline_t0 = time.perf_counter()

    data_root = (
        Path(args.data_root).expanduser().resolve()
        if args.data_root
        else Path(DEFAULT_DATA_ROOT).resolve()
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")
    print(f"Final eval data_root: {data_root}", flush=True)

    tok_dir = run_dir / "tokenizer"
    model_name = _load_model_name(run_dir, "roberta-large")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_dir) if tok_dir.is_dir() else model_name, use_fast=True
    )

    split_sub, split_src = _resolve_splitted_subdir(
        run_dir, args.splitted_data_subdir, args.dataset
    )
    random_state, random_state_source = _resolve_random_state(run_dir)
    print(f"Final eval random_state: {random_state} (source={random_state_source})")
    print(f"Final eval split layout: {split_sub} (source={split_src})")
    if split_src == "echohate_split_layout(default/env)":
        ms = run_dir / "metric_summary.json"
        if ms.is_file():
            j = json.loads(ms.read_text(encoding="utf-8"))
            if "splitted_data_subdir" not in (j.get("dataset_info") or {}):
                print(
                    "WARNING: metric_summary.json has no dataset_info.splitted_data_subdir; "
                    "pass --splitted_data_subdir to match the split used during training.",
                    flush=True,
                )

    train_df, validation_df, test_df = load_splits_baseline(data_root, args.dataset, split_sub)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_labels = train_df["binary_label"].astype(int).to_numpy()
    validation_labels = validation_df["binary_label"].astype(int).to_numpy()
    test_labels = test_df["binary_label"].astype(int).to_numpy()

    setup_wall_sec = float(time.perf_counter() - pipeline_t0)
    infer_t0 = time.perf_counter()

    train_probs = _predict_probs_mech(
        run_dir, model_name, tokenizer, train_df, args.max_length, args.batch_size, device
    )
    validation_probs = _predict_probs_mech(
        run_dir, model_name, tokenizer, validation_df, args.max_length, args.batch_size, device
    )
    test_probs = _predict_probs_mech(
        run_dir, model_name, tokenizer, test_df, args.max_length, args.batch_size, device
    )
    if args.fixed_threshold is not None:
        best_t = float(args.fixed_threshold)
    elif args.tune_split == "validation":
        best_t = tune_threshold_f1(validation_labels, validation_probs, selection_metric)
    else:
        best_t = 0.5
    validation_metrics = _metrics_from_preds(
        validation_labels, (validation_probs >= best_t).astype(int), validation_probs
    )

    posthoc_inference_wall_sec = float(time.perf_counter() - infer_t0)
    n_pred_samples = int(len(train_df) + len(validation_df) + len(test_df))
    pred_samples_per_sec = (
        (n_pred_samples / posthoc_inference_wall_sec) if posthoc_inference_wall_sec > 0 else None
    )
    print(
        f"⏱ Timing (wall clock): setup={setup_wall_sec:.1f}s  "
        f"posthoc_inference={posthoc_inference_wall_sec:.1f}s  "
        f"rows={n_pred_samples}  rows/s={pred_samples_per_sec:.1f}"
        if pred_samples_per_sec is not None
        else f"⏱ Timing (wall clock): setup={setup_wall_sec:.1f}s  posthoc_inference={posthoc_inference_wall_sec:.1f}s"
    )

    train_preds = (train_probs >= best_t).astype(int)
    test_preds = (test_probs >= best_t).astype(int)
    train_metrics = _metrics_from_preds(train_labels, train_preds, train_probs)
    test_metrics = _metrics_from_preds(test_labels, test_preds, test_probs)

    out_train_csv = run_dir / f"{args.output_prefix}_predictions_train.csv"
    out_validation_csv = run_dir / f"{args.output_prefix}_predictions_validation.csv"
    out_csv = run_dir / f"{args.output_prefix}_predictions_test.csv"
    out_json = run_dir / f"{args.output_prefix}_metric_summary_test.json"
    guarded_paths = [out_train_csv, out_validation_csv, out_csv, out_json]
    if not bool(args.overwrite_outputs):
        existed = [str(p) for p in guarded_paths if p.exists()]
        if existed:
            raise SystemExit(
                "Refusing to overwrite existing final-eval outputs. "
                "Use --output_prefix to create a new set, or pass --overwrite_outputs.\n"
                + "\n".join(existed)
            )

    train_pred_df = pd.DataFrame(
        {
            "row_id": train_df["row_id"].astype(str).tolist(),
            "text": train_df["text"].fillna("").astype(str).tolist(),
            "class": train_labels.tolist(),
            "prob": train_probs.tolist(),
            "pred": train_preds.tolist(),
            "threshold_used": [best_t] * len(train_df),
        }
    )
    train_pred_df.to_csv(out_train_csv, index=False)
    validation_preds = (validation_probs >= best_t).astype(int)
    validation_pred_df = pd.DataFrame(
        {
            "row_id": validation_df["row_id"].astype(str).tolist(),
            "text": validation_df["text"].fillna("").astype(str).tolist(),
            "class": validation_labels.tolist(),
            "prob": validation_probs.tolist(),
            "pred": validation_preds.tolist(),
            "threshold_used": [best_t] * len(validation_df),
        }
    )
    validation_pred_df.to_csv(out_validation_csv, index=False)
    pred_df = pd.DataFrame(
        {
            "row_id": test_df["row_id"].astype(str).tolist(),
            "text": test_df["text"].fillna("").astype(str).tolist(),
            "class": test_labels.tolist(),
            "prob": test_probs.tolist(),
            "pred": test_preds.tolist(),
            "threshold_used": [best_t] * len(test_df),
        }
    )
    pred_df.to_csv(out_csv, index=False)

    total_pipeline_wall_sec = float(time.perf_counter() - pipeline_t0)
    timing_detail: Dict[str, Any] = {
        "pipeline_started_at_utc": pipeline_started_at_utc,
        "setup_wall_sec": round(setup_wall_sec, 4),
        "posthoc_inference_wall_sec": round(posthoc_inference_wall_sec, 4),
        "total_pipeline_wall_sec": round(total_pipeline_wall_sec, 4),
        "posthoc_prediction_rows": n_pred_samples,
        "posthoc_prediction_rows_per_sec": (
            round(pred_samples_per_sec, 4) if pred_samples_per_sec is not None else None
        ),
        "notes": "Final-eval inference only (train+validation+test forward). Mech distill.",
    }

    summary = {
        "dataset": args.dataset,
        "approach": "mech",
        "run_dir": str(run_dir),
        "split_layout": split_sub,
        "split_layout_source": split_src,
        "random_state": random_state,
        "random_state_source": random_state_source,
        "timing": timing_detail,
        "threshold_policy": {
            "tune_split": args.tune_split,
            "fixed_threshold": (float(args.fixed_threshold) if args.fixed_threshold is not None else None),
            "sweep_min": 0.05,
            "sweep_max": 0.95,
            "sweep_step": 0.05,
            "selection_metric": selection_metric,
            "threshold_selection": threshold_selection,
            "best_threshold_validation": float(best_t),
            "applied_threshold_test": float(best_t),
        },
        "metrics_train_at_selected_threshold": train_metrics,
        "metrics_validation_at_selected_threshold": validation_metrics,
        "metrics_test": test_metrics,
        "n_train": int(len(train_df)),
        "n_validation": int(len(validation_df)),
        "n_test": int(len(test_df)),
        "output_predictions_train_csv": str(out_train_csv),
        "output_predictions_validation_csv": str(out_validation_csv),
        "output_predictions_test_csv": str(out_csv),
    }
    _atomic_write_text(out_json, json.dumps(summary, indent=2))
    _verify_non_empty_files(guarded_paths)
    val_sel = selection_f1_from_metrics(validation_metrics, selection_metric)
    print(
        f"Saved final eval: thr={best_t:.2f}  "
        f"val_{selection_metric}_f1={val_sel:.4f}  "
        f"val_f1_macro={float(validation_metrics['f1_macro']):.4f}  "
        f"test_f1_macro={float(test_metrics['f1_macro']):.4f}  → {out_json}"
    )


if __name__ == "__main__":
    main()
