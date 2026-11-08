#!/usr/bin/env python3
"""
EchoHate: standalone RoBERTa baseline (no llm/aux losses), using distill-style HF Trainer flow.

**Full fine-tune** — all RoBERTa encoder weights and the classification head are updated.

Default CLI hyperparameters are epochs 6, batch 16, learning rate 1e-5, weight decay 0.01, and maximum length 256.

Data (default): train on **train.csv**, tune on **validation.csv**, infer on **test.csv**
(unless ``--skip-test-infer``). Best epoch uses val τ tuned on grid 0.05–0.95 step 0.05
(by **macro** F1; Trainer metric = macro-F1 at that τ). Applied test/val predictions
use that τ. Train/test threshold curves are not written.

Optional ``--train-on-train-val``: train on **train.csv ∪ validation.csv**, no held-out val for
early-stopping (fixed epochs), infer on **test.csv**. Applied test/val F1 threshold is the
**val-tuned τ from the matching 80% train-only** ``roberta_large_baseline/metric_summary.json``
(``threshold_policy.best_threshold_validation``) — not retuned on the train∪val run, not fixed 0.5.

With ``--epochs-from-train-baseline``, set ``num_epochs`` to the best validation epoch from that
same train-only baseline ``metric_summary.json``.

Writes ``predictions_train.csv``, ``predictions_eval.csv`` (validation, for downstream compat),
and ``predictions_test.csv``.

Default output (no ``--output_dir``): ``<data_root>/<dataset>/roberta_large_baseline/`` for ``roberta-large``
(override folder name with env ``ECHOHATE_ROBERTA_BASELINE_DIR``). All checkpoints, metrics, and prediction CSVs
live in that single directory — not under nested ``lambda_* / seed_*`` subfolders.

Each run writes **two threshold variants** from the same checkpoint (no second training):
  - **Val-tuned τ** (grid 0.05–0.95 step 0.05 by macro F1 on validation): ``predictions_{train,eval,test}.csv``
  - **Fixed τ=0.5**: ``predictions_{train,eval,test}_tau050.csv``
Both variants appear in ``metric_summary.json`` / ``metrics.json``.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from pipeline.common.paths import (
    ECHOHATE_ROOT,
    SPLITTED_DATA_SUBDIR,
    get_data_root,
    resolve_baseline_run_dir,
)
from pipeline.common.project_config import default_random_state, set_seed
from pipeline.stage2.training_dynamics import TrainingDynamicsCallback


def _torchrun_local_rank() -> int:
    """``LOCAL_RANK`` from ``torchrun``; ``-1`` when launched with plain ``python``."""
    raw = os.environ.get("LOCAL_RANK", "")
    if raw is None or str(raw).strip() == "":
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def _is_rank0_for_writes() -> bool:
    """Rank 0 (or single process) writes predictions, metrics, and per-epoch dynamics files."""
    return _torchrun_local_rank() <= 0


class ContrastiveLossCosine(nn.Module):
    """
    Pairwise cosine contrastive loss on batch vectors (logit space).
    Training uses **logits** ``[B, 2]`` as embeddings, same as their ``train.py`` path.
    """

    def __init__(self, margin: float = 0.75):
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return embeddings.sum() * 0.0
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        labels = labels.unsqueeze(1)
        label_matrix = (labels != labels.T).float()
        positive_loss = (1 - label_matrix) * (1 - cosine_sim)
        negative_loss = label_matrix * F.relu(cosine_sim - self.margin)
        return (positive_loss + negative_loss).sum() / (batch_size * (batch_size - 1))


class RoBERTaBinaryClassifier(nn.Module):
    """
    Same pattern as ``custom_model_tackling_hate.LLMClassifier``: ``AutoModel`` backbone, CLS pooling,
    classification head, and **loss computed in ``forward``** when ``labels`` is provided.
    Extra batch keys (e.g. ``row_ids``) are absorbed via ``**kwargs``.
    """

    def __init__(
        self,
        model_name: str = "roberta-large",
        loss_type: str = "ce",
        ce_contrastive_lambda: Optional[float] = None,
        contrastive_margin: float = 0.5,
    ):
        super().__init__()
        self.loss_type = (loss_type or "ce").strip().lower()
        if self.loss_type not in ("ce", "bce"):
            raise ValueError(f"loss_type must be 'ce' or 'bce', got {loss_type!r}")
        self.model_name = model_name
        self.ce_contrastive_lambda: Optional[float] = None
        self._contrastive_loss: Optional[ContrastiveLossCosine] = None
        if self.loss_type == "ce" and ce_contrastive_lambda is not None:
            lam = float(ce_contrastive_lambda)
            if not (0.0 < lam <= 1.0):
                raise ValueError(
                    f"ce_contrastive_lambda must be in (0, 1], got {lam}. "
                    "Omit the flag (None) for CE-only; λ=0 is pure contrastive, not 'off'."
                )
            self.ce_contrastive_lambda = lam
            self._contrastive_loss = ContrastiveLossCosine(margin=float(contrastive_margin))
        try:
            self.backbone = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        except TypeError:
            self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.config = self.backbone.config
        hidden_size = int(self.config.hidden_size)
        self.class_weight_target: str = "pos"
        if self.loss_type == "ce":
            self.num_labels = 2
            self.classifier = nn.Linear(hidden_size, 2)
            # Unweighted CE, same as custom_model_tackling_hate.LLMClassifier (no class weights)
            self._ce_loss_unweighted = nn.CrossEntropyLoss()
        else:
            self.num_labels = 1
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state
        pooled = hidden[:, 0, :]
        logits = self.classifier(pooled)

        if labels is None:
            return {"logits": logits}

        if self.loss_type == "ce":
            labels_long = labels.long()
            w = getattr(self, "ce_class_weight", None)
            if w is not None:
                w = w.to(device=logits.device, dtype=logits.dtype)
            sw = kwargs.get("ce_weight")
            if sw is not None:
                sw = sw.to(device=logits.device, dtype=logits.dtype).view(-1)
                ce_per = nn.functional.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels_long.view(-1),
                    weight=w,
                    reduction="none",
                )
                ce_loss = (ce_per * sw).mean()
            elif w is None:
                ce_loss = self._ce_loss_unweighted(
                    logits.view(-1, self.num_labels),
                    labels_long.view(-1),
                )
            else:
                ce_loss = nn.functional.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels_long.view(-1),
                    weight=w,
                )
            if self._contrastive_loss is not None and self.ce_contrastive_lambda is not None:
                lam = float(self.ce_contrastive_lambda)
                ctr = self._contrastive_loss(logits.float(), labels_long.view(-1))
                loss = lam * ce_loss + (1.0 - lam) * ctr
            else:
                loss = ce_loss
            return {"loss": loss, "logits": logits}

        labels_f = labels.float()
        log = logits.squeeze(-1)
        pw = getattr(self, "bce_pos_weight", None)
        if pw is not None:
            pw = pw.to(device=logits.device, dtype=logits.dtype)
        ct = (getattr(self, "class_weight_target", "pos") or "pos").strip().lower()
        if pw is None:
            loss = nn.functional.binary_cross_entropy_with_logits(log, labels_f)
        elif ct == "neg":
            loss = nn.functional.binary_cross_entropy_with_logits(
                -log, 1.0 - labels_f, pos_weight=pw
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(log, labels_f, pos_weight=pw)
        return {"loss": loss, "logits": logits}


class BinaryDataCollator:
    def __init__(self, tokenizer: Any, max_length: int = 256, fixed_max_length: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Mimic distill distill-style behavior (fixed padding to max_length).
        self.fixed_max_length = fixed_max_length

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        row_ids = [f.get("row_id", "") for f in features]
        has_ce_w = any("ce_weight" in f for f in features)
        input_ids = []
        attention_masks = []
        for f in features:
            ids = f["input_ids"][: self.max_length]
            mask = f["attention_mask"][: self.max_length]
            if self.fixed_max_length and len(ids) < self.max_length:
                pad_len = self.max_length - len(ids)
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = 1  # fallback; should not happen for RoBERTa
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len
            input_ids.append(ids)
            attention_masks.append(mask)
        if self.fixed_max_length:
            batch = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        else:
            batch = self.tokenizer.pad(
                {"input_ids": input_ids, "attention_mask": attention_masks},
                padding=True,
                return_tensors="pt",
            )
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        batch["row_ids"] = row_ids
        if has_ce_w:
            batch["ce_weight"] = torch.tensor(
                [float(f.get("ce_weight", 1.0)) for f in features],
                dtype=torch.float32,
            )
        return batch


def _compute_pos_weight(labels: np.ndarray, mode: str, target: str = "pos") -> Optional[float]:
    m = (mode or "freq").strip().lower()
    if m == "none":
        return None
    t = (target or "pos").strip().lower()
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if t == "neg":
        if n_neg <= 0:
            return None
        w = n_pos / n_neg
    else:
        if n_pos <= 0:
            return None
        w = n_neg / n_pos
    if m == "sqrt":
        w = float(np.sqrt(w))
    return max(w, 1e-6)


def _compute_ce_class_weights(labels: np.ndarray, mode: str) -> Optional[torch.Tensor]:
    """
    Balanced class weights for CrossEntropyLoss: weight[c] = N / (2 * n_c).
    ``pos_weight_mode`` none → unweighted CE; freq / sqrt → weighted (sqrt dampens extreme ratios).
    """
    m = (mode or "freq").strip().lower()
    if m == "none":
        return None
    n = float(len(labels))
    n0 = max(float((labels == 0).sum()), 1.0)
    n1 = max(float((labels == 1).sum()), 1.0)
    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)
    if m == "sqrt":
        w0 = float(np.sqrt(w0))
        w1 = float(np.sqrt(w1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def _oversample_minority_to_balance(df: pd.DataFrame, label_col: str, seed: int) -> pd.DataFrame:
    """
    Duplicate minority-class rows with replacement until both classes have equal count.
    Shuffles the resulting dataframe with ``seed`` for reproducibility.
    """
    y = df[label_col].astype(int).values
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    if len(idx0) == 0 or len(idx1) == 0:
        return df
    n0, n1 = len(idx0), len(idx1)
    rng = np.random.RandomState(int(seed))
    if n0 == n1:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if n0 < n1:
        need = n1 - n0
        extra_idx = rng.choice(idx0, size=need, replace=True)
    else:
        need = n0 - n1
        extra_idx = rng.choice(idx1, size=need, replace=True)
    extra = df.iloc[extra_idx].copy()
    out = pd.concat([df, extra], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _extract_trainer_eval_logits(predictions: Any) -> np.ndarray:
    """
    HF Trainer may pass logits alone, or a tuple of tensors when the model forward
    returns multiple keys (e.g. cue + mechanism head: logits, mech_logits, ...).
    """
    pred_arr = predictions
    if isinstance(pred_arr, (tuple, list)) and len(pred_arr) > 0:
        pred_arr = pred_arr[0]
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    if pred_arr.ndim == 2 and pred_arr.shape[-1] == 2:
        return pred_arr
    if pred_arr.dtype == object and pred_arr.ndim >= 1:
        for i in range(int(pred_arr.shape[0])):
            cand = np.asarray(pred_arr[i], dtype=np.float64)
            if cand.ndim == 2 and cand.shape[-1] == 2:
                return cand
    return pred_arr


def _trainer_eval_probs_and_labels(eval_pred: EvalPrediction) -> Tuple[np.ndarray, np.ndarray]:
    """Extract positive-class probabilities and int labels from a Trainer EvalPrediction."""
    pred_arr = _extract_trainer_eval_logits(eval_pred.predictions)
    labels = np.asarray(eval_pred.label_ids, dtype=np.float64).ravel()
    y = labels.astype(int)
    if pred_arr.ndim == 2 and pred_arr.shape[-1] == 2:
        row_max = np.max(pred_arr, axis=-1, keepdims=True)
        expv = np.exp(pred_arr - row_max)
        probs = (expv[:, 1] / np.sum(expv, axis=-1)).astype(np.float64)
    else:
        logits = np.squeeze(pred_arr, axis=-1) if pred_arr.ndim > 1 else pred_arr
        probs = 1.0 / (1.0 + np.exp(-logits))
    return probs.astype(np.float64), y


FIXED_PREDICTION_TAU = 0.5


def _apply_tau_preds(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    out = df.copy()
    out["pred"] = (out["prob"].astype(float).to_numpy() >= float(tau)).astype(int)
    out["correct"] = (out["pred"].astype(int) == out["label"].astype(int)).astype(int)
    out["threshold_used"] = float(tau)
    return out


def _metrics_at_tau(labels: np.ndarray, probs: np.ndarray, tau: float) -> Dict[str, Any]:
    m = _metrics_from_preds(labels, (probs >= float(tau)).astype(int), probs)
    m["n_samples"] = int(len(labels))
    m["threshold_used"] = float(tau)
    return m


def _write_prediction_variant(
    raw_df: pd.DataFrame,
    *,
    tau: float,
    out_dir: Path,
    stem: str,
) -> pd.DataFrame:
    applied = _apply_tau_preds(raw_df, tau)
    applied.to_csv(out_dir / f"{stem}.csv", index=False)
    return applied


def tune_threshold_f1(labels: np.ndarray, probs: np.ndarray, metric: str = "macro") -> float:
    """Decision τ on 0.05…0.95 step 0.05, chosen by macro or binary F1."""
    from sklearn.metrics import f1_score as _f1_score

    m = str(metric).strip().lower()
    if m not in ("macro", "binary"):
        raise ValueError(f"tune_threshold_f1 metric must be macro|binary, got {metric!r}")
    average = "macro" if m == "macro" else "binary"
    y = np.asarray(labels, dtype=int).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    grid = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)]
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = float(_f1_score(y, (p >= t).astype(int), average=average, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def selection_f1_from_metrics(metrics: Dict[str, Any], metric: str = "macro") -> float:
    """Score used to pick τ / epoch / λ on validation."""
    m = str(metric).strip().lower()
    if m == "binary":
        return float(metrics.get("f1", metrics.get("f1_binary", 0.0)))
    return float(metrics["f1_macro"])


def _tune_threshold_macro_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    return tune_threshold_f1(labels, probs, "macro")


def _tune_threshold_binary_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    return tune_threshold_f1(labels, probs, "binary")


def _compute_metrics_for_trainer(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Metrics for Hugging Face Trainer eval on the validation split (τ = 0.5 / argmax)."""
    probs, y = _trainer_eval_probs_and_labels(eval_pred)
    preds = (probs >= 0.5).astype(int)
    m = _metrics_from_preds(y, preds, probs)
    # Keys aligned with custom_model_tackling_hate.compute_metrics (HF prefixes eval_)
    return {
        "accuracy": float(m["accuracy"]),
        "f1_macro": float(m["f1_macro"]),
        "precision_macro": float(m["precision_macro"]),
        "recall_macro": float(m["recall_macro"]),
        "auc_macro": float(m["auc"]),
    }


def _compute_metrics_for_trainer_best_threshold(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Trainer eval: τ sweep on val; score epoch by macro-F1 at that τ.

    - decision threshold grid **0.05 to 0.95 in steps of 0.05**
    - choose τ by **macro** F1 on that grid
    - report **macro** F1 at that τ for ``metric_for_best_model``
    """
    probs, y = _trainer_eval_probs_and_labels(eval_pred)
    best_t = _tune_threshold_macro_f1(y, probs)
    preds = (probs >= best_t).astype(int)
    m = _metrics_from_preds(y, preds, probs)
    return {
        "accuracy": float(m["accuracy"]),
        "f1_macro": float(m["f1_macro"]),
        "precision_macro": float(m["precision_macro"]),
        "recall_macro": float(m["recall_macro"]),
        "auc_macro": float(m["auc"]),
        "best_threshold": float(best_t),
    }


def _metrics_from_preds(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
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
        "f1_macro": float(f1_macro),
        "precision_micro": float(prec_micro),
        "recall_micro": float(rec_micro),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "auc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _sweep_thresholds_macro_f1(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, Any], List[Dict[str, float]]]:
    thresholds = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    best_t = 0.5
    best_metrics: Optional[Dict[str, Any]] = None
    curve: List[Dict[str, float]] = []
    best_f1 = -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        m = _metrics_from_preds(labels, preds, probs)
        f1m = float(m["f1_macro"])
        curve.append({"threshold": float(t), "f1_macro": f1m})
        if f1m > best_f1:
            best_f1 = f1m
            best_t = float(t)
            best_metrics = m
    return best_t, (best_metrics or _metrics_from_preds(labels, (probs >= 0.5).astype(int), probs)), curve


def tokenize_texts_like_custom_model(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 256,
) -> Dict[str, Any]:
    """
    Match ``custom_model_tackling_hate.tokenize`` + ``Dataset.map(..., batched=True)``:
    ``tokenizer(..., padding='max_length', truncation=True, max_length=max_length)``.
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )


def prepare_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    text_col: str = "text",
    max_length: int = 256,
    pad_to_max_length: bool = False,
) -> Dataset:
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df["binary_label"].astype(float).tolist()
    row_ids = df["row_id"].tolist()
    if pad_to_max_length:
        encodings = tokenize_texts_like_custom_model(texts, tokenizer, max_length=max_length)
    else:
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
    payload: Dict[str, Any] = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        "row_id": row_ids,
    }
    if "ce_weight" in df.columns:
        payload["ce_weight"] = df["ce_weight"].astype(float).tolist()
    return Dataset.from_dict(payload)


def get_predictions(
    model: nn.Module,
    dataset: Dataset,
    df: pd.DataFrame,
    tokenizer: Any,
    batch_size: int,
    device: str,
    max_length: int = 256,
    fixed_max_length: bool = False,
    include_sample_loss: bool = False,
) -> pd.DataFrame:
    # HF Trainer (eval_strategy="no") calls model.train() only once before the epoch
    # loop. Restoring the prior mode is required so on_epoch_end dynamics do not leave
    # subsequent training epochs stuck in eval() (no dropout).
    was_training = bool(model.training)
    model.eval()
    model.to(device)
    try:
        collator = BinaryDataCollator(tokenizer, max_length=max_length, fixed_max_length=fixed_max_length)
        all_row_ids, all_probs, all_preds, all_labels = [], [], [], []
        for i in range(0, len(dataset), batch_size):
            batch_data = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            batch = collator(batch_data)
            row_ids = batch.pop("row_ids")
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs["logits"]
                bs = len(batch_data)
                if logits.dim() == 2 and logits.shape[-1] == 2:
                    if logits.shape[0] != bs:
                        raise RuntimeError(
                            f"get_predictions: expected logits shape ({bs}, 2), got {tuple(logits.shape)}."
                        )
                    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    preds = torch.argmax(logits, dim=-1).cpu().numpy().astype(int)
                else:
                    logits = logits.squeeze(-1)
                    if logits.ndim != 1 or logits.shape[0] != bs:
                        raise RuntimeError(
                            f"get_predictions: expected logits shape ({bs},) or ({bs}, 1), got {tuple(logits.shape)} — "
                            "check RoBERTaBinaryClassifier forward / batching."
                        )
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= 0.5).astype(int)
            all_row_ids.extend(row_ids)
            all_probs.extend(np.atleast_1d(probs).tolist())
            all_preds.extend(np.atleast_1d(preds).tolist())
            all_labels.extend(labels.numpy().tolist())
        probs_arr = np.asarray(all_probs, dtype=np.float64)
        if probs_arr.size > 1:
            std = float(np.std(probs_arr))
            if std < 1e-7:
                print(
                    f"WARNING: get_predictions: probabilities are nearly constant (std={std:.2e}, "
                    f"min={float(probs_arr.min()):.6f}, max={float(probs_arr.max()):.6f}). "
                    "AUC will be ~0.5 and thresholding will not separate classes — check training "
                    "(loss curve) or that the model is not stuck outputting one logit per example."
                )
        preds_df = pd.DataFrame(
            {
                "row_id": all_row_ids,
                "label": all_labels,
                "prob": all_probs,
                "pred": all_preds,
                "correct": [int(p == l) for p, l in zip(all_preds, all_labels)],
            }
        )
        if include_sample_loss:
            y = preds_df["label"].astype(float).to_numpy()
            p = preds_df["prob"].astype(float).to_numpy()
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            preds_df["sample_loss"] = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        text_df = df[["row_id", "text", "class"]].copy()
        return preds_df.merge(text_df, on="row_id", how="left")
    finally:
        model.train(was_training)


def load_splits_baseline(
    data_root: Path,
    dataset_name: str,
    splitted_data_subdir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, test from data/<dataset>/<subdir>/. Each has binary_label and stable row_id."""
    from pipeline.common.paths import resolve_split_csv

    sub = (splitted_data_subdir or "").strip() or SPLITTED_DATA_SUBDIR
    base = data_root / dataset_name / sub
    train_df = pd.read_csv(resolve_split_csv(base, "train"))
    validation_df = pd.read_csv(resolve_split_csv(base, "validation"))
    test_df = pd.read_csv(resolve_split_csv(base, "test"))

    for name, df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        if "text" not in df.columns or "class" not in df.columns:
            raise ValueError(f"{name}.csv missing required columns: text, class")
        df["binary_label"] = df["class"].astype(int)

    # Preserve stable ids when present (CV folds / custom splits); else synthesize by position.
    def _ensure_row_ids(df: pd.DataFrame, prefix: str) -> None:
        if "row_id" in df.columns and df["row_id"].notna().all() and (df["row_id"].astype(str).str.len() > 0).all():
            df["row_id"] = df["row_id"].astype(str)
        else:
            df["row_id"] = [f"{prefix}_{i}" for i in range(len(df))]

    _ensure_row_ids(train_df, "train")
    _ensure_row_ids(validation_df, "validation")
    _ensure_row_ids(test_df, "test")

    return train_df, validation_df, test_df


def resolve_best_epoch_from_train_baseline(
    data_root: Path,
    dataset_name: str,
    metric_summary: Optional[Path] = None,
    train_baseline_subdir: Optional[str] = None,
) -> Tuple[int, Path, Dict[str, Any]]:
    """
    Read ``best_epoch_by_eval_f1_macro.epoch`` from a train-only baseline metric_summary.json.

    Returns (num_epochs_ceil, summary_path, best_epoch_info).
    """
    if metric_summary is not None:
        summary_path = Path(metric_summary).expanduser().resolve()
    else:
        sub = (train_baseline_subdir or "").strip() or os.environ.get(
            "ECHOHATE_ROBERTA_TRAIN_BASELINE_DIR", "roberta_large_baseline"
        ).strip() or "roberta_large_baseline"
        summary_path = (data_root / dataset_name / sub / "metric_summary.json").resolve()
    if not summary_path.is_file():
        raise SystemExit(
            f"ERROR: --epochs-from-train-baseline needs metric_summary.json at {summary_path}"
        )
    try:
        j = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"ERROR: failed to read {summary_path}: {e}") from e
    be = ((j.get("training_config") or {}).get("best_epoch_by_eval_f1_macro") or {})
    raw = be.get("epoch")
    if raw is None:
        raise SystemExit(
            f"ERROR: no training_config.best_epoch_by_eval_f1_macro.epoch in {summary_path}"
        )
    # HF Trainer logs fractional epochs (e.g. 3.0); train for that many full epochs.
    num_epochs = int(max(1, int(np.ceil(float(raw)))))
    info = {
        "source_metric_summary": str(summary_path),
        "train_only_best_epoch_raw": float(raw),
        "num_epochs_applied": int(num_epochs),
        "eval_f1_macro": be.get("eval_f1_macro"),
        "best_model_checkpoint": be.get("best_model_checkpoint"),
    }
    return num_epochs, summary_path, info


def _prefer_echohate_path_str(path: Path) -> str:
    """Record .../echohate/... when that path is an alias of .../co2hate/... (same inode)."""
    s = str(path)
    if "/co2hate/" not in s.replace("\\", "/"):
        return s
    try:
        phys = path.resolve()
        # co2hate → echohate sibling of the same checkout
        parts = list(phys.parts)
        if "co2hate" not in parts:
            return s
        i = parts.index("co2hate")
        alias = Path(*parts[:i], "echohate", *parts[i + 1 :])
        if alias.exists() and alias.resolve() == phys:
            return str(alias)
        # rewrite string only (symlink may live at experiment/echohate → co2hate)
        parent = phys
        while parent.name not in ("co2hate", "echohate") and parent != parent.parent:
            parent = parent.parent
        if parent.name == "co2hate":
            ech = parent.parent / "echohate"
            if ech.exists() and ech.resolve() == parent.resolve():
                return s.replace("/co2hate/", "/echohate/").replace("\\co2hate\\", "\\echohate\\")
    except OSError:
        pass
    return s


def resolve_val_threshold_from_train_baseline(
    data_root: Path,
    dataset_name: str,
    metric_summary: Optional[Path] = None,
    train_baseline_subdir: Optional[str] = None,
) -> Tuple[float, Path, Dict[str, Any]]:
    """
    Read val-tuned τ from a train-only baseline ``metric_summary.json``.

    Prefers ``threshold_policy.best_threshold_validation``, then
    ``applied_threshold_validation`` / ``applied_threshold_test``.
    """
    if metric_summary is not None:
        summary_path = Path(metric_summary).expanduser().resolve()
    else:
        sub = (train_baseline_subdir or "").strip() or os.environ.get(
            "ECHOHATE_ROBERTA_TRAIN_BASELINE_DIR", "roberta_large_baseline"
        ).strip() or "roberta_large_baseline"
        summary_path = (data_root / dataset_name / sub / "metric_summary.json").resolve()
    if not summary_path.is_file():
        raise SystemExit(
            f"ERROR: train-on-train-val needs train-only metric_summary.json for val τ at {summary_path}"
        )
    try:
        j = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"ERROR: failed to read {summary_path}: {e}") from e
    tp = j.get("threshold_policy") or {}
    raw = (
        tp.get("best_threshold_validation")
        if tp.get("best_threshold_validation") is not None
        else tp.get("applied_threshold_validation")
        if tp.get("applied_threshold_validation") is not None
        else tp.get("applied_threshold_test")
    )
    if raw is None:
        raise SystemExit(
            f"ERROR: no threshold_policy.best_threshold_validation (or applied_*) in {summary_path}"
        )
    thr = float(raw)
    if not (0.0 <= thr <= 1.0):
        raise SystemExit(f"ERROR: invalid train-only val threshold {thr} in {summary_path}")
    info = {
        "source_metric_summary": _prefer_echohate_path_str(summary_path),
        "best_threshold_validation": thr,
        "threshold_key_used": (
            "best_threshold_validation"
            if tp.get("best_threshold_validation") is not None
            else "applied_threshold_validation"
            if tp.get("applied_threshold_validation") is not None
            else "applied_threshold_test"
        ),
    }
    return thr, summary_path, info


def main() -> None:
    p = argparse.ArgumentParser(description="EchoHate standalone RoBERTa baseline (distill-style Trainer)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", default=str(get_data_root()))
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--model_name",
        default="roberta-large",
        help="HF model id. Paper runs use roberta-large.",
    )
    p.add_argument("--num_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--random_state", type=int, default=None, help="RNG seed (default: config/echohate_project_config.json seeds.random_state)")
    p.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "bce"],
        help=(
            "ce: two-class head + CrossEntropyLoss (like custom_model_tackling_hate). "
            "bce: MLP + one logit + BCEWithLogits (legacy)."
        ),
    )
    p.add_argument(
        "--pos_weight_mode",
        type=str,
        default="none",
        choices=["freq", "sqrt", "none"],
        help="CE/BCE class imbalance (default none = unweighted CE, same as custom_model_tackling_hate).",
    )
    p.add_argument(
        "--class_weight_target",
        type=str,
        default="pos",
        choices=["pos", "neg"],
        help="BCE only: which class gets pos_weight semantics. Ignored for --loss-type ce.",
    )
    p.add_argument(
        "--ce-contrastive-lambda",
        "--ce_contrastive_lambda",
        type=float,
        default=None,
        dest="ce_contrastive_lambda",
        metavar="LAMBDA",
        help=(
            "Mixed training loss when set: LAMBDA*CE + (1-LAMBDA)*ContrastiveLossCosine(logits). "
            "LAMBDA must be in (0, 1]. Omit for CE-only (do not pass 0 — that is not 'off'). "
            "Requires --loss-type ce."
        ),
    )
    p.add_argument(
        "--contrastive-margin",
        "--contrastive_margin",
        type=float,
        default=0.75,
        dest="contrastive_margin",
        help="Margin in ContrastiveLossCosine on logits (default 0.75).",
    )
    p.add_argument(
        "--balance-train-classes",
        dest="balance_train_classes",
        action="store_true",
        help=(
            "Oversample the minority class with replacement until the training set is class-balanced. "
            "Apply this when you want imbalance handled by sampling only; pair with --pos_weight_mode none "
            "to avoid double-counting with CE class weights."
        ),
    )
    p.add_argument(
        "--pad-to-max-length",
        dest="pad_to_max_length",
        action="store_true",
        help="Fixed padding to max_length (default on, matches custom tokenize).",
    )
    p.add_argument(
        "--no-pad-to-max-length",
        dest="pad_to_max_length",
        action="store_false",
        help="Dynamic padding in the collator.",
    )
    p.set_defaults(pad_to_max_length=True)
    p.add_argument(
        "--skip-test-infer",
        dest="skip_test_infer",
        action="store_true",
        help="Skip test.csv inference and test_metrics (default: infer test after tuning on validation).",
    )
    p.add_argument(
        "--train-on-train-val",
        dest="train_on_train_val",
        action="store_true",
        help=(
            "Train on train.csv ∪ validation.csv (no held-out val). Disables per-epoch eval / "
            "best-checkpoint selection. Applied test F1 threshold is controlled by "
            "--trainval-applied-threshold (default: from80)."
        ),
    )
    p.add_argument(
        "--trainval-applied-threshold",
        dest="trainval_applied_threshold",
        choices=("from80", "0.5"),
        default="from80",
        help=(
            "With --train-on-train-val: applied τ for metrics/test. "
            "'from80' = val-tuned τ from matching 80%% train-only baseline metric_summary; "
            "'0.5' = fixed 0.5 (legacy trainval protocol)."
        ),
    )
    p.add_argument(
        "--epochs-from-train-baseline",
        dest="epochs_from_train_baseline",
        action="store_true",
        help=(
            "Set --num_epochs to the best val epoch from a train-only baseline metric_summary.json "
            "(see --epochs-from-metric-summary / ECHOHATE_ROBERTA_TRAIN_BASELINE_DIR). "
            "Intended with --train-on-train-val."
        ),
    )
    p.add_argument(
        "--epochs-from-metric-summary",
        dest="epochs_from_metric_summary",
        default=None,
        help=(
            "Path to a train-only metric_summary.json (overrides default "
            "<data_root>/<dataset>/roberta_large_baseline/metric_summary.json)."
        ),
    )
    p.add_argument("--require_gpu", action="store_true")
    p.add_argument(
        "--track-dynamics",
        action="store_true",
        help="Record per-epoch training dynamics on train split and save training_dynamics.csv.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 on (redundant on CUDA: fp16 is default when CUDA is available, like custom).",
    )
    p.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 even when CUDA is available (custom defaults fp16 on with CUDA).",
    )
    p.add_argument(
        "--bf16",
        action="store_true",
        help="bfloat16 training (Ampere+ GPUs). Mutually exclusive with fp16.",
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Gradient clipping (Trainer default is 1.0 if unset). Pass e.g. 1.0 explicitly for full FT stability.",
    )
    p.add_argument(
        "--no-eval-each-epoch",
        dest="eval_each_epoch",
        action="store_false",
        help="Disable per-epoch eval (custom runs eval every epoch and loads best by eval_f1_macro).",
    )
    p.set_defaults(eval_each_epoch=True)
    args = p.parse_args()
    if args.random_state is None:
        args.random_state = int(default_random_state())

    if args.bf16:
        fp16_train = False
    elif args.no_fp16:
        fp16_train = False
    else:
        # Match custom: fp16 on when CUDA unless --no-fp16 or --bf16; --fp16 is a no-op on CUDA
        fp16_train = bool(torch.cuda.is_available())

    if args.bf16 and args.fp16:
        raise SystemExit("ERROR: use at most one of --fp16 and --bf16.")

    if args.ce_contrastive_lambda is not None and str(args.loss_type).strip().lower() != "ce":
        raise SystemExit("ERROR: --ce-contrastive-lambda requires --loss-type ce.")

    if args.require_gpu and not torch.cuda.is_available():
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise SystemExit(f"ERROR: --require_gpu set but torch.cuda.is_available() is False. CUDA_VISIBLE_DEVICES={cvd}")

    set_seed(args.random_state)
    data_root = Path(args.data_root).expanduser()
    out_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else resolve_baseline_run_dir(data_root, args.dataset, args.model_name)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_bucket = os.environ.get("ECHOHATE_DATA_SEED_DIR", "").strip()

    epochs_from_train_info: Optional[Dict[str, Any]] = None
    if bool(getattr(args, "epochs_from_train_baseline", False)) or (
        getattr(args, "epochs_from_metric_summary", None) is not None
    ):
        n_ep, src_path, epochs_from_train_info = resolve_best_epoch_from_train_baseline(
            data_root,
            args.dataset,
            metric_summary=(
                Path(args.epochs_from_metric_summary).expanduser()
                if getattr(args, "epochs_from_metric_summary", None)
                else None
            ),
        )
        print(
            f"[epochs-from-train-baseline] num_epochs {args.num_epochs} -> {n_ep} "
            f"(from {src_path}; raw_epoch={epochs_from_train_info['train_only_best_epoch_raw']})"
        )
        args.num_epochs = int(n_ep)
    if seed_bucket:
        def _path_has_seed_bucket(path: str, seed: str) -> bool:
            """True iff ``seed`` appears as a full path segment (not a prefix of another bucket)."""
            p = path.rstrip("/")
            if p == seed or p == f"data/{seed}":
                return True
            if f"/{seed}/" in f"{p}/":
                return True
            return p.endswith(f"/{seed}")

        dr_s, out_s = str(data_root), str(out_dir)
        if not _path_has_seed_bucket(dr_s, seed_bucket) or not _path_has_seed_bucket(out_s, seed_bucket):
            raise ValueError(
                f"ECHOHATE_DATA_SEED_DIR={seed_bucket!r} but data_root={data_root} "
                f"and output_dir={out_dir} are not under the same seed bucket "
                f"(path-segment match; seed_0_model ≠ seed_0_model_wCL)"
            )

    pipeline_t0 = time.perf_counter()
    pipeline_started_at_utc = datetime.now(timezone.utc).isoformat()

    train_df, validation_df, test_df = load_splits_baseline(data_root, args.dataset)

    train_on_train_val = bool(getattr(args, "train_on_train_val", False))
    if train_on_train_val:
        # Concatenate official splits; keep original row_ids (train_* / validation_*).
        train_combined = pd.concat([train_df, validation_df], ignore_index=True)
        if args.eval_each_epoch:
            print(
                "[train-on-train-val] disabling per-epoch eval / load_best_model_at_end "
                "(no held-out validation)."
            )
            args.eval_each_epoch = False
    else:
        train_combined = train_df.copy()

    if args.balance_train_classes:
        n_before = len(train_combined)
        train_combined = _oversample_minority_to_balance(train_combined, "binary_label", args.random_state)
        print(
            f"balance_train_classes: oversampled minority — {n_before} -> {len(train_combined)} rows "
            f"(counts 0/1: {(train_combined['binary_label'] == 0).sum()}/{(train_combined['binary_label'] == 1).sum()})"
        )
        # Oversampling can duplicate row_ids; re-index for uniqueness.
        train_combined["row_id"] = [f"train_combined_{i}" for i in range(len(train_combined))]

    # Held-out validation for Trainer early-stop + threshold tuning (unused when train_on_train_val).
    # Post-hoc predictions_eval.csv still uses the original validation split in both modes.
    tune_df = validation_df

    train_only_threshold_info: Optional[Dict[str, Any]] = None
    if train_on_train_val:
        print(
            f"Train: {len(train_combined)} rows (train+validation)  "
            f"Held-out tune: none (fixed epochs={args.num_epochs}; "
            f"test τ from 80% train-only baseline)  "
            f"Test infer: {'skipped' if args.skip_test_infer else len(test_df)} rows"
        )
    else:
        print(
            f"Train: {len(train_combined)} rows  "
            f"Tune (Trainer eval + threshold): validation {len(tune_df)} rows  "
            f"Test infer: {'skipped' if args.skip_test_infer else len(test_df)} rows"
        )
    train_labels_np = train_combined["binary_label"].astype(int).values
    loss_type = str(args.loss_type).strip().lower()
    pos_weight_scalar: Optional[float] = None
    ce_class_weight_tensor: Optional[torch.Tensor] = None
    if loss_type == "bce":
        pos_weight_scalar = _compute_pos_weight(train_labels_np, args.pos_weight_mode, args.class_weight_target)
        print(
            f"loss_type=bce pos_weight_mode={args.pos_weight_mode} class_weight_target={args.class_weight_target} "
            f"pos_weight={pos_weight_scalar}"
        )
    else:
        ce_class_weight_tensor = _compute_ce_class_weights(train_labels_np, args.pos_weight_mode)
        print(f"loss_type=ce pos_weight_mode={args.pos_weight_mode} ce_class_weight={ce_class_weight_tensor}")

    model = RoBERTaBinaryClassifier(
        model_name=args.model_name,
        loss_type=loss_type,
        ce_contrastive_lambda=args.ce_contrastive_lambda,
        contrastive_margin=float(args.contrastive_margin),
    )
    model.class_weight_target = str(args.class_weight_target)
    if loss_type == "ce" and ce_class_weight_tensor is not None:
        model.register_buffer("ce_class_weight", ce_class_weight_tensor.detach().float().clone())
    if loss_type == "bce" and pos_weight_scalar is not None:
        model.register_buffer(
            "bce_pos_weight",
            torch.tensor([float(pos_weight_scalar)], dtype=torch.float32),
        )

    enc_n = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_n = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    enc_tot = sum(p.numel() for p in model.backbone.parameters())
    print(
        f"Full fine-tune: encoder trainable {enc_n:,} / {enc_tot:,} params; "
        f"classifier head trainable {head_n:,}"
    )

    _ddp_lr = _torchrun_local_rank()
    if torch.cuda.is_available():
        if _ddp_lr >= 0:
            torch.cuda.set_device(_ddp_lr)
            model = model.to(torch.device("cuda", _ddp_lr))
            runtime_device: str = f"cuda:{_ddp_lr}"
        else:
            model = model.to("cuda")
            runtime_device = "cuda"
    else:
        # Trainer is pinned to CPU when CUDA is absent (see use_cpu below).
        runtime_device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = prepare_dataset(
        train_combined,
        tokenizer,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
    )
    tune_ds = prepare_dataset(
        tune_df,
        tokenizer,
        max_length=args.max_length,
        pad_to_max_length=args.pad_to_max_length,
    )

    eval_batch = max(1, int(args.eval_batch_size))
    train_batch = max(1, int(args.batch_size))
    warmup_steps = int(((len(train_ds) + train_batch - 1) // train_batch) * args.num_epochs * args.warmup_ratio)

    ta_kw: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        num_train_epochs=float(args.num_epochs),
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        warmup_steps=warmup_steps,
        eval_strategy="epoch" if args.eval_each_epoch else "no",
        save_strategy="epoch" if args.eval_each_epoch else "no",
        # Julian / DGX: keep at most one HF checkpoint when saving each epoch.
        save_total_limit=1 if args.eval_each_epoch else None,
        load_best_model_at_end=bool(args.eval_each_epoch),
        metric_for_best_model=("eval_f1_macro" if args.eval_each_epoch else None),
        greater_is_better=True if args.eval_each_epoch else None,
        seed=int(args.random_state),
        report_to="none",
        remove_unused_columns=False,
        # Custom nn.Module (not PreTrainedModel): HF cannot infer label columns; without this,
        # prediction_step leaves labels=None and compute_metrics never runs (no eval_f1_macro).
        label_names=["labels"],
        dataloader_num_workers=0,
        fp16=bool(fp16_train),
        bf16=bool(args.bf16),
        ddp_find_unused_parameters=False,
        use_cpu=not bool(torch.cuda.is_available()),
    )
    if args.max_grad_norm is not None:
        ta_kw["max_grad_norm"] = float(args.max_grad_norm)
    if not args.eval_each_epoch:
        ta_kw.pop("metric_for_best_model", None)
        ta_kw.pop("greater_is_better", None)

    if fp16_train or args.bf16:
        print(f"Mixed precision: fp16={fp16_train} bf16={args.bf16}")
    if args.max_grad_norm is not None:
        print(f"max_grad_norm={args.max_grad_norm}")
    if args.eval_each_epoch:
        print(
            "Per-epoch eval: metric_for_best_model=eval_f1_macro on validation "
            "(τ∈[0.05,0.95] step 0.05 by macro F1; score=macro-F1; preds use tuned τ). "
            "Test inferred post-hoc unless --skip-test-infer."
        )

    training_args = TrainingArguments(**{k: v for k, v in ta_kw.items() if v is not None})

    _baseline_collator = BinaryDataCollator(
        tokenizer, max_length=args.max_length, fixed_max_length=args.pad_to_max_length
    )
    dynamics_callback: Optional[TrainingDynamicsCallback] = None
    if args.track_dynamics:
        dynamics_callback = TrainingDynamicsCallback(
            train_dataset=train_ds,
            train_df=train_combined,
            collate_fn=_baseline_collator,
            eval_batch_size=eval_batch,
            device=runtime_device,
            output_dir=out_dir,
            label_col="binary_label",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=tune_ds if args.eval_each_epoch else None,
        compute_metrics=_compute_metrics_for_trainer_best_threshold if args.eval_each_epoch else None,
        data_collator=_baseline_collator,
        processing_class=tokenizer,
        callbacks=[dynamics_callback] if dynamics_callback is not None else None,
    )

    best_epoch_from_eval: Optional[Dict[str, Any]] = None
    train_t0 = time.perf_counter()
    setup_wall_sec = float(train_t0 - pipeline_t0)
    train_result = trainer.train()
    train_wall_sec = float(time.perf_counter() - train_t0)
    hf_train_metrics: Dict[str, Any] = {}
    if train_result is not None and getattr(train_result, "metrics", None):
        for _k, _v in train_result.metrics.items():
            if isinstance(_v, bool) or _v is None or isinstance(_v, str):
                hf_train_metrics[_k] = _v
            elif hasattr(_v, "item"):
                hf_train_metrics[_k] = _v.item()
            elif isinstance(_v, float):
                hf_train_metrics[_k] = _v
            elif isinstance(_v, int):
                hf_train_metrics[_k] = _v
            else:
                hf_train_metrics[_k] = str(_v)
    print(
        f"\n⏱ Timing (wall clock): setup={setup_wall_sec:.1f}s  Trainer.train={train_wall_sec:.1f}s"
        + (
            f"  HF train_runtime={hf_train_metrics.get('train_runtime', 'n/a')}"
            if hf_train_metrics
            else ""
        )
    )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if _is_rank0_for_writes():
        if args.eval_each_epoch:
            epoch_key = "eval_f1_macro"
            eval_logs = [h for h in trainer.state.log_history if epoch_key in h]
            if eval_logs:
                best_log = max(eval_logs, key=lambda h: float(h[epoch_key]))
                best_epoch_from_eval = {
                    "epoch": best_log.get("epoch"),
                    "eval_f1_macro": float(best_log.get("eval_f1_macro", 0.0)),
                    "eval_accuracy": float(best_log.get("eval_accuracy", 0.0)),
                    "eval_auc_macro": float(
                        best_log.get("eval_auc_macro", best_log.get("eval_auc", 0.0))
                    ),
                    "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
                    "trainer_best_metric": float(trainer.state.best_metric)
                    if getattr(trainer.state, "best_metric", None) is not None
                    else None,
                }
                print(f"\n=== Best epoch (by {epoch_key} on validation split) ===")
                print(f"  epoch: {best_log.get('epoch', 'n/a')}")
                print(f"  eval_f1_macro: {float(best_log.get('eval_f1_macro', 0.0)):.6f}")
                print(f"  eval_accuracy: {float(best_log.get('eval_accuracy', 0.0)):.6f}")
                print(
                    f"  eval_auc_macro: {float(best_log.get('eval_auc_macro', best_log.get('eval_auc', 0.0))):.6f}"
                )
                if getattr(trainer.state, "best_model_checkpoint", None):
                    print(f"  checkpoint: {trainer.state.best_model_checkpoint}")
                if getattr(trainer.state, "best_metric", None) is not None:
                    print(f"  trainer.state.best_metric ({epoch_key}): {trainer.state.best_metric:.6f}")
            else:
                print(f"WARNING: no {epoch_key} entries in log_history (eval may have failed).")

        infer_model = trainer.model
        if hasattr(infer_model, "module"):
            infer_model = infer_model.module
        infer_model.eval()

        device = runtime_device

        infer_t0 = time.perf_counter()
        train_preds_df = get_predictions(
            infer_model,
            train_ds,
            train_combined,
            tokenizer,
            batch_size=eval_batch,
            device=device,
            max_length=args.max_length,
            fixed_max_length=args.pad_to_max_length,
        )
        validation_preds_df = get_predictions(
            infer_model,
            tune_ds,
            tune_df,
            tokenizer,
            batch_size=eval_batch,
            device=device,
            max_length=args.max_length,
            fixed_max_length=args.pad_to_max_length,
        )
        test_preds_df: Optional[pd.DataFrame] = None
        if not args.skip_test_infer:
            test_ds = prepare_dataset(
                test_df,
                tokenizer,
                max_length=args.max_length,
                pad_to_max_length=args.pad_to_max_length,
            )
            test_preds_df = get_predictions(
                infer_model,
                test_ds,
                test_df,
                tokenizer,
                batch_size=eval_batch,
                device=device,
                max_length=args.max_length,
                fixed_max_length=args.pad_to_max_length,
            )
        posthoc_inference_wall_sec = float(time.perf_counter() - infer_t0)
        n_pred_samples = int(len(train_ds) + len(tune_ds))
        if test_preds_df is not None:
            n_pred_samples += int(len(test_preds_df))
        pred_samples_per_sec = (n_pred_samples / posthoc_inference_wall_sec) if posthoc_inference_wall_sec > 0 else None
        print(
            f"⏱ Post-hoc predictions (train+eval splits): {posthoc_inference_wall_sec:.1f}s wall, "
            f"{n_pred_samples} rows"
            + (f", ~{pred_samples_per_sec:.1f} rows/s" if pred_samples_per_sec else "")
        )

        train_labels = train_preds_df["label"].astype(int).values
        train_probs = train_preds_df["prob"].astype(float).values
        validation_labels = validation_preds_df["label"].astype(int).values
        validation_probs = validation_preds_df["prob"].astype(float).values

        # Tune τ on 0.05…0.95 step 0.05 (macro F1); metrics + preds use applied τ.
        best_train_t = _tune_threshold_macro_f1(train_labels, train_probs)
        best_val_t = _tune_threshold_macro_f1(validation_labels, validation_probs)

        # Default: apply val-tuned threshold to train/validation/test.
        # train-on-train-val: do not retune τ on val (it was in the train pool).
        # --trainval-applied-threshold: from80 (copy 80% baseline τ) or 0.5 (legacy fixed).
        if train_on_train_val:
            thr_mode = str(getattr(args, "trainval_applied_threshold", "from80") or "from80")
            if thr_mode == "0.5":
                applied_t = 0.5
                train_only_threshold_info = None
                print("[train-on-train-val] applied τ=0.5000 (fixed; legacy protocol)")
            else:
                applied_t, _thr_src, train_only_threshold_info = resolve_val_threshold_from_train_baseline(
                    data_root=Path(args.data_root),
                    dataset_name=str(args.dataset),
                    metric_summary=(
                        Path(args.epochs_from_metric_summary).expanduser().resolve()
                        if getattr(args, "epochs_from_metric_summary", None)
                        else None
                    ),
                )
                print(
                    f"[train-on-train-val] applied τ={applied_t:.4f} from train-only "
                    f"{train_only_threshold_info['source_metric_summary']} "
                    f"({train_only_threshold_info['threshold_key_used']})"
                )
        else:
            applied_t = float(best_val_t)

        raw_train_preds_df = train_preds_df.copy()
        raw_validation_preds_df = validation_preds_df.copy()
        raw_test_preds_df = test_preds_df.copy() if test_preds_df is not None else None

        train_best_metrics = _metrics_at_tau(train_labels, train_probs, applied_t)
        validation_applied_metrics = _metrics_at_tau(validation_labels, validation_probs, applied_t)
        validation_best_metrics = dict(validation_applied_metrics)

        test_best_metrics: Optional[Dict[str, Any]] = None
        test_labels: Optional[np.ndarray] = None
        test_probs: Optional[np.ndarray] = None
        if raw_test_preds_df is not None:
            test_labels = raw_test_preds_df["label"].astype(int).values
            test_probs = raw_test_preds_df["prob"].astype(float).values
            test_best_metrics = _metrics_at_tau(test_labels, test_probs, applied_t)

        train_tau050_metrics = _metrics_at_tau(train_labels, train_probs, FIXED_PREDICTION_TAU)
        validation_tau050_metrics = _metrics_at_tau(
            validation_labels, validation_probs, FIXED_PREDICTION_TAU
        )
        test_tau050_metrics: Optional[Dict[str, Any]] = None
        if test_labels is not None and test_probs is not None:
            test_tau050_metrics = _metrics_at_tau(test_labels, test_probs, FIXED_PREDICTION_TAU)

        model_dir = out_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "pytorch_model.pt")
        with open(model_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": args.model_name,
                    "architecture": "RoBERTaBinaryClassifier",
                    "loss_type": str(loss_type),
                    "ce_contrastive_lambda": (
                        float(args.ce_contrastive_lambda) if args.ce_contrastive_lambda is not None else None
                    ),
                    "contrastive_margin": float(args.contrastive_margin),
                },
                f,
                indent=2,
            )
        tokenizer.save_pretrained(str(out_dir / "tokenizer"))

        _write_prediction_variant(
            raw_train_preds_df, tau=applied_t, out_dir=out_dir, stem="predictions_train"
        )
        _write_prediction_variant(
            raw_validation_preds_df, tau=applied_t, out_dir=out_dir, stem="predictions_eval"
        )
        if raw_test_preds_df is not None:
            test_preds_df = _write_prediction_variant(
                raw_test_preds_df, tau=applied_t, out_dir=out_dir, stem="predictions_test"
            )
        _write_prediction_variant(
            raw_train_preds_df, tau=FIXED_PREDICTION_TAU, out_dir=out_dir, stem="predictions_train_tau050"
        )
        _write_prediction_variant(
            raw_validation_preds_df,
            tau=FIXED_PREDICTION_TAU,
            out_dir=out_dir,
            stem="predictions_eval_tau050",
        )
        if raw_test_preds_df is not None:
            _write_prediction_variant(
                raw_test_preds_df, tau=FIXED_PREDICTION_TAU, out_dir=out_dir, stem="predictions_test_tau050"
            )
        dynamics_summary_rows = None
        if dynamics_callback is not None:
            dynamics_df = dynamics_callback.finalize()
            dynamics_summary_rows = int(len(dynamics_df)) if dynamics_df is not None else 0

        total_pipeline_wall_sec = float(time.perf_counter() - pipeline_t0)
        timing_detail: Dict[str, Any] = {
            "pipeline_started_at_utc": pipeline_started_at_utc,
            "setup_wall_sec": round(setup_wall_sec, 4),
            "trainer_train_wall_sec": round(train_wall_sec, 4),
            "posthoc_inference_wall_sec": round(posthoc_inference_wall_sec, 4),
            "total_pipeline_wall_sec": round(total_pipeline_wall_sec, 4),
            "posthoc_prediction_rows": n_pred_samples,
            "posthoc_prediction_rows_per_sec": round(pred_samples_per_sec, 4) if pred_samples_per_sec is not None else None,
            "huggingface_trainer_metrics": hf_train_metrics,
            "notes": (
                "Compare with LLM jobs using total_pipeline_wall_sec (full run) or trainer_train_wall_sec (HF training only). "
                "posthoc_inference_wall_sec is batched forward on train+eval for predictions CSVs (not generation)."
            ),
        }

        summary: Dict[str, Any] = {
            "model_info": {
                "model_name": args.model_name,
                "output_dir": str(out_dir),
                "training_date": datetime.now().isoformat(),
            },
            "timing": timing_detail,
            "dataset_info": {
                "training_rows": "train_plus_validation" if train_on_train_val else "train_only",
                "protocol": (
                    "train_on_train_plus_validation_infer_on_test"
                    if train_on_train_val
                    else "train_on_train_tune_on_validation_infer_on_test"
                ),
                "train_on_train_val": bool(train_on_train_val),
                "trainval_applied_threshold": (
                    str(getattr(args, "trainval_applied_threshold", "from80"))
                    if train_on_train_val
                    else None
                ),
                "epochs_from_train_baseline": epochs_from_train_info,
                "threshold_from_train_baseline": train_only_threshold_info,
                "skip_test_infer": bool(args.skip_test_infer),
                "balance_train_classes": bool(args.balance_train_classes),
                "train_samples": int(len(train_ds)),
                "validation_samples": int(len(tune_ds)),
                "test_samples": int(len(test_preds_df)) if test_preds_df is not None else 0,
                "track_dynamics": bool(args.track_dynamics),
                "training_dynamics_rows": dynamics_summary_rows,
            },
            "training_config": {
                "num_epochs": int(args.num_epochs),
                "batch_size": int(args.batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "warmup_ratio": float(args.warmup_ratio),
                "fp16": bool(fp16_train),
                "bf16": bool(args.bf16),
                "max_grad_norm": (float(args.max_grad_norm) if args.max_grad_norm is not None else None),
                "eval_each_epoch": bool(args.eval_each_epoch),
                "best_epoch_by_eval_f1_macro": best_epoch_from_eval,
                "max_length": int(args.max_length),
                "random_state": int(args.random_state),
                "loss_type": str(loss_type),
                "balance_train_classes": bool(args.balance_train_classes),
                "class_weighting": (
                    f"ce_weight:{args.pos_weight_mode}"
                    if loss_type == "ce"
                    else f"pos_weight:{args.pos_weight_mode}:{args.class_weight_target}"
                ),
                "pos_weight_mode": str(args.pos_weight_mode),
                "class_weight_target": str(args.class_weight_target),
                "pos_weight_value": (float(pos_weight_scalar) if pos_weight_scalar is not None else None),
                "ce_class_weight": (
                    ce_class_weight_tensor.detach().cpu().tolist() if ce_class_weight_tensor is not None else None
                ),
                "ce_contrastive_lambda": (
                    float(args.ce_contrastive_lambda) if args.ce_contrastive_lambda is not None else None
                ),
                "contrastive_margin": float(args.contrastive_margin),
            },
            "threshold_policy": {
                "sweep_min": 0.05,
                "sweep_max": 0.95,
                "sweep_step": 0.05,
                "threshold_selection": "macro_f1",
                "epoch_selection": "macro_f1_at_best_threshold",
                "selection_metric": "f1_macro",
                "tune_split": (
                    (
                        None
                        if str(getattr(args, "trainval_applied_threshold", "from80")) == "0.5"
                        else "train_only_baseline_validation"
                    )
                    if train_on_train_val
                    else "validation"
                ),
                "threshold_note": (
                    (
                        "Fixed applied threshold 0.5 because validation was included in training "
                        "(best_threshold_validation is diagnostic-only)."
                        if str(getattr(args, "trainval_applied_threshold", "from80")) == "0.5"
                        else (
                            "Applied τ copied from 80% train-only baseline "
                            "threshold_policy.best_threshold_validation (validation was in training; "
                            "best_threshold_validation below is this run's diagnostic sweep only)."
                        )
                    )
                    if train_on_train_val
                    else (
                        "Applied τ selected on validation: "
                        "grid 0.05–0.95 step 0.05 by macro F1; metrics/preds at that τ."
                    )
                ),
                "trainval_applied_threshold_mode": (
                    str(getattr(args, "trainval_applied_threshold", "from80"))
                    if train_on_train_val
                    else None
                ),
                "best_threshold_train": float(best_train_t),
                "best_threshold_validation": float(best_val_t),
                "applied_threshold_validation": float(applied_t),
                "applied_threshold_test": float(applied_t) if test_best_metrics is not None else None,
                "fixed_threshold_tau050": float(FIXED_PREDICTION_TAU),
                "train_only_baseline_threshold": (
                    float(train_only_threshold_info["best_threshold_validation"])
                    if train_only_threshold_info
                    else None
                ),
            },
            "threshold_variants": {
                "val_tuned": {
                    "applied_threshold": float(applied_t),
                    "predictions_train": "predictions_train.csv",
                    "predictions_eval": "predictions_eval.csv",
                    "predictions_test": "predictions_test.csv" if test_best_metrics is not None else None,
                    "train_metrics": train_best_metrics,
                    "validation_metrics": validation_applied_metrics,
                    "test_metrics": test_best_metrics,
                },
                "tau050": {
                    "applied_threshold": float(FIXED_PREDICTION_TAU),
                    "predictions_train": "predictions_train_tau050.csv",
                    "predictions_eval": "predictions_eval_tau050.csv",
                    "predictions_test": (
                        "predictions_test_tau050.csv" if test_tau050_metrics is not None else None
                    ),
                    "train_metrics": train_tau050_metrics,
                    "validation_metrics": validation_tau050_metrics,
                    "test_metrics": test_tau050_metrics,
                },
            },
            "train_metrics": train_best_metrics,
            "validation_metrics": validation_applied_metrics,
            "validation_metrics_at_best_threshold": validation_best_metrics,
            "eval_metrics": validation_applied_metrics,
            "test_metrics": test_best_metrics,
            "train_metrics_tau050": train_tau050_metrics,
            "validation_metrics_tau050": validation_tau050_metrics,
            "test_metrics_tau050": test_tau050_metrics,
        }

        with open(out_dir / "metric_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved: {out_dir}")
        _pred_parts = [str(out_dir / "predictions_train.csv"), str(out_dir / "predictions_eval.csv")]
        _pred_tau050 = [
            str(out_dir / "predictions_train_tau050.csv"),
            str(out_dir / "predictions_eval_tau050.csv"),
        ]
        if raw_test_preds_df is not None:
            _pred_parts.append(str(out_dir / "predictions_test.csv"))
            _pred_tau050.append(str(out_dir / "predictions_test_tau050.csv"))
        print(f"Predictions (val-tuned τ={applied_t:.4f}): {'  '.join(_pred_parts)}")
        print(f"Predictions (fixed τ={FIXED_PREDICTION_TAU:.1f}): {'  '.join(_pred_tau050)}")
        if train_on_train_val:
            print(
                f"Validation val-tuned (applied thr={applied_t:.2f}; "
                f"diagnostic best_val_t={best_val_t:.2f}): "
                f"acc={validation_applied_metrics['accuracy']:.4f} "
                f"f1_macro={validation_applied_metrics['f1_macro']:.4f} "
                f"auc={validation_applied_metrics['auc']:.4f}"
            )
        else:
            print(
                f"Validation val-tuned (thr={applied_t:.2f}): "
                f"acc={validation_applied_metrics['accuracy']:.4f} "
                f"f1_macro={validation_applied_metrics['f1_macro']:.4f} "
                f"auc={validation_applied_metrics['auc']:.4f}"
            )
        print(
            f"Validation tau050 (thr={FIXED_PREDICTION_TAU:.1f}): "
            f"acc={validation_tau050_metrics['accuracy']:.4f} "
            f"f1_macro={validation_tau050_metrics['f1_macro']:.4f} "
            f"auc={validation_tau050_metrics['auc']:.4f}"
        )
        if test_best_metrics is not None and test_tau050_metrics is not None:
            _thr_src = "fixed 0.5 (train+val)" if train_on_train_val else "from validation"
            print(
                f"Test val-tuned (threshold={applied_t:.2f} {_thr_src}): "
                f"acc={test_best_metrics['accuracy']:.4f} "
                f"f1_macro={test_best_metrics['f1_macro']:.4f} auc={test_best_metrics['auc']:.4f}"
            )
            print(
                f"Test tau050 (threshold={FIXED_PREDICTION_TAU:.1f}): "
                f"acc={test_tau050_metrics['accuracy']:.4f} "
                f"f1_macro={test_tau050_metrics['f1_macro']:.4f} auc={test_tau050_metrics['auc']:.4f}"
            )
        print(
            f"⏱ Total pipeline wall: {total_pipeline_wall_sec:.1f}s  "
            f"(train {train_wall_sec:.1f}s + post-hoc preds {posthoc_inference_wall_sec:.1f}s + setup {setup_wall_sec:.1f}s) — "
            f"see timing in metric_summary.json for LLM comparison"
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    main()

