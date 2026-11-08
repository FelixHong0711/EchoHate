"""Per-epoch training dynamics (same schema as roberta_baseline --track-dynamics)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainerCallback


def dynamics_meta_from_train_df(df: pd.DataFrame, *, label_col: str) -> pd.DataFrame:
    """Row metadata merged into dynamics outputs (row_id, label, text, class)."""
    out = df.copy()
    if label_col not in out.columns:
        raise KeyError(f"dynamics_meta: missing label column {label_col!r}")
    out["label"] = out[label_col].astype(int)
    if "class" not in out.columns:
        out["class"] = out["label"]
    if "text" not in out.columns:
        out["text"] = ""
    return out[["row_id", "label", "text", "class"]].copy()


def collect_binary_train_predictions(
    model: nn.Module,
    dataset: Any,
    collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
    device: str,
    batch_size: int,
    row_ids: Sequence[str],
    *,
    label_key: str = "labels",
) -> pd.DataFrame:
    """Batched forward pass; returns row_id, label, prob, pred, correct, sample_loss."""
    if len(dataset) != len(row_ids):
        raise ValueError(
            f"collect_binary_train_predictions: len(dataset)={len(dataset)} != len(row_ids)={len(row_ids)}"
        )
    was_training = bool(model.training)
    model.eval()
    model.to(device)
    try:
        all_row_ids: List[str] = []
        all_probs: List[float] = []
        all_preds: List[int] = []
        all_labels: List[int] = []
        for i in range(0, len(dataset), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            batch_data = [dataset[j] for j in batch_indices]
            batch = collate_fn(batch_data)
            labels_t = batch.pop(label_key)
            labels_list = labels_t.detach().cpu().numpy().tolist()
            if isinstance(labels_list, (float, int)):
                labels_list = [labels_list]
            batch_tensors = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            with torch.no_grad():
                outputs = model(**batch_tensors)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                bs = len(batch_indices)
                if logits.dim() == 2 and logits.shape[-1] == 2:
                    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(int)
                else:
                    logits = logits.squeeze(-1)
                    probs = torch.sigmoid(logits).detach().cpu().numpy()
                    preds = (np.atleast_1d(probs) >= 0.5).astype(int)
            all_row_ids.extend([str(row_ids[j]) for j in batch_indices])
            all_probs.extend(np.atleast_1d(probs).tolist())
            all_preds.extend(np.atleast_1d(preds).tolist())
            all_labels.extend([int(x) for x in np.atleast_1d(labels_list).tolist()])
        probs_arr = np.asarray(all_probs, dtype=np.float64)
        preds_df = pd.DataFrame(
            {
                "row_id": all_row_ids,
                "label": all_labels,
                "prob": probs_arr,
                "pred": all_preds,
                "correct": [int(p == l) for p, l in zip(all_preds, all_labels)],
            }
        )
        y = preds_df["label"].astype(float).to_numpy()
        p = np.clip(probs_arr, 1e-7, 1.0 - 1e-7)
        preds_df["sample_loss"] = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return preds_df
    finally:
        model.train(was_training)


class TrainingDynamicsCallback(TrainerCallback):
    """Record per-epoch train predictions → training_dynamics.csv (baseline-compatible)."""

    def __init__(
        self,
        train_dataset: Any,
        train_df: pd.DataFrame,
        collate_fn: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
        eval_batch_size: int,
        device: str,
        output_dir: Path,
        *,
        label_col: str = "binary_label",
    ) -> None:
        self.train_dataset = train_dataset
        meta = dynamics_meta_from_train_df(train_df, label_col=label_col)
        self.train_df = meta
        self.row_ids = meta["row_id"].astype(str).tolist()
        self.collate_fn = collate_fn
        self.eval_batch_size = int(eval_batch_size)
        self.device = device
        self.output_dir = Path(output_dir)
        self.records: List[pd.DataFrame] = []
        self.dynamics_dir = self.output_dir / "training_dynamics_epochs"
        self.dynamics_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            model.train()
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        local_rank = getattr(args, "local_rank", -1)
        if local_rank is None:
            local_rank = -1
        if int(local_rank) > 0:
            return control
        epoch_num = int(round(float(state.epoch or 0.0)))
        if epoch_num <= 0:
            return control
        epoch_df = collect_binary_train_predictions(
            model,
            self.train_dataset,
            self.collate_fn,
            self.device,
            self.eval_batch_size,
            self.row_ids,
        )
        epoch_df = epoch_df.merge(self.train_df, on="row_id", how="left", suffixes=("", "_meta"))
        if "label_meta" in epoch_df.columns:
            epoch_df["label"] = epoch_df["label"].fillna(epoch_df["label_meta"]).astype(int)
            epoch_df = epoch_df.drop(columns=["label_meta"])
        epoch_df["epoch"] = epoch_num
        epoch_df["p_gold"] = np.where(
            epoch_df["label"].astype(int).to_numpy() == 1,
            epoch_df["prob"].astype(float).to_numpy(),
            1.0 - epoch_df["prob"].astype(float).to_numpy(),
        )
        epoch_df["conf_error_flag"] = (
            (epoch_df["pred"].astype(int) != epoch_df["label"].astype(int))
            & (
                np.maximum(
                    epoch_df["prob"].astype(float).to_numpy(),
                    1.0 - epoch_df["prob"].astype(float).to_numpy(),
                )
                >= 0.8
            )
        ).astype(int)
        keep_cols = [
            "epoch",
            "row_id",
            "label",
            "prob",
            "pred",
            "correct",
            "sample_loss",
            "p_gold",
            "conf_error_flag",
        ]
        epoch_df = epoch_df[keep_cols].copy()
        epoch_df.to_csv(self.dynamics_dir / f"epoch_{epoch_num:02d}_train_dynamics.csv", index=False)
        self.records.append(epoch_df)
        return control

    def finalize(self) -> Optional[pd.DataFrame]:
        if not self.records:
            return None
        all_epochs = pd.concat(self.records, ignore_index=True)
        agg = (
            all_epochs.groupby("row_id", as_index=False)
            .agg(
                gold_label=("label", "first"),
                p_gold_mean=("p_gold", "mean"),
                p_gold_std=("p_gold", "std"),
                loss_mean=("sample_loss", "mean"),
                loss_std=("sample_loss", "std"),
                conf_error_count=("conf_error_flag", "sum"),
                epochs_seen=("epoch", "nunique"),
            )
            .fillna({"p_gold_std": 0.0, "loss_std": 0.0})
        )
        full = agg.merge(self.train_df[["row_id", "text", "class"]], on="row_id", how="left")
        all_epochs.to_csv(self.output_dir / "training_dynamics_all_epochs.csv", index=False)
        full.to_csv(self.output_dir / "training_dynamics.csv", index=False)
        return full
