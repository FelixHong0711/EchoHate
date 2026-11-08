#!/usr/bin/env python3
"""
Union unique hard rows from baseline selection dirs across seeds (and baselines).

Use when each seed (and wCL vs woCL) has a different selection set but train splits /
row_ids are shared: run LLM hard GT infer **once** on the union, then map preds back
per baseline with ``distribute_baseline_union_llm_preds``.

Writes under the anchor baseline run::

    data/<anchor_seed>/<DS>/<anchor_baseline>/<selection_subdir>_cross_union/
        selected_hard_rows.csv
        selected_hard_rows_for_inference.csv
        hard_train_for_toy_prompt.csv
        union_summary.json

Examples::

    # Classic var25 (default)
    python -m pipeline.stage1.union_baseline_var25_selections \\
      --dataset SBIC \\
      --seed-dirs seed_0_mech,seed_1_mech,seed_42_mech \\
      --baselines roberta_large_baseline_wCL \\
      --anchor-seed-dir seed_0_mech

    # rand25 / var50 / rand50
    python -m pipeline.stage1.union_baseline_var25_selections \\
      --dataset IHC --selection-subdir selection_balanced_random25 \\
      --seed-dirs seed_0_mech,seed_1_mech,seed_42_mech \\
      --baselines roberta_large_baseline_wCL
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pipeline.stage1.selection_balanced_random import load_train_frame
from pipeline.stage1.selection_row_ids import map_row_ids_for_inference
from pipeline.stage1.select_hard_cases_for_mech import _hard_train_for_toy_prompt

DEFAULT_BASELINES = (
    "roberta_large_baseline_wCL",
    "roberta_large_baseline_woCL",
)
DEFAULT_SEED_DIRS = (
    "seed_0_mech",
    "seed_1_mech",
    "seed_42_mech",
    "seed_123_mech",
    "seed_1234_mech",
)
DEFAULT_DATASETS = (
    "IHC",
    "DynaHate",
    "HatEval",
    "Ethos",
    "drop_offensive",
    "toxigen_new",
    "SBIC",
)
DEFAULT_SEL_SUBDIR = "selection_variability_top25"


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def out_subdir_for(selection_subdir: str, out_subdir: Optional[str] = None) -> str:
    if out_subdir and str(out_subdir).strip():
        return str(out_subdir).strip()
    s = str(selection_subdir).strip().rstrip("/")
    if s.endswith("_cross_union"):
        return s
    return f"{s}_cross_union"


def _discover_selection_csvs(
    data_root: Path,
    *,
    dataset: str,
    seed_dirs: Sequence[str],
    baselines: Sequence[str],
    selection_subdir: str,
) -> List[Tuple[Path, str]]:
    """Return (csv_path, source_label) for each existing selection."""
    found: List[Tuple[Path, str]] = []
    for seed in seed_dirs:
        for bl in baselines:
            p = (
                data_root
                / seed
                / dataset
                / bl
                / selection_subdir
                / "selected_hard_rows.csv"
            )
            if p.is_file():
                found.append((p, f"{seed}/{bl}"))
    return found


def _union_frames(
    items: Sequence[Tuple[Path, str]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames: List[pd.DataFrame] = []
    per_src: Dict[str, int] = {}
    for p, label in items:
        df = pd.read_csv(p)
        if "row_id" not in df.columns:
            raise ValueError(f"missing row_id in {p}")
        df = df.copy()
        df["row_id"] = df["row_id"].astype(str)
        df["__src"] = label
        per_src[label] = len(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), {"n_sources": 0, "per_source": {}, "n_union": 0}

    all_df = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("p_gold_std", "variability_sigma") if c in all_df.columns]
    if sort_cols:
        all_df = all_df.sort_values(sort_cols, ascending=False, kind="mergesort")
    src_map = (
        all_df.groupby("row_id", sort=False)["__src"]
        .agg(lambda s: sorted(set(s.tolist())))
        .to_dict()
    )
    uniq = all_df.drop_duplicates(subset=["row_id"], keep="first").copy()
    uniq["source_baselines"] = uniq["row_id"].map(lambda r: "|".join(src_map.get(r, [])))
    uniq["n_source_baselines"] = uniq["row_id"].map(lambda r: len(src_map.get(r, [])))
    uniq = uniq.drop(columns=["__src"], errors="ignore")
    summary = {
        "n_sources": len(items),
        "per_source": per_src,
        "n_rows_concat": int(len(all_df)),
        "n_union": int(len(uniq)),
        "n_duplicates_removed": int(len(all_df) - len(uniq)),
    }
    return uniq, summary


def process_union(
    *,
    data_root: Path,
    dataset: str,
    seed_dirs: Sequence[str],
    baselines: Sequence[str],
    anchor_seed_dir: str,
    anchor_baseline: str,
    split_subdir: str,
    selection_subdir: str,
    out_subdir: str,
) -> Dict[str, Any]:
    items = _discover_selection_csvs(
        data_root,
        dataset=dataset,
        seed_dirs=seed_dirs,
        baselines=baselines,
        selection_subdir=selection_subdir,
    )
    anchor_run = data_root / anchor_seed_dir / dataset / anchor_baseline
    out_dir = anchor_run / out_subdir
    train_csv = data_root / anchor_seed_dir / dataset / split_subdir / "train.csv"

    if not items:
        return {
            "dataset": dataset,
            "skipped": True,
            "reason": f"no {selection_subdir}/selected_hard_rows.csv found",
            "anchor_run": str(anchor_run),
        }
    if not train_csv.is_file():
        return {
            "dataset": dataset,
            "skipped": True,
            "reason": f"missing train.csv: {train_csv}",
        }

    union, usum = _union_frames(items)
    train = load_train_frame(train_csv)
    merged = union[["row_id"]].merge(train, on="row_id", how="inner", suffixes=("", "_train"))
    for c in (
        "p_gold_std",
        "variability_sigma",
        "p_gold_mean",
        "source_baselines",
        "n_source_baselines",
    ):
        if c in union.columns:
            merged = merged.merge(union[["row_id", c]], on="row_id", how="left")

    infer = map_row_ids_for_inference(merged[["row_id"]], train_csv)
    toy = _hard_train_for_toy_prompt(merged)

    out_dir.mkdir(parents=True, exist_ok=True)
    hard_path = out_dir / "selected_hard_rows.csv"
    infer_path = out_dir / "selected_hard_rows_for_inference.csv"
    toy_path = out_dir / "hard_train_for_toy_prompt.csv"
    merged.to_csv(hard_path, index=False)
    infer.to_csv(infer_path, index=False)
    toy.to_csv(toy_path, index=False)

    summary: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": f"union_baseline_{selection_subdir}",
        "selection_subdir": selection_subdir,
        "dataset": dataset,
        "anchor_seed_dir": anchor_seed_dir,
        "anchor_baseline": anchor_baseline,
        "anchor_run": str(anchor_run),
        "out_dir": str(out_dir),
        "seed_dirs": list(seed_dirs),
        "baselines": list(baselines),
        "source_csvs": [{"path": str(p), "label": lbl} for p, lbl in items],
        **usum,
        "n_after_train_join": int(len(merged)),
        "dropped_missing_train": int(usum["n_union"] - len(merged)),
    }
    (out_dir / "union_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--dataset",
        default=None,
        help="Single dataset (omit with --datasets for batch)",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help=f"Comma list (default when neither set: {','.join(DEFAULT_DATASETS)})",
    )
    p.add_argument(
        "--seed-dirs",
        default=",".join(DEFAULT_SEED_DIRS),
        help="Comma list of seed dirs to union (default: 5 mech seeds)",
    )
    p.add_argument(
        "--baselines",
        default=",".join(DEFAULT_BASELINES),
        help="Comma list of baseline run dir names",
    )
    p.add_argument("--anchor-seed-dir", default="seed_0_mech")
    p.add_argument("--anchor-baseline", default="roberta_large_baseline_wCL")
    p.add_argument("--split-subdir", default="splitted_data_80_10_10")
    p.add_argument(
        "--selection-subdir",
        default=DEFAULT_SEL_SUBDIR,
        help="Per-run selection folder to union (default: selection_variability_top25)",
    )
    p.add_argument(
        "--out-subdir",
        default=None,
        help="Anchor output folder (default: <selection-subdir>_cross_union)",
    )
    args = p.parse_args(argv)

    if args.dataset and args.datasets:
        raise SystemExit("pass only one of --dataset / --datasets")
    if args.dataset:
        datasets = [args.dataset.strip()]
    elif args.datasets:
        datasets = _parse_csv_list(args.datasets)
    else:
        datasets = list(DEFAULT_DATASETS)

    sel = str(args.selection_subdir).strip()
    out = out_subdir_for(sel, args.out_subdir)
    seed_dirs = _parse_csv_list(args.seed_dirs)
    baselines = _parse_csv_list(args.baselines)

    n_ok = 0
    n_skip = 0
    for ds in datasets:
        info = process_union(
            data_root=args.data_root,
            dataset=ds,
            seed_dirs=seed_dirs,
            baselines=baselines,
            anchor_seed_dir=args.anchor_seed_dir.strip(),
            anchor_baseline=args.anchor_baseline.strip(),
            split_subdir=args.split_subdir,
            selection_subdir=sel,
            out_subdir=out,
        )
        if info.get("skipped"):
            print(f"[skip] {info.get('dataset')}: {info.get('reason')}")
            n_skip += 1
            continue
        print(
            f"[ok] {info['dataset']}: union={info['n_union']} "
            f"from {info['n_sources']} sources "
            f"(removed {info['n_duplicates_removed']} dups) → {info['out_dir']}"
        )
        n_ok += 1
    print(f"done: ok={n_ok} skipped={n_skip} selection={sel} out={out}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
