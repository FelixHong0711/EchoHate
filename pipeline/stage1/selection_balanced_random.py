#!/usr/bin/env python3
"""
Balanced random hard-case selection: stratified sample of train rows per gold class.

Per class c, sample ``sample_frac`` of rows (default 50%) with a fixed RNG seed.
Does **not** require training_dynamics.csv — only ``splitted_data_80_10_10/train.csv``.

Writes under ``<run_dir>/selection_balanced_random50/``:
  - selected_hard_rows.csv
  - selected_hard_rows_for_inference.csv
  - hard_train_for_toy_prompt.csv
  - scored_train_rows.csv
  - selection_summary.json

Copies the selected CSVs into ``<run_dir>/selection_gmm/`` for toy_prompt / adapt.

Random state defaults to ``training_random_state`` in ``data/<seed-dir>/seed_manifest.json``
(e.g. ``eacl_seed0_seq`` → 0 unless overridden).

Usage::

  python -m pipeline.stage1.selection_balanced_random \\
    --seed-dir eacl_seed0_seq \\
    --baselines roberta_large_baseline_wCL,roberta_large_baseline_woCL

  python -m pipeline.stage1.selection_balanced_random \\
    --run-dir data/eacl_seed0_seq/HatEval/roberta_large_baseline_wCL \\
    --input-csv data/eacl_seed0_seq/HatEval/splitted_data_80_10_10/train.csv
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from pipeline.stage1.selection_row_ids import map_row_ids_for_inference

SELECTION_SUBDIR = "selection_balanced_random50"  # default for sample_frac=0.5
MIRROR_SUBDIR = "selection_gmm"
DEFAULT_SAMPLE_FRAC = 0.5


def subdir_for_sample_frac(sample_frac: float) -> str:
    """``0.5`` → ``selection_balanced_random50``; ``0.25`` → ``…_random25``."""
    pct = int(round(float(sample_frac) * 100.0))
    if pct < 1 or pct > 100:
        raise ValueError(f"sample_frac pct out of range: {pct} (from {sample_frac})")
    return f"selection_balanced_random{pct}"


DEFAULT_BASELINES = (
    "roberta_large_baseline_wCL",
    "roberta_large_baseline_woCL",
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


def _ceil_sample_k(n: int, frac: float) -> int:
    if n <= 0:
        return 0
    return max(1, int(math.ceil(float(frac) * float(n))))


def resolve_training_random_state(
    data_root: Path,
    seed_dir: str,
    override: Optional[int],
) -> int:
    if override is not None:
        return int(override)
    manifest = Path(data_root) / seed_dir / "seed_manifest.json"
    if manifest.is_file():
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        if "training_random_state" in doc:
            return int(doc["training_random_state"])
    return 0


def load_train_frame(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    if "text" not in df.columns or "class" not in df.columns:
        raise ValueError(f"{train_csv} must have text, class columns; got {list(df.columns)}")
    out = df.copy()
    out["gold_label"] = out["class"].astype(int)
    if "row_id" not in out.columns or out["row_id"].isna().any():
        out["row_id"] = [f"train_{i}" for i in range(len(out))]
    else:
        out["row_id"] = out["row_id"].astype(str)
    return out


def select_balanced_random(
    train_df: pd.DataFrame,
    *,
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
    random_state: int = 0,
    label_col: str = "gold_label",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Stratified random sample per class. Returns (selected, scored_full, summary)."""
    if train_df.empty:
        raise ValueError("empty train dataframe")
    if label_col not in train_df.columns:
        raise ValueError(f"missing {label_col}")

    df = train_df.copy()
    df[label_col] = df[label_col].astype(int)
    df["selected_random"] = False
    df["selected_rank_in_class"] = pd.NA

    per_class: Dict[str, Any] = {}
    selected_parts: List[pd.DataFrame] = []
    for c in sorted(df[label_col].unique().tolist()):
        sub = df[df[label_col] == c]
        n = len(sub)
        k = _ceil_sample_k(n, sample_frac)
        if k <= 0:
            per_class[str(c)] = {"n": int(n), "k": 0, "sample_frac": float(sample_frac)}
            continue
        # Per-class seed offset for stable, reproducible draws.
        rs = int(random_state) + int(c) * 1_000_003
        picked = sub.sample(n=min(k, n), random_state=rs, replace=False).copy()
        picked = picked.sort_index()
        picked["selected_rank_in_class"] = list(range(1, len(picked) + 1))
        picked["selected_random"] = True
        selected_parts.append(picked)
        per_class[str(c)] = {
            "n": int(n),
            "k": int(len(picked)),
            "sample_frac": float(sample_frac),
            "random_state": int(rs),
        }
        df.loc[picked.index, "selected_random"] = True
        df.loc[picked.index, "selected_rank_in_class"] = picked["selected_rank_in_class"].to_numpy()

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else df.iloc[0:0].copy()
    )
    summary = {
        "method": "balanced_random_frac_per_class",
        "sample_frac": float(sample_frac),
        "random_state_base": int(random_state),
        "label_col": label_col,
        "n_train": int(len(df)),
        "n_selected": int(len(selected)),
        "selected_frac_overall": (float(len(selected)) / float(len(df))) if len(df) else 0.0,
        "per_class": per_class,
        "formula": {
            "Q_c": f"Random sample {int(round(100 * sample_frac))}% of train rows with y=c",
            "Q": "union_c Q_c",
        },
    }
    return selected, df, summary


def _hard_train_for_toy_prompt(selected: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_id", "text", "class", "gold_label"]
    hard = selected[cols].copy()
    if "label" in selected.columns:
        hard["label"] = selected["label"]
    else:
        hard["label"] = hard["gold_label"].map({0: "not_hate", 1: "hate"}).astype(str)
    return hard


def _write_selection(
    run_dir: Path,
    scored_full: pd.DataFrame,
    selected: pd.DataFrame,
    summary: Dict[str, Any],
    *,
    selection_subdir: str,
    input_csv: Optional[Path] = None,
    mirror_to_selection_gmm: bool = True,
) -> Dict[str, Path]:
    out_dir = run_dir / selection_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_path = out_dir / "scored_train_rows.csv"
    scored_full.to_csv(scored_path, index=False)

    hard_cols = [
        c
        for c in (
            "row_id",
            "gold_label",
            "selected_rank_in_class",
            "text",
            "class",
            "label",
        )
        if c in selected.columns
    ]
    hard = selected[hard_cols].drop_duplicates(subset=["row_id"]).copy()
    hard_path = out_dir / "selected_hard_rows.csv"
    hard.to_csv(hard_path, index=False)

    toy_hard = _hard_train_for_toy_prompt(selected)
    toy_path = out_dir / "hard_train_for_toy_prompt.csv"
    toy_hard.to_csv(toy_path, index=False)

    if input_csv is not None and input_csv.is_file():
        infer = map_row_ids_for_inference(hard[["row_id"]], input_csv)
    else:
        infer = hard[["row_id"]].drop_duplicates()
    infer_path = out_dir / "selected_hard_rows_for_inference.csv"
    infer.to_csv(infer_path, index=False)

    summary = dict(summary)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["run_dir"] = str(run_dir)
    summary["selection_subdir"] = selection_subdir
    summary["paths"] = {
        "scored_train_rows": str(scored_path),
        "selected_hard_rows": str(hard_path),
        "hard_train_for_toy_prompt": str(toy_path),
        "selected_hard_rows_for_inference": str(infer_path),
    }
    summary_path = out_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    written = {
        "scored": scored_path,
        "hard": hard_path,
        "toy_hard": toy_path,
        "infer": infer_path,
        "summary": summary_path,
    }

    if mirror_to_selection_gmm:
        mirror = run_dir / MIRROR_SUBDIR
        mirror.mkdir(parents=True, exist_ok=True)
        hard.to_csv(mirror / "selected_hard_rows.csv", index=False)
        infer.to_csv(mirror / "selected_hard_rows_for_inference.csv", index=False)
        toy_hard.to_csv(mirror / "hard_train_for_toy_prompt.csv", index=False)
        (mirror / "selection_method.json").write_text(
            json.dumps(
                {
                    "method": "balanced_random_frac_per_class",
                    "source_subdir": selection_subdir,
                    "sample_frac": summary.get("sample_frac"),
                    "random_state_base": summary.get("random_state_base"),
                    "n_selected": summary.get("n_selected"),
                    "note": (
                        f"Mirrored from {selection_subdir} for toy_prompt / "
                        "mech loaders that resolve hard ids under selection_gmm/."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["mirror_hard"] = mirror / "selected_hard_rows.csv"
        written["mirror_infer"] = mirror / "selected_hard_rows_for_inference.csv"
        written["mirror_toy_hard"] = mirror / "hard_train_for_toy_prompt.csv"

    return written


def process_run_dir(
    run_dir: Path,
    train_csv: Path,
    *,
    sample_frac: float = DEFAULT_SAMPLE_FRAC,
    random_state: int = 0,
    mirror_to_selection_gmm: bool = True,
    selection_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    train_csv = Path(train_csv).expanduser().resolve()
    if not train_csv.is_file():
        raise FileNotFoundError(f"missing {train_csv}")
    train_df = load_train_frame(train_csv)
    selected, scored, summary = select_balanced_random(
        train_df,
        sample_frac=sample_frac,
        random_state=random_state,
    )
    subdir = selection_subdir or subdir_for_sample_frac(sample_frac)
    written = _write_selection(
        run_dir,
        scored,
        selected,
        summary,
        selection_subdir=subdir,
        input_csv=train_csv,
        mirror_to_selection_gmm=mirror_to_selection_gmm,
    )
    return {
        "run_dir": str(run_dir),
        "train_csv": str(train_csv),
        "selection_subdir": subdir,
        "n_train": int(summary["n_train"]),
        "n_selected": int(summary["n_selected"]),
        "per_class": summary["per_class"],
        "random_state_base": int(random_state),
        "written": {k: str(v) for k, v in written.items()},
    }


def _default_train_csv(data_root: Path, seed_dir: str, dataset: str, split_subdir: str) -> Path:
    return Path(data_root) / seed_dir / dataset / split_subdir / "train.csv"


def process_seed_baselines(
    *,
    data_root: Path,
    seed_dir: str,
    datasets: Sequence[str],
    baselines: Sequence[str],
    sample_frac: float,
    random_state: int,
    split_subdir: str = "splitted_data_80_10_10",
    mirror_to_selection_gmm: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    root = Path(data_root)
    for ds in datasets:
        train_csv = _default_train_csv(root, seed_dir, ds, split_subdir)
        if not train_csv.is_file():
            results.append(
                {
                    "dataset": ds,
                    "skipped": True,
                    "reason": f"missing {train_csv}",
                }
            )
            continue
        for bl in baselines:
            run_dir = root / seed_dir / ds / bl
            if not run_dir.is_dir():
                results.append(
                    {
                        "dataset": ds,
                        "baseline": bl,
                        "skipped": True,
                        "reason": f"missing {run_dir}",
                    }
                )
                continue
            info = process_run_dir(
                run_dir,
                train_csv,
                sample_frac=sample_frac,
                random_state=random_state,
                mirror_to_selection_gmm=mirror_to_selection_gmm,
            )
            info["dataset"] = ds
            info["baseline"] = bl
            results.append(info)
            pc = info.get("per_class") or {}
            c0 = pc.get("0", {})
            c1 = pc.get("1", {})
            print(
                f"[ok] {ds}/{bl}: selected={info['n_selected']}/{info['n_train']} "
                f"(c0 {c0.get('k')}/{c0.get('n')}, c1 {c1.get('k')}/{c1.get('n')}) "
                f"rs={random_state}"
            )
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Balanced random hard-case selection (stratified sample_frac per class)."
    )
    p.add_argument("--run-dir", default=None, help="Single baseline run dir")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed-dir", default="eacl_seed0_seq")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    p.add_argument("--split-subdir", default="splitted_data_80_10_10")
    p.add_argument(
        "--sample-frac",
        type=float,
        default=DEFAULT_SAMPLE_FRAC,
        help="Per-class random sample fraction (default 0.5)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="RNG base (default: training_random_state from seed_manifest.json)",
    )
    p.add_argument("--input-csv", default=None, help="Train CSV (single --run-dir mode)")
    p.add_argument(
        "--no-mirror-selection-gmm",
        action="store_true",
        help="Do not copy selected_hard CSVs into selection_gmm/",
    )
    args = p.parse_args()
    if not (0.0 < float(args.sample_frac) <= 1.0):
        raise SystemExit("--sample-frac must be in (0, 1]")

    data_root = Path(args.data_root)
    rs = resolve_training_random_state(data_root, args.seed_dir, args.random_state)
    mirror = not args.no_mirror_selection_gmm

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if args.input_csv:
            train_csv = Path(args.input_csv)
        else:
            parts = run_dir.parts
            try:
                idx = parts.index(args.seed_dir)
                dataset = parts[idx + 1]
            except (ValueError, IndexError):
                raise SystemExit("cannot infer dataset; pass --input-csv")
            train_csv = _default_train_csv(data_root, args.seed_dir, dataset, args.split_subdir)
        info = process_run_dir(
            run_dir,
            train_csv,
            sample_frac=float(args.sample_frac),
            random_state=rs,
            mirror_to_selection_gmm=mirror,
        )
        print(json.dumps(info, indent=2))
        return

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    print(f"seed_dir={args.seed_dir} random_state={rs} sample_frac={args.sample_frac}")
    results = process_seed_baselines(
        data_root=data_root,
        seed_dir=args.seed_dir,
        datasets=datasets,
        baselines=baselines,
        sample_frac=float(args.sample_frac),
        random_state=rs,
        split_subdir=args.split_subdir,
        mirror_to_selection_gmm=mirror,
    )
    n_ok = sum(1 for r in results if not r.get("skipped"))
    print(f"done: ok={n_ok} skipped={len(results) - n_ok}")


if __name__ == "__main__":
    main()
