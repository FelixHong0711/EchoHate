#!/usr/bin/env python3
"""
Select ambiguous / hard training rows via checkpoint-probability variability.

Methodological connection (binary classification): the safety-filter style score

  σ_i = max_{y ∈ Y} Std_{e ∈ E}[ p_θ^{(e)}(y | x_i) ]

is equivalent to gold-label variability already stored in training_dynamics.csv:

  σ_i^gold = Std_e[ p_θ^{(e)}(y_i | x_i) ]   (= column ``p_gold_std``)

because p(0|x)=1-p(1|x) ⇒ Std[p(0|x)]=Std[p(1|x)].

Per-class top-25% selection (stratified; avoids majority-class domination):

  Q_c = Top_25%{ σ_i : y_i = c },   Q = ⋃_{c ∈ {0,1}} Q_c

Writes under ``<run_dir>/selection_variability_top25/``:
  - selected_hard_rows.csv
  - selected_hard_rows_for_inference.csv
  - selection_summary.json
  - scored_train_rows.csv  (all train rows + selected flag)

Also copies the selected_hard CSVs into ``<run_dir>/selection_gmm/`` for toy_prompt / adapt.
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

SELECTION_SUBDIR = "selection_variability_top25"  # default for top_frac=0.25
MIRROR_SUBDIR = "selection_gmm"
DEFAULT_TOP_FRAC = 0.25


def subdir_for_top_frac(top_frac: float) -> str:
    """``0.25`` → ``selection_variability_top25``; ``0.5`` → ``…_top50``."""
    pct = int(round(float(top_frac) * 100.0))
    if pct < 1 or pct > 100:
        raise ValueError(f"top_frac pct out of range: {pct} (from {top_frac})")
    return f"selection_variability_top{pct}"


DEFAULT_BASELINES = (
    "roberta_large_baseline_wCL",
    "roberta_large_baseline_woCL",
    "bert_large_uncased_baseline_wCL",
    "bert_large_uncased_baseline_woCL",
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


def _ceil_top_k(n: int, frac: float) -> int:
    if n <= 0:
        return 0
    return max(1, int(math.ceil(float(frac) * float(n))))


def select_top_variability_per_class(
    dynamics: pd.DataFrame,
    *,
    top_frac: float = DEFAULT_TOP_FRAC,
    std_col: str = "p_gold_std",
    label_col: str = "gold_label",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Return (selected_rows, scored_full_table, summary).

    Within each gold class, keep the top ``top_frac`` by ``std_col`` (highest first).
    Ties beyond k are broken by ``nlargest`` (stable by pandas index order).
    """
    if dynamics.empty:
        raise ValueError("empty training_dynamics")
    need = {"row_id", std_col, label_col}
    missing = need - set(dynamics.columns)
    if missing:
        raise ValueError(f"training_dynamics missing columns: {sorted(missing)}")

    df = dynamics.copy()
    df[label_col] = df[label_col].astype(float).round().astype(int)
    df[std_col] = df[std_col].astype(float).fillna(0.0)
    df["variability_sigma"] = df[std_col]
    df["selected_ambiguous"] = False
    df["selected_rank_in_class"] = pd.NA
    df["class_sigma_threshold"] = pd.NA

    per_class: Dict[str, Any] = {}
    selected_parts: List[pd.DataFrame] = []
    for c in sorted(df[label_col].unique().tolist()):
        sub = df[df[label_col] == c]
        n = len(sub)
        k = _ceil_top_k(n, top_frac)
        if k <= 0:
            per_class[str(c)] = {"n": int(n), "k": 0, "threshold_p_gold_std": None}
            continue
        top = sub.nlargest(k, std_col, keep="first").copy()
        thr = float(top[std_col].min()) if len(top) else None
        top = top.sort_values(std_col, ascending=False, kind="mergesort")
        top["selected_rank_in_class"] = list(range(1, len(top) + 1))
        top["class_sigma_threshold"] = thr
        top["selected_ambiguous"] = True
        selected_parts.append(top)
        per_class[str(c)] = {
            "n": int(n),
            "k": int(k),
            "top_frac": float(top_frac),
            "threshold_p_gold_std": thr,
            "selected_sigma_mean": float(top[std_col].mean()) if len(top) else None,
            "selected_sigma_min": float(top[std_col].min()) if len(top) else None,
            "selected_sigma_max": float(top[std_col].max()) if len(top) else None,
        }
        df.loc[top.index, "selected_ambiguous"] = True
        df.loc[top.index, "selected_rank_in_class"] = top["selected_rank_in_class"].to_numpy()
        df.loc[top.index, "class_sigma_threshold"] = thr

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else df.iloc[0:0].copy()
    )
    summary = {
        "method": "variability_top_frac_per_class",
        "equivalence_note": (
            "Binary: max_y Std[p(y|x)] == Std[p(y_gold|x)] because p(0)=1-p(1); "
            "uses training_dynamics.p_gold_std."
        ),
        "top_frac": float(top_frac),
        "std_col": std_col,
        "label_col": label_col,
        "n_train": int(len(df)),
        "n_selected": int(len(selected)),
        "selected_frac_overall": (float(len(selected)) / float(len(df))) if len(df) else 0.0,
        "per_class": per_class,
        "formula": {
            "sigma_i": "Std_e[p_theta^(e)(y_i | x_i)]",
            "Q_c": f"Top_{int(round(100 * top_frac))}%{{sigma_i : y_i = c}}",
            "Q": "union_c Q_c",
        },
    }
    return selected, df, summary


def _hard_train_for_toy_prompt(selected: pd.DataFrame, train_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if "text" in selected.columns and "class" in selected.columns:
        frame = selected
    elif train_csv is not None and train_csv.is_file():
        from pipeline.stage1.selection_balanced_random import load_train_frame

        train_df = load_train_frame(train_csv)
        ids = set(selected["row_id"].astype(str).tolist())
        frame = train_df[train_df["row_id"].astype(str).isin(ids)].copy()
        if frame.empty:
            return None
    else:
        return None
    cols = ["row_id", "text", "class"]
    hard = frame[cols].drop_duplicates(subset=["row_id"]).copy()
    hard["gold_label"] = hard["class"].astype(int)
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
            "p_gold_mean",
            "p_gold_std",
            "variability_sigma",
            "selected_rank_in_class",
            "class_sigma_threshold",
            "loss_mean",
            "conf_error_count",
            "epochs_seen",
            "text",
            "class",
        )
        if c in selected.columns
    ]
    hard = selected[hard_cols].drop_duplicates(subset=["row_id"]).copy()
    hard_path = out_dir / "selected_hard_rows.csv"
    hard.to_csv(hard_path, index=False)

    if input_csv is not None and input_csv.is_file():
        infer = map_row_ids_for_inference(hard[["row_id"]], input_csv)
    else:
        infer = hard[["row_id"]].drop_duplicates()
    infer_path = out_dir / "selected_hard_rows_for_inference.csv"
    infer.to_csv(infer_path, index=False)

    toy_hard = _hard_train_for_toy_prompt(selected, input_csv)
    toy_path: Optional[Path] = None
    if toy_hard is not None:
        toy_path = out_dir / "hard_train_for_toy_prompt.csv"
        toy_hard.to_csv(toy_path, index=False)

    summary = dict(summary)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["run_dir"] = str(run_dir)
    summary["selection_subdir"] = selection_subdir
    summary["paths"] = {
        "scored_train_rows": str(scored_path),
        "selected_hard_rows": str(hard_path),
        "selected_hard_rows_for_inference": str(infer_path),
    }
    if toy_path is not None:
        summary["paths"]["hard_train_for_toy_prompt"] = str(toy_path)
    summary_path = out_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    written = {
        "scored": scored_path,
        "hard": hard_path,
        "infer": infer_path,
        "summary": summary_path,
    }
    if toy_path is not None:
        written["toy_hard"] = toy_path

    if mirror_to_selection_gmm:
        mirror = run_dir / MIRROR_SUBDIR
        mirror.mkdir(parents=True, exist_ok=True)
        hard.to_csv(mirror / "selected_hard_rows.csv", index=False)
        infer.to_csv(mirror / "selected_hard_rows_for_inference.csv", index=False)
        if toy_hard is not None:
            toy_hard.to_csv(mirror / "hard_train_for_toy_prompt.csv", index=False)
        (mirror / "selection_method.json").write_text(
            json.dumps(
                {
                    "method": "variability_top_frac_per_class",
                    "source_subdir": selection_subdir,
                    "top_frac": summary.get("top_frac"),
                    "n_selected": summary.get("n_selected"),
                    "note": (
                        f"Mirrored from {selection_subdir} for LLM_lora_hard loaders "
                        "that resolve hard ids under selection_gmm/."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written["mirror_hard"] = mirror / "selected_hard_rows.csv"
        written["mirror_infer"] = mirror / "selected_hard_rows_for_inference.csv"

    return written


def process_run_dir(
    run_dir: Path,
    *,
    top_frac: float = DEFAULT_TOP_FRAC,
    input_csv: Optional[Path] = None,
    mirror_to_selection_gmm: bool = True,
    selection_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    dyn_path = run_dir / "training_dynamics.csv"
    if not dyn_path.is_file():
        raise FileNotFoundError(f"missing {dyn_path}")
    dynamics = pd.read_csv(dyn_path)
    selected, scored, summary = select_top_variability_per_class(
        dynamics, top_frac=top_frac
    )
    subdir = selection_subdir or subdir_for_top_frac(top_frac)
    written = _write_selection(
        run_dir,
        scored,
        selected,
        summary,
        selection_subdir=subdir,
        input_csv=input_csv,
        mirror_to_selection_gmm=mirror_to_selection_gmm,
    )
    return {
        "run_dir": str(run_dir),
        "selection_subdir": subdir,
        "n_train": int(summary["n_train"]),
        "n_selected": int(summary["n_selected"]),
        "per_class": summary["per_class"],
        "written": {k: str(v) for k, v in written.items()},
    }


def _default_train_csv(data_root: Path, seed_dir: str, dataset: str) -> Path:
    return (
        Path(data_root)
        / seed_dir
        / dataset
        / "splitted_data_80_10_10"
        / "train.csv"
    )


def process_seed_baselines(
    *,
    data_root: Path,
    seed_dir: str,
    datasets: Sequence[str],
    baselines: Sequence[str],
    top_frac: float,
    mirror_to_selection_gmm: bool = True,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    root = Path(data_root)
    for ds in datasets:
        train_csv = _default_train_csv(root, seed_dir, ds)
        for bl in baselines:
            run_dir = root / seed_dir / ds / bl
            if not (run_dir / "training_dynamics.csv").is_file():
                results.append(
                    {
                        "run_dir": str(run_dir),
                        "skipped": True,
                        "reason": "missing training_dynamics.csv",
                    }
                )
                continue
            info = process_run_dir(
                run_dir,
                top_frac=top_frac,
                input_csv=train_csv if train_csv.is_file() else None,
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
                f"(c0 {c0.get('k')}/{c0.get('n')}, c1 {c1.get('k')}/{c1.get('n')})"
            )
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Extract top-25% (default) ambiguous train rows per gold class using "
            "p_gold_std variability (binary-equivalent max_y Std[p(y|x)])."
        )
    )
    p.add_argument(
        "--run-dir",
        default=None,
        help="Single baseline run dir containing training_dynamics.csv",
    )
    p.add_argument(
        "--data-root",
        default="data",
        help="EchoHate data root (default: data)",
    )
    p.add_argument(
        "--seed-dir",
        default="eacl_seed0",
        help="Seed bucket under data-root (default: eacl_seed0)",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma list of datasets (batch mode when --run-dir omitted)",
    )
    p.add_argument(
        "--baselines",
        default=",".join(DEFAULT_BASELINES),
        help="Comma list of baseline run folder names",
    )
    p.add_argument(
        "--top-frac",
        type=float,
        default=DEFAULT_TOP_FRAC,
        help="Per-class top fraction by variability (default 0.25)",
    )
    p.add_argument(
        "--input-csv",
        default=None,
        help="Train CSV for row_id remap (single --run-dir mode)",
    )
    p.add_argument(
        "--no-mirror-selection-gmm",
        action="store_true",
        help="Do not copy selected_hard CSVs into selection_gmm/",
    )
    args = p.parse_args()
    if not (0.0 < float(args.top_frac) <= 1.0):
        raise SystemExit(f"--top-frac must be in (0,1], got {args.top_frac}")

    mirror = not bool(args.no_mirror_selection_gmm)
    if args.run_dir:
        info = process_run_dir(
            Path(args.run_dir),
            top_frac=float(args.top_frac),
            input_csv=Path(args.input_csv) if args.input_csv else None,
            mirror_to_selection_gmm=mirror,
        )
        print(json.dumps(info, indent=2))
        return

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    baselines = [x.strip() for x in str(args.baselines).split(",") if x.strip()]
    results = process_seed_baselines(
        data_root=Path(args.data_root),
        seed_dir=str(args.seed_dir),
        datasets=datasets,
        baselines=baselines,
        top_frac=float(args.top_frac),
        mirror_to_selection_gmm=mirror,
    )
    ok = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    print(
        f"Done. processed={len(ok)} skipped={len(skipped)} "
        f"top_frac={args.top_frac} mirror_gmm={mirror}"
    )
    if skipped:
        for r in skipped:
            print(f"  skip {r['run_dir']}: {r.get('reason')}")


if __name__ == "__main__":
    main()
