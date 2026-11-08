#!/usr/bin/env python3
"""Unified CLI: python -m pipeline <stage> <module> [args…]"""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import Optional

_MODULES = {
    # preprocessing
    ("preprocessing", "split"): "pipeline.preprocessing.split",
    # stage 1 (hard selection: var25 / rand via select_hard_cases_for_mech)
    ("stage1", "select_hard_cases_for_mech"): "pipeline.stage1.select_hard_cases_for_mech",
    ("stage1", "selection_variability_top25"): "pipeline.stage1.selection_variability_top25",
    ("stage1", "selection_balanced_random"): "pipeline.stage1.selection_balanced_random",
    # stage 2 (encoder baseline, mechanism supervision, frozen-threshold eval)
    ("stage2", "roberta_baseline"): "pipeline.stage2.roberta_baseline",
    ("stage2", "mech_distill"): "pipeline.stage2.mech_distill",
    ("stage2", "run_eval_test_from_checkpoint"): "pipeline.stage2.run_eval_test_from_checkpoint",
}


def _usage() -> str:
    lines = ["python -m pipeline <stage> <module> [module args…]", ""]
    cur = None
    for (stage, name), _mod in sorted(_MODULES.items()):
        if stage != cur:
            lines.append(f"{stage}:")
            cur = stage
        lines.append(f"  {name}")
    lines.append("")
    lines.append("Preprocessing datasets: python -m pipeline.preprocessing <cmd>")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage())
        return
    if len(argv) < 2:
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)
    key = (argv[0], argv[1])
    mod = _MODULES.get(key)
    if not mod:
        print(f"Unknown: {key[0]} {key[1]}", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)
    sys.argv = [mod.replace(".", "/") + ".py"] + argv[2:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
