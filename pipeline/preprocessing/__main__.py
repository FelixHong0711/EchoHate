#!/usr/bin/env python3
"""Unified CLI for EchoHate data preprocessing (python -m pipeline.preprocessing <cmd>)."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from typing import Callable, List, Optional


def _run_module_main(dotted: str, argv: Optional[List[str]] = None) -> None:
    mod = importlib.import_module(dotted)
    old = sys.argv
    sys.argv = [dotted.rsplit(".", 1)[-1]] + (argv or [])
    try:
        params = inspect.signature(mod.main).parameters
        if params:
            mod.main(argv)
        else:
            mod.main()
    finally:
        sys.argv = old


def _cmd_dataset(command: str, rest: List[str]) -> None:
    from pipeline.preprocessing.datasets import run_command

    run_command(command, rest)


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in ("-h", "--help"):
        p = argparse.ArgumentParser(
            description="EchoHate dataset preprocessing (7 datasets, 811 splits)."
        )
        p.add_argument("command", nargs="?", choices=sorted(_COMMAND_HELP))
        p.print_help()
        print("\nCommands:", ", ".join(sorted(_COMMAND_HELP)))
        return

    cmd, rest = argv[0], argv[1:]
    if cmd not in _DISPATCH:
        raise SystemExit(f"Unknown command {cmd!r}. Choose from: {', '.join(sorted(_DISPATCH))}")
    _DISPATCH[cmd](rest)


def _cmd_split(rest: List[str]) -> None:
    _run_module_main("pipeline.preprocessing.split", rest)


_DATASET_COMMANDS = (
    "dynahate",
    "hatexplain-random-split-variants",
    "hateval",
    "ihc",
    "ethos",
    "toxigen",
    "sbic",
)

_COMMAND_HELP = {"split", *_DATASET_COMMANDS}

_DISPATCH: dict[str, Callable[[List[str]], None]] = {
    "split": _cmd_split,
    **{cmd: (lambda r, c=cmd: _cmd_dataset(c, r)) for cmd in _DATASET_COMMANDS},
}


if __name__ == "__main__":
    main()
