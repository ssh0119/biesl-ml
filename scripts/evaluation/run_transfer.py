#!/usr/bin/env python
"""Run the three-way (local-only / WESAD-only / combined) comparison.

Usage:
    uv run python scripts/evaluation/run_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transfer import run_three_way   # noqa: E402


if __name__ == "__main__":
    run_three_way()

