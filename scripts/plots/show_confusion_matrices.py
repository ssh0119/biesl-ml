#!/usr/bin/env python
"""Print and plot confusion matrices for the local-only runs.

Inputs:
    outputs/local_loro.json           ??classical LORO
    outputs/local_randomsplit.json    ??classical 70:15:15 random, 5 seeds
    outputs/dl_local_loro.json        ??1D-CNN LORO
    outputs/dl_local_randomsplit.json ??1D-CNN 70:15:15 random, 5 seeds

Outputs:
    stdout ??markdown tables (one per (protocol 횞 model)), row-normalized %
    figures/confusion/*.png ??one heatmap per matrix

Usage:
    uv run python scripts/plots/show_confusion_matrices.py > confusion-matrices.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

from src.pipeline import PHASE_CLASSES   # noqa: E402


import argparse

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--inputs-dir", default="outputs",
                     help="dir containing the four JSON files (default: outputs)")
_parser.add_argument("--figures-dir", default="figures/confusion",
                     help="dir for output PNGs (default: figures/confusion)")
_args, _ = _parser.parse_known_args()
OUT = Path(_args.inputs_dir)
FIG_DIR = Path(_args.figures_dir)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"!! missing: {path}", file=sys.stderr)
        return None
    with open(path) as fh:
        return json.load(fh)


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    rs = cm.sum(axis=1, keepdims=True)
    safe = np.where(rs == 0, 1.0, rs)
    return cm / safe * 100.0


def _sum_seed_cms_classical(per_seed: dict, model: str) -> np.ndarray:
    total = np.zeros((len(PHASE_CLASSES), len(PHASE_CLASSES)), dtype=int)
    for _, payload in per_seed.items():
        cm = np.asarray(payload["models"][model]["test"]["confusion"], dtype=int)
        total += cm
    return total


def _sum_seed_cms_dl(per_seed: dict) -> np.ndarray:
    total = np.zeros((len(PHASE_CLASSES), len(PHASE_CLASSES)), dtype=int)
    for _, payload in per_seed.items():
        cm = np.asarray(payload["test"]["confusion"], dtype=int)
        total += cm
    return total


def _print_markdown(name: str, cm: np.ndarray, n: int | None = None) -> None:
    cm_pct = _row_normalize(cm)
    supports = cm.sum(axis=1)
    print(f"\n#### {name}" + (f"  *(n_test={n})*" if n is not None else ""))
    head = "| true \\\\ pred | " + " | ".join(PHASE_CLASSES) + " | support |"
    sep = "|---|" + "|".join("---" for _ in PHASE_CLASSES) + "|---|"
    print(head)
    print(sep)
    for i, c in enumerate(PHASE_CLASSES):
        cells = " | ".join(f"{cm_pct[i, j]:.1f}%" for j in range(cm_pct.shape[1]))
        print(f"| **{c}** | {cells} | {int(supports[i])} |")


def _save_png(name: str, cm: np.ndarray, slug: str) -> None:
    cm_pct = _row_normalize(cm)
    supports = cm.sum(axis=1).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm_pct, cmap="Blues", aspect="equal", vmin=0, vmax=100)
    ax.set_xticks(range(len(PHASE_CLASSES)))
    ax.set_yticks(range(len(PHASE_CLASSES)))
    ax.set_xticklabels(PHASE_CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(
        [f"{c}\n(n={n})" for c, n in zip(PHASE_CLASSES, supports)]
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(name, fontsize=9)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            v = cm_pct[i, j]
            color = "white" if v >= 50 else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color=color, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045)
    cbar.set_label("% of true class")
    plt.tight_layout()
    path = FIG_DIR / f"{slug}.png"
    plt.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  -> {path}", file=sys.stderr)


# ---- LORO ------------------------------------------------------------------

def render_loro() -> None:
    print("\n## LORO confusion matrices (sum across recording folds)\n")

    cls = _load(OUT / "local_loro.json")
    if cls is not None:
        print(f"\n### Classical ??{cls.get('label', 'PDF features only')}")
        for model in ("knn", "randomforest", "xgboost"):
            cm = np.asarray(cls["summary"][model]["confusion_total"], dtype=int)
            _print_markdown(f"{model.upper()} ??LORO", cm, int(cm.sum()))
            _save_png(f"{model.upper()}  LORO", cm, f"loro__classical_{model}")

    dl = _load(OUT / "dl_local_loro.json")
    if dl is not None:
        print(f"\n### 1D-CNN ??{dl.get('label', 'PDF channels')}")
        cm = np.asarray(dl["summary"]["confusion_total"], dtype=int)
        _print_markdown("1D-CNN ??LORO", cm, int(cm.sum()))
        _save_png("1D-CNN  LORO", cm, "loro__cnn")


# ---- Random split ---------------------------------------------------------

def render_random_split() -> None:
    print("\n\n## Random 70:15:15 confusion matrices (sum across 5 seeds)\n")

    cls = _load(OUT / "local_randomsplit.json")
    if cls is not None:
        print(f"\n### Classical ??{cls.get('label', 'PDF features only')}")
        for model in ("knn", "randomforest", "xgboost"):
            cm = _sum_seed_cms_classical(cls["per_seed"], model)
            _print_markdown(f"{model.upper()} ??random 70:15:15", cm, int(cm.sum()))
            _save_png(f"{model.upper()}  random 70:15:15", cm,
                      f"randomsplit__classical_{model}")

    dl = _load(OUT / "dl_local_randomsplit.json")
    if dl is not None:
        print(f"\n### 1D-CNN ??{dl.get('label', 'PDF channels')}")
        cm = _sum_seed_cms_dl(dl["per_seed"])
        _print_markdown("1D-CNN ??random 70:15:15", cm, int(cm.sum()))
        _save_png("1D-CNN  random 70:15:15", cm, "randomsplit__cnn")


def main() -> None:
    render_loro()
    render_random_split()
    print(f"\nAll PNGs saved under {FIG_DIR}/", file=sys.stderr)


if __name__ == "__main__":
    main()

