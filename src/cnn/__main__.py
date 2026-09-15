"""CLI for raw-window preparation, LORO evaluation, and final CNN training."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import MAX_EPOCHS, PATIENCE
from .data import build_raw_dataset, load_cache, save_cache
from .evaluation import evaluate_loro
from .production import recommended_epochs, train_and_save


def _dataset(args: argparse.Namespace) -> dict:
    cache = Path(args.cache) if args.cache else Path(args.output_dir) / "raw_windows_250hz.npz"
    if cache.exists() and not args.rebuild_cache:
        print(f"loading cache: {cache}")
        return load_cache(cache)
    print(f"building 250 Hz raw-window cache from: {args.data_dir}")
    dataset = build_raw_dataset(args.data_dir)
    save_cache(dataset, cache)
    print(f"saved cache: {cache}")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 folder-label 1D-CNN")
    parser.add_argument("command", choices=("prepare", "evaluate", "train", "run-all"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output_new/cnn_phase2")
    parser.add_argument("--cache")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--final-epochs", type=int)
    parser.add_argument("--test-recording", help="evaluate only this recording (smoke-test fold)")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    dataset = _dataset(args)
    if args.command == "prepare":
        return
    if args.command in {"evaluate", "run-all"}:
        tests = [args.test_recording] if args.test_recording else None
        evaluation = evaluate_loro(
            dataset, args.output_dir, max_epochs=args.max_epochs,
            patience=args.patience, test_recordings=tests,
        )
        print(f"pooled accuracy={evaluation['accuracy']:.4f}, macro_f1={evaluation['macro_f1']:.4f}")
        if args.command == "evaluate":
            return
        epochs = args.final_epochs or recommended_epochs(evaluation)
    else:
        epochs = args.final_epochs or args.max_epochs
    path = train_and_save(dataset, args.output_dir, epochs=epochs)
    print(f"saved final model: {path} (epochs={epochs})")


if __name__ == "__main__":
    main()
