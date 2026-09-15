"""Command-line entry point for the canonical Plan C workflow."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import CLASSES, FEATURES, XGB_PARAMS
from .data import prepare_dataset
from .evaluation import evaluate_loro, save_evaluation
from .training import train_production_model


def _write_run_config(output_dir: Path, command: str, data_dir: str, df) -> None:
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "data_dir": data_dir,
        "recordings": sorted(df["rec_name"].unique().tolist()),
        "n_windows": int(len(df)),
        "classes": list(CLASSES),
        "features": list(FEATURES),
        "model_parameters": XGB_PARAMS,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan C data, evaluation, and production training workflow")
    parser.add_argument("command", choices=("prepare", "evaluate", "train", "run-all"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/plan_c")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Preparing Plan C dataset from {args.data_dir} ...")
    df = prepare_dataset(args.data_dir)
    print(f"Prepared {len(df)} windows from {df['rec_name'].nunique()} recordings")
    _write_run_config(output_dir, args.command, args.data_dir, df)

    if args.command in {"prepare", "run-all"}:
        feature_path = output_dir / "features.csv"
        df.to_csv(feature_path, index=False)
        print(f"Features: {feature_path}")

    if args.command in {"evaluate", "run-all"}:
        print("Running leave-one-recording-out evaluation ...")
        result = evaluate_loro(df)
        metrics_path, figure_path = save_evaluation(result, output_dir / "evaluation")
        print(f"Accuracy: {result['pooled_accuracy']:.1%}")
        print(f"Macro F1: {result['pooled_macro_f1']:.3f}")
        print(f"Metrics: {metrics_path}")
        print(f"Confusion matrix: {figure_path}")

    if args.command in {"train", "run-all"}:
        print("Training production model on all recordings ...")
        artifacts = train_production_model(df, output_dir / "production")
        for name, path in artifacts.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
