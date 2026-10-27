# Evaluation

- `run_test_selected_suite.py`: isolated, deliberately test-selected diagnostic experiment; launch with `python train_test_selected.py`. See [protocol and commands](../../report/test-selected-experiment.md). It does not change the rigorous runner.

- `run_plan_c.py`: legacy Plan C LORO evaluation.
- `run_split_reports.py`: classical model LORO comparison.
- `train_and_plot_all_cm.py`: historical multi-model comparison.
- `run_transfer.py`, `run_dl_transfer.py`: transfer evaluations.

Canonical command: `python -m src.plan_c evaluate --output-dir outputs/plan_c`.
