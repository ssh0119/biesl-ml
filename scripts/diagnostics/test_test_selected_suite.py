"""Small synthetic checks; never starts a real data_new training run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluation import run_test_selected_suite as suite


class TinyCNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=5):
        super().__init__()
        self.linear = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        return self.linear(x.mean(-1))


class TestTestSelectedSuite(unittest.TestCase):
    def test_best_checkpoint_is_restored_all_epochs_scanned_and_ties_keep_first(self):
        train = np.ones((5, 3, 8), dtype=np.float32)
        target = np.arange(5)
        test = np.full((2, 3, 8), 100., dtype=np.float32)
        test_y = np.array([0, 1])
        snapshots = []

        def loader(X, y, **kwargs):
            np.testing.assert_array_equal(X, train)  # test signals must not reach optimizer
            np.testing.assert_array_equal(y, target)
            return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=5)

        def predict(model, X, **kwargs):
            np.testing.assert_array_equal(X, test)
            snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            # Epoch 1 best, epoch 2 worse, epoch 3 ties the best.
            return np.eye(5)[[0, 1] if len(snapshots) != 2 else [2, 2]]

        for metric, perfect in (("accuracy", False), ("macro_f1", False), ("accuracy", True), ("macro_f1", True)):
            snapshots.clear()
            with patch.object(suite, "DEVICE", "cpu"), patch.object(suite, "Phase2CNN1D", TinyCNN), \
                 patch.object(suite, "_class_weights", return_value=torch.ones(5)), \
                 patch.object(suite, "_loader", side_effect=loader), \
                 patch.object(suite, "predict_probabilities", side_effect=predict):
                model, selected = suite.train_test_selected(train, target, test, test_y, max_epochs=3, selection_metric=metric, stop_on_perfect=perfect)
            self.assertEqual(selected["best_epoch"], 1)
            self.assertEqual(len(selected["history"]), 1 if perfect else 3)
            self.assertEqual(selected["early_stopping"], perfect)
            self.assertFalse(model.training)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, snapshots[0][k])
            if not perfect:
                self.assertTrue(any(not torch.equal(snapshots[0][k], snapshots[-1][k]) for k in snapshots[0]))

    def test_imperfect_plateau_only_stops_when_patience_enabled(self):
        X = np.ones((5, 3, 8), np.float32)
        y = np.arange(5)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=5)
        p = np.eye(5)[[0, 1, 2, 3, 0]]  # 80%, never a perfect score
        for patience, expected in ((0, 5), (2, 3)):
            with patch.object(suite, "DEVICE", "cpu"), patch.object(suite, "Phase2CNN1D", TinyCNN), \
                 patch.object(suite, "_class_weights", return_value=torch.ones(5)), \
                 patch.object(suite, "_loader", return_value=loader), \
                 patch.object(suite, "predict_probabilities", return_value=p):
                _, selected = suite.train_test_selected(X, y, X, y, max_epochs=5, selection_metric="accuracy", patience=patience)
            self.assertEqual(selected["epochs_scanned"], expected)
            self.assertEqual(selected["best_epoch"], 1)

    def test_report_labels_bias_and_preserves_all_method_predictions(self):
        y = np.arange(5)
        data = {"y": y, "recordings": np.array(["rec"] * 5), "anchors": np.arange(5.)}
        probabilities = {k: np.eye(5) for k in suite.METHODS}
        temp_root = Path("outputs/test_selected_checks").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="report_", dir=temp_root) as directory:
            out = Path(directory)
            suite.save_report(out, data, probabilities, [{}], {"selection_metric": "accuracy", "max_epochs": 3})
            result = json.loads((out / "results.json").read_text())
            self.assertFalse(result["unbiased_test_evaluation"])
            self.assertEqual(set(result["results"]), set(suite.METHODS))
            self.assertEqual(result["results"]["cnn"]["accuracy"], 1.)
            self.assertIn("AND", result["results"]["test_tuned_fusion"]["selection_bias"])
            for name in suite.METHODS:
                self.assertTrue((out / f"confusion_matrix_{name}.png").is_file())

    def test_historical_output_is_rejected_before_writing(self):
        with self.assertRaises(SystemExit):
            suite.main(["--output-dir", "output_data_new", "--prepare-only"])


if __name__ == "__main__":
    unittest.main()
