"""Tests for metrics.py save_results_to_csv with counting_accuracy."""

import csv

from metrics import save_results_to_csv


class TestSaveResultsToCsvCountingAccuracy:
    """Tests that counting_accuracy is written to the CSV when present in metrics."""

    def _make_config(self):
        return {
            'model': 'attention',
            'epochs': 1,
            'lr': 0.0005,
            'reg': 0.0001,
            'target_number': 9,
            'mean_bag_length': 10,
            'var_bag_length': 2,
            'num_bags_train': 200,
            'num_bags_test': 50,
            'seed': 1,
            'test_loss': 0.5,
            'test_error': 0.1,
        }

    def _make_metrics(self, include_counting=False):
        metrics = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.8,
            'f1_score': 0.82,
            'auc': 0.95,
        }
        if include_counting:
            metrics['counting_accuracy'] = 0.76
        return metrics

    def test_counting_accuracy_written_to_csv(self, tmp_path):
        filepath = str(tmp_path / 'results.csv')
        config = self._make_config()
        metrics = self._make_metrics(include_counting=True)
        save_results_to_csv(filepath, config, metrics)

        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        assert len(rows) == 1
        assert 'counting_accuracy' in rows[0]
        assert float(rows[0]['counting_accuracy']) == 0.76

    def test_no_counting_accuracy_when_not_provided(self, tmp_path):
        filepath = str(tmp_path / 'results.csv')
        config = self._make_config()
        metrics = self._make_metrics(include_counting=False)
        save_results_to_csv(filepath, config, metrics)

        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        assert len(rows) == 1
        assert 'counting_accuracy' not in rows[0]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
