"""Functions to calculate performance metrics and save results to a CSV file."""

import csv
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, mean_absolute_error, mean_squared_error)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate Accuracy, Precision, Recall, F1-Score and AUC.

    Args:
        y_true (array-like): Ground-truth binary labels (0 or 1).
        y_pred (array-like): Predicted binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC requires at least one positive and one negative sample
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = float('nan')

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
    }

def calculate_counting_metrics(count_truth, count_pred):
    """Calculate counting accuracy, MAE, and RMSE.

    Args:
        count_truth (array-like): Ground-truth counts (non-negative integers).
        count_pred (array-like): Predicted counts (non-negative integers).

    Returns:
        dict: A dictionary containing the computed counting metrics.
    """
    count_truth = np.array(count_truth)
    count_pred = np.array(count_pred)

    counting_accuracy = sum(1 for truth, pred in zip(count_truth, count_pred) if truth == pred) / len(count_pred)
    counting_mae = mean_absolute_error(count_truth, count_pred)
    counting_rmse = np.sqrt(mean_squared_error(count_truth, count_pred))

    return {
        'counting_accuracy': counting_accuracy,
        'counting_mae': counting_mae,
        'counting_rmse': counting_rmse
    }

def save_results_to_csv(filepath, config, metrics):
    """Append a run's configuration and metrics to a CSV file.

    If the file does not exist it is created with a header row.
    A 'timestamp' column is automatically added to each row.

    Note:
        The column structure is derived from the keys of *config* and *metrics*
        on the first write.  Subsequent runs must provide the same set of keys
        so that columns remain aligned.  Delete or rename the file when changing
        the configuration parameters that are logged.

    Args:
        filepath (str): Path to the CSV file.
        config (dict): Network and dataset configuration parameters.
        metrics (dict): Performance metrics returned by :func:`calculate_metrics`.
    """
    row = {'timestamp': datetime.now().isoformat()}
    row.update(config)
    row.update(metrics)

    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
