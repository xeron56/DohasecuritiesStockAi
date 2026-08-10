"""Transparent forecast-accuracy metrics for held-out closing prices."""

from __future__ import annotations

import numpy as np

from .schema import AccuracyMetrics


def calculate_accuracy_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    *,
    last_context_value: float,
) -> AccuracyMetrics:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    q10 = np.asarray(q10, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)
    if not (actual.shape == predicted.shape == q10.shape == q90.shape):
        raise ValueError("Actual, point, q10, and q90 arrays must have identical shapes.")
    if actual.size == 0:
        raise ValueError("At least one held-out point is required for scoring.")

    error = predicted - actual
    absolute_error = np.abs(error)
    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(np.square(error))))

    non_zero = actual != 0
    mape = (
        float(np.mean(absolute_error[non_zero] / np.abs(actual[non_zero])) * 100)
        if np.any(non_zero)
        else None
    )
    denominator = np.abs(actual) + np.abs(predicted)
    valid_smape = denominator > 0
    smape = (
        float(np.mean(200 * absolute_error[valid_smape] / denominator[valid_smape]))
        if np.any(valid_smape)
        else 0.0
    )
    accuracy = max(0.0, min(100.0, 100.0 - smape))

    centered = actual - np.mean(actual)
    total_variance = float(np.sum(np.square(centered)))
    r_squared = float(1 - np.sum(np.square(error)) / total_variance) if total_variance > 0 else None

    actual_path = np.concatenate(([last_context_value], actual))
    predicted_path = np.concatenate(([last_context_value], predicted))
    actual_direction = np.sign(np.diff(actual_path))
    predicted_direction = np.sign(np.diff(predicted_path))
    directional_accuracy = (
        float(np.mean(actual_direction == predicted_direction) * 100)
        if actual_direction.size
        else None
    )

    coverage = float(np.mean((actual >= q10) & (actual <= q90)) * 100)
    naive = np.full_like(actual, last_context_value)
    naive_mae = float(np.mean(np.abs(naive - actual)))
    skill_vs_naive = float((1 - mae / naive_mae) * 100) if naive_mae > 0 else None

    return AccuracyMetrics(
        accuracy_score=round(accuracy, 4),
        accuracy_definition="100 minus symmetric mean absolute percentage error (sMAPE)",
        mae=round(mae, 6),
        rmse=round(rmse, 6),
        mape_percent=round(mape, 4) if mape is not None else None,
        smape_percent=round(smape, 4),
        r_squared=round(r_squared, 6) if r_squared is not None else None,
        directional_accuracy_percent=(
            round(directional_accuracy, 4) if directional_accuracy is not None else None
        ),
        interval_80_coverage_percent=round(coverage, 4),
        naive_mae=round(naive_mae, 6),
        skill_vs_naive_percent=(round(skill_vs_naive, 4) if skill_vs_naive is not None else None),
    )
