"""Auditable receipts from the stationary-point polish benchmark."""

from __future__ import annotations

import json

import numpy as np

from benchmarks import stationary_point_polish as benchmark


def _expected_latency_summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples)
    return {
        "median_us": float(np.median(values)),
        "p95_us": float(np.percentile(values, 95)),
        "minimum_us": float(np.min(values)),
        "maximum_us": float(np.max(values)),
        "mean_us": float(np.mean(values)),
    }


def test_latency_measurement_retains_each_sample(monkeypatch):
    timestamps_ns = iter((1_000, 12_000, 20_000, 37_000, 50_000, 73_000))
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: next(timestamps_ns))

    measurement = benchmark._latencies(
        lambda argument: argument,
        benchmark.jnp.asarray(0.0),
        repetitions=3,
    )

    samples = [11.0, 17.0, 23.0]
    assert measurement == {"samples_us": samples, **_expected_latency_summary(samples)}


def test_small_receipt_persists_raw_samples_and_recomputable_aggregates(
    monkeypatch,
):
    repetitions = 3
    latency_arrays = iter(
        (
            [11.0, 17.0, 23.0],
            [3.0, 5.0, 13.0],
            [31.0, 37.0, 41.0],
            [7.0, 19.0, 29.0],
        )
    )

    def measured_latencies(_function, _argument, measured_repetitions):
        assert measured_repetitions == repetitions
        samples = next(latency_arrays)
        return {"samples_us": samples, **_expected_latency_summary(samples)}

    monkeypatch.setattr(benchmark, "_latencies", measured_latencies)
    receipt = benchmark.run(repetitions)

    arms = (
        "cold_tolerance_exit",
        "warm_tolerance_exit",
        "tracked_sequence",
        "batched_fixed_cap",
    )
    expected_samples = (
        [11.0, 17.0, 23.0],
        [3.0, 5.0, 13.0],
        [31.0, 37.0, 41.0],
        [7.0, 19.0, 29.0],
    )
    for arm, samples in zip(arms, expected_samples, strict=True):
        measurement = receipt[arm]
        assert measurement["samples_us"] == samples
        for field, expected in _expected_latency_summary(samples).items():
            assert measurement[field] == expected

    cold_counts = np.asarray(receipt["cold_tolerance_exit"]["iteration_counts"])
    warm_counts = np.asarray(receipt["warm_tolerance_exit"]["iteration_counts"])
    tracked_counts = np.asarray(receipt["tracked_sequence"]["iteration_counts"])

    assert cold_counts.shape == (32,)
    assert warm_counts.shape == (32,)
    assert tracked_counts.shape == (16, 32)
    assert receipt["cold_tolerance_exit"]["maximum_iterations"] == int(
        cold_counts.max()
    )
    assert receipt["cold_tolerance_exit"]["mean_iterations"] == float(
        cold_counts.mean()
    )
    assert receipt["warm_tolerance_exit"]["maximum_iterations"] == int(
        warm_counts.max()
    )
    assert receipt["warm_tolerance_exit"]["mean_iterations"] == float(
        warm_counts.mean()
    )
    assert receipt["tracked_sequence"]["maximum_iterations_per_field"] == (
        tracked_counts.max(axis=1).tolist()
    )
    assert receipt["tracked_sequence"]["mean_iterations"] == float(
        tracked_counts.mean()
    )

    serialized = json.dumps(receipt, allow_nan=False)
    assert json.loads(serialized) == receipt
