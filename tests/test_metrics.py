import pytest

from app.metrics import ContextMetrics


def test_metrics_for_input_within_budget() -> None:
    metrics = ContextMetrics.calculate(
        input_tokens=4000,
        max_input_tokens=7000,
    )

    assert metrics.input_tokens == 4000
    assert metrics.max_input_tokens == 7000
    assert metrics.remaining_tokens == 3000
    assert metrics.overflow_tokens == 0
    assert metrics.utilization_percent == pytest.approx(57.14, abs=0.01)
    assert metrics.fits is True


def test_metrics_for_oversized_input() -> None:
    metrics = ContextMetrics.calculate(
        input_tokens=8500,
        max_input_tokens=7000,
    )

    assert metrics.input_tokens == 8500
    assert metrics.remaining_tokens == 0
    assert metrics.overflow_tokens == 1500
    assert metrics.utilization_percent == pytest.approx(121.43, abs=0.01)
    assert metrics.fits is False


def test_metrics_for_exact_budget() -> None:
    metrics = ContextMetrics.calculate(
        input_tokens=7000,
        max_input_tokens=7000,
    )

    assert metrics.remaining_tokens == 0
    assert metrics.overflow_tokens == 0
    assert metrics.utilization_percent == 100
    assert metrics.fits is True


def test_negative_input_tokens_are_rejected() -> None:
    with pytest.raises(ValueError):
        ContextMetrics.calculate(
            input_tokens=-1,
            max_input_tokens=7000,
        )


def test_invalid_max_input_tokens_are_rejected() -> None:
    with pytest.raises(ValueError):
        ContextMetrics.calculate(
            input_tokens=100,
            max_input_tokens=0,
        )