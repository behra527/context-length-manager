import pytest

from app.budget import ContextBudget


def test_max_input_tokens() -> None:
    budget = ContextBudget(
        context_window=8192,
        max_output_tokens=1000,
        safety_margin=200,
    )

    assert budget.max_input_tokens == 6992


def test_input_fits_within_budget() -> None:
    budget = ContextBudget(
        context_window=8192,
        max_output_tokens=1000,
        safety_margin=200,
    )

    assert budget.fits(6000) is True


def test_input_does_not_fit() -> None:
    budget = ContextBudget(
        context_window=8192,
        max_output_tokens=1000,
        safety_margin=200,
    )

    assert budget.fits(7000) is False


def test_remaining_tokens() -> None:
    budget = ContextBudget(
        context_window=8192,
        max_output_tokens=1000,
        safety_margin=200,
    )

    assert budget.remaining(6000) == 992


def test_remaining_tokens_never_goes_negative() -> None:
    budget = ContextBudget(
        context_window=8192,
        max_output_tokens=1000,
        safety_margin=200,
    )

    assert budget.remaining(8000) == 0


def test_negative_input_tokens_are_rejected() -> None:
    budget = ContextBudget(
        context_window=8192,
        max_output_tokens=1000,
        safety_margin=200,
    )

    with pytest.raises(ValueError):
        budget.fits(-1)


def test_invalid_budget_configuration() -> None:
    with pytest.raises(ValueError):
        ContextBudget(
            context_window=1000,
            max_output_tokens=900,
            safety_margin=200,
        )