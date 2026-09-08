from __future__ import annotations

import pytest

from trading_intel.core.errors import (
    BudgetExceededError,
    ClockError,
    ClockRewindError,
    ConfigError,
    DataQualityError,
    LookaheadError,
    NaiveDatetimeError,
    NetworkAccessDenied,
    SchemaValidationError,
    TemporalIntegrityError,
    TradingIntelError,
)

ALL_ERRORS = [
    ConfigError,
    ClockError,
    NaiveDatetimeError,
    ClockRewindError,
    TemporalIntegrityError,
    LookaheadError,
    NetworkAccessDenied,
    DataQualityError,
    SchemaValidationError,
    BudgetExceededError,
]


@pytest.mark.parametrize("error", ALL_ERRORS, ids=lambda e: e.__name__)
def test_every_error_descends_from_the_base(error: type[TradingIntelError]) -> None:
    assert issubclass(error, TradingIntelError)


@pytest.mark.parametrize("error", [NaiveDatetimeError, ClockRewindError])
def test_clock_errors_are_grouped(error: type[ClockError]) -> None:
    assert issubclass(error, ClockError)


def test_context_defaults_to_empty() -> None:
    err = ConfigError("something broke")
    assert err.context == {}
    assert str(err) == "something broke"


def test_context_is_rendered_and_sorted() -> None:
    err = ConfigError("bad value", field="risk/max_net_exposure", env="prod")
    rendered = str(err)
    assert rendered.startswith("bad value (")
    assert rendered.index("env=") < rendered.index("field=")
    assert err.context["env"] == "prod"


def test_context_only_error_renders_just_the_context() -> None:
    assert str(ConfigError(env="prod")) == "env='prod'"


def test_repr_shows_the_context() -> None:
    err = ConfigError("bad", env="prod")
    assert repr(err) == "ConfigError('bad', context={'env': 'prod'})"


def test_errors_are_catchable_as_the_base_class() -> None:
    with pytest.raises(TradingIntelError):
        raise LookaheadError("read the future", asof="2020-03-19T00:00:00+00:00")
