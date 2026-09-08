"""Every contract model must stay frozen and reject unknown fields.

This is written reflectively on purpose: a model added in Phase 3 that forgets
``frozen=True`` fails here without anyone remembering to update a list.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import pytest
from pydantic import BaseModel

from trading_intel.core import events, types

MODULES: tuple[ModuleType, ...] = (types, events)


def _declared_models(module: ModuleType) -> list[type[BaseModel]]:
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == module.__name__
    ]


ALL_MODELS = [model for module in MODULES for model in _declared_models(module)]


def test_there_are_models_to_check() -> None:
    # Guards against the reflection silently finding nothing and passing.
    assert len(ALL_MODELS) >= 10, [m.__name__ for m in ALL_MODELS]


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_model_is_frozen(model: type[BaseModel]) -> None:
    assert model.model_config.get("frozen") is True, f"{model.__name__} is not frozen"


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_model_forbids_extra_fields(model: type[BaseModel]) -> None:
    assert model.model_config.get("extra") == "forbid", (
        f"{model.__name__} does not forbid extra fields"
    )
