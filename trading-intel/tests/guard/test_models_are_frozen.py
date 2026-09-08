"""每一個契約模型都必須維持 frozen 並拒收未知欄位。

刻意用反射寫成：Phase 3 新增的模型若忘了設 ``frozen=True``，
不需要任何人記得更新清單，這裡就會失敗。
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
    # 防止反射什麼都沒找到卻假性通過。
    assert len(ALL_MODELS) >= 10, [m.__name__ for m in ALL_MODELS]


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_model_is_frozen(model: type[BaseModel]) -> None:
    assert model.model_config.get("frozen") is True, f"{model.__name__} 未設為 frozen"


@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_model_forbids_extra_fields(model: type[BaseModel]) -> None:
    assert model.model_config.get("extra") == "forbid", f"{model.__name__} 未禁止額外欄位"
