"""特徵註冊表：版本、相依欄位、預期值域。

SPEC 5.1 要求特徵版本化，理由是可重現性：改了參數就是新特徵，舊版本必須留著，
否則半年前那個訊號永遠無法重現。

註冊表同時記錄**預期值域**。這不是文件，是執行期的檢查：一個宣稱輸出落在
[-1, 1] 的特徵突然吐出 15，代表計算出錯或輸入資料有問題，此時該報錯而不是
讓那個 15 一路流進訊號。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from trading_intel.core.errors import SchemaValidationError

FloatArray = npt.NDArray[np.float64]

#: 特徵函式的簽章：``f(data, asof, **params) -> Series``（SPEC 5.1）。
FeatureFunction = Callable[..., FloatArray]

_ASOF_PARAMETER: Final = "asof"
_DATA_PARAMETER: Final = "data"


@dataclass(frozen=True)
class FeatureSpec:
    """一個特徵的完整定義。"""

    name: str
    version: str
    function: FeatureFunction
    #: 這個特徵需要輸入資料的哪些欄位。缺欄位時直接拒絕計算。
    required_columns: tuple[str, ...]
    #: 預期輸出值域 (下界, 上界)。None 代表不設限。
    expected_range: tuple[float, float] | None = None
    #: 是否為均值回歸類特徵。是的話，序列未通過定態檢定就必須回傳 NaN。
    mean_reverting: bool = False
    description: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """``mom_20d@v1`` 這種形式。訊號記錄的是這個，不是裸名稱。"""
        return f"{self.name}@{self.version}"


class FeatureRegistry:
    """特徵的登記處。

    同名同版本重複註冊會被拒絕——那通常代表有人改了實作卻忘了升版本，
    而那會讓舊訊號無法重現。
    """

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> FeatureSpec:
        if spec.qualified_name in self._specs:
            raise SchemaValidationError(
                "特徵的同一版本已註冊過。改了實作請升版本，不要覆寫舊版",
                feature=spec.qualified_name,
            )
        signature = inspect.signature(spec.function)
        missing = {_DATA_PARAMETER, _ASOF_PARAMETER} - set(signature.parameters)
        if missing:
            raise SchemaValidationError(
                "特徵函式必須接受 data 與 asof 兩個參數（CLAUDE.md 第 2 條）",
                feature=spec.qualified_name,
                missing=", ".join(sorted(missing)),
            )
        self._specs[spec.qualified_name] = spec
        return spec

    def get(self, name: str, version: str) -> FeatureSpec:
        key = f"{name}@{version}"
        if key not in self._specs:
            raise SchemaValidationError("未註冊的特徵", feature=key)
        return self._specs[key]

    def versions_of(self, name: str) -> tuple[str, ...]:
        return tuple(sorted(spec.version for spec in self._specs.values() if spec.name == name))

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, qualified_name: object) -> bool:
        return qualified_name in self._specs

    def names(self) -> tuple[str, ...]:
        return tuple(sorted({spec.name for spec in self._specs.values()}))

    def compute(
        self,
        name: str,
        version: str,
        data: Any,
        *,
        asof: datetime,
        **params: Any,
    ) -> FloatArray:
        """計算一個特徵，並在前後做把關。

        前：檢查輸入含有宣告的相依欄位。
        後：檢查輸出落在宣告的值域內。
        """
        spec = self.get(name, version)
        _validate_columns(spec, data)
        merged = {**spec.params, **params}
        result = np.asarray(spec.function(data, asof=asof, **merged), dtype=np.float64)
        _validate_range(spec, result)
        return result


def _validate_columns(spec: FeatureSpec, data: Any) -> None:
    columns = getattr(data, "columns", None)
    if columns is None:
        return
    available = set(map(str, columns))
    missing = [column for column in spec.required_columns if column not in available]
    if missing:
        raise SchemaValidationError(
            "輸入資料缺少特徵所需的欄位",
            feature=spec.qualified_name,
            missing=", ".join(missing),
        )


def _validate_range(spec: FeatureSpec, values: FloatArray) -> None:
    if spec.expected_range is None:
        return
    lower, upper = spec.expected_range
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    observed_low = float(finite.min())
    observed_high = float(finite.max())
    if observed_low < lower or observed_high > upper:
        raise SchemaValidationError(
            "特徵輸出超出宣告的預期值域，可能是計算錯誤或輸入資料有問題",
            feature=spec.qualified_name,
            expected=f"[{lower}, {upper}]",
            observed=f"[{observed_low}, {observed_high}]",
        )


def neutralize(
    values: npt.ArrayLike,
    groups: Sequence[str],
    *,
    market_cap: npt.ArrayLike | None = None,
) -> FloatArray:
    """橫斷面中性化：先扣掉產業效應，再扣掉市值效應。

    SPEC 5.1：「橫斷面特徵一律先做產業中性化與市值中性化，再標準化」。

    不做的後果：一個「動量」特徵在半導體大漲的那個月，會把所有半導體股都排在
    前面——那不是選股，那是押注一個產業。中性化把「這檔比它的同業強多少」
    與「這個產業比別的產業強多少」分開。
    """
    array = np.asarray(values, dtype=np.float64).ravel().copy()
    if array.size != len(groups):
        raise SchemaValidationError(
            "數值與分組的長度不一致",
            values=array.size,
            groups=len(groups),
        )

    # 產業中性化：減去各組的平均。
    result = array.copy()
    for group in set(groups):
        mask = np.array([item == group for item in groups])
        members = result[mask]
        finite = members[np.isfinite(members)]
        if finite.size == 0:
            continue
        result[mask] = members - float(finite.mean())

    if market_cap is None:
        return result

    # 市值中性化：對 log(市值) 做迴歸，取殘差。
    caps = np.asarray(market_cap, dtype=np.float64).ravel()
    if caps.size != result.size:
        raise SchemaValidationError(
            "市值與數值的長度不一致", values=result.size, market_cap=caps.size
        )
    usable = np.isfinite(result) & np.isfinite(caps) & (caps > 0)
    if usable.sum() < 3:
        return result

    log_cap = np.log(caps[usable])
    design = np.column_stack([np.ones_like(log_cap), log_cap])
    coefficients, *_ = np.linalg.lstsq(design, result[usable], rcond=None)
    result[usable] = result[usable] - design @ coefficients
    return result
