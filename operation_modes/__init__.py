"""運転モード

PSA担当者向け説明:
各運転モード（流通吸着、バッチ吸着、真空脱着など）の定義と計算ロジックを提供します。

主要なエクスポート:
- OperationMode: 運転モードのEnum
- ADSORPTION_MODES: 吸着計算を行うモードの集合
- UPSTREAM_MODES: 上流モードの集合
- DOWNSTREAM_MODES: 下流モードの集合
"""

from .mode_types import (
    OperationMode,
    ADSORPTION_MODES,
    UPSTREAM_MODES,
    DOWNSTREAM_MODES,
    UPSTREAM_DOWNSTREAM_PAIRS,
    EQUALIZATION_MODES,
    PRESSURE_UPDATE_MODES,
    MOLE_FRACTION_UPDATE_MODES,
    BATCH_PRESSURE_CALCULATION_MODES,
    FLOW_PRESSURE_MODES,
    get_mode_category,
)

__all__ = [
    "OperationMode",
    "ADSORPTION_MODES",
    "UPSTREAM_MODES",
    "DOWNSTREAM_MODES",
    "UPSTREAM_DOWNSTREAM_PAIRS",
    "EQUALIZATION_MODES",
    "PRESSURE_UPDATE_MODES",
    "MOLE_FRACTION_UPDATE_MODES",
    "BATCH_PRESSURE_CALCULATION_MODES",
    "FLOW_PRESSURE_MODES",
    "get_mode_category",
]
