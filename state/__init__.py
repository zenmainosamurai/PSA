"""状態管理

PSA担当者向け説明:
塔の状態変数（温度、圧力、吸着量など）と計算結果のデータクラスを提供します。

主要なエクスポート:
- StateVariables: 塔の状態変数管理
- TowerStateArrays: 塔ごとの状態配列
- MaterialBalanceResult: 物質収支結果
- HeatBalanceResult: 熱収支結果
- MassBalanceResults: 物質収支結果コレクション
"""

# State management
from .state_variables import StateVariables, TowerStateArrays

# Result structures from results.py
from .results import (
    # Basic data structures
    GasFlow,
    GasProperties,
    AdsorptionState,
    PressureState,
    CellTemperatures,
    HeatFlux,
    HeatTransferCoefficients,
    WallHeatFlux,
    # Main result classes
    MaterialBalanceResult,
    HeatBalanceResult,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    # Collections
    MassBalanceResults,
    HeatBalanceResults,
    MoleFractionResults,
    # Specialized results
    VacuumPumpingResult,
    DepressurizationResult,
    DownstreamFlowResult,
    DesorptionMoleFractionResult,
    MassBalanceCalculationResult,
    # Combined results
    MassAndHeatBalanceResults,
)

__all__ = [
    # State management
    "StateVariables",
    "TowerStateArrays",
    # Basic data structures
    "GasFlow",
    "GasProperties",
    "AdsorptionState",
    "PressureState",
    "CellTemperatures",
    "HeatFlux",
    "HeatTransferCoefficients",
    "WallHeatFlux",
    # Main result classes
    "MaterialBalanceResult",
    "HeatBalanceResult",
    "WallHeatBalanceResult",
    "LidHeatBalanceResult",
    # Collections
    "MassBalanceResults",
    "HeatBalanceResults",
    "MoleFractionResults",
    # Specialized results
    "VacuumPumpingResult",
    "DepressurizationResult",
    "DownstreamFlowResult",
    "DesorptionMoleFractionResult",
    "MassBalanceCalculationResult",
    # Combined results
    "MassAndHeatBalanceResults",
]
