"""状態管理

PSA担当者向け説明:
塔の状態変数（温度、圧力、吸着量など）と計算結果のデータクラスを提供します。

主要なエクスポート:
- TowerResults: 1塔の計算結果
- OperationResult: 運転モード計算の統一結果
- MaterialBalanceResult: 物質収支結果
- HeatBalanceResult: 熱収支結果
"""

from .calculation_results import (
    # 基本データクラス
    GasFlow,
    GasProperties,
    AdsorptionState,
    PressureState,
    CellTemperatures,
    HeatTransferCoefficients,
    HeatFlux,
    WallHeatFlux,
    
    # 物質収支結果
    MaterialBalanceResult,
    MassBalanceCalculationResult,
    
    # 熱収支結果
    HeatBalanceResult,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    
    # 真空脱着関連
    VacuumPumpingResult,
    DesorptionMoleFractionResult,
    
    # 均圧関連
    DepressurizationResult,
    DownstreamFlowResult,
    
    # 統合結果
    TowerResults,
    OperationResult,
)

__all__ = [
    # 基本データクラス
    "GasFlow",
    "GasProperties",
    "AdsorptionState",
    "PressureState",
    "CellTemperatures",
    "HeatTransferCoefficients",
    "HeatFlux",
    "WallHeatFlux",
    
    # 物質収支結果
    "MaterialBalanceResult",
    "MassBalanceCalculationResult",
    
    # 熱収支結果
    "HeatBalanceResult",
    "WallHeatBalanceResult",
    "LidHeatBalanceResult",
    
    # 真空脱着関連
    "VacuumPumpingResult",
    "DesorptionMoleFractionResult",
    
    # 均圧関連
    "DepressurizationResult",
    "DownstreamFlowResult",
    
    # 統合結果
    "TowerResults",
    "OperationResult",
]
