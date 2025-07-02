# State management
from .state_variables import StateVariables, TowerStateArrays

# Result structures - organized by category
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
    # Combined results
    "MassAndHeatBalanceResults",
]
