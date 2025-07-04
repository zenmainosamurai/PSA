from dataclasses import dataclass
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import (
        MassBalanceResults,
        HeatBalanceResults,
        WallHeatBalanceResult,
        LidHeatBalanceResult,
        MoleFractionResults,
        VacuumPumpingResult,
        DownstreamFlowResult,
    )


@dataclass
class BaseOperationResult:
    material: "MassBalanceResults"
    heat: "HeatBalanceResults"
    heat_wall: Dict[int, "WallHeatBalanceResult"]
    heat_lid: Dict[str, "LidHeatBalanceResult"]

    def get_record_items(self) -> Dict[str, any]:
        return {
            "material": self.material,
            "heat": self.heat,
            "heat_wall": self.heat_wall,
            "heat_lid": self.heat_lid,
        }


@dataclass
class StopModeResult(BaseOperationResult):
    pass


@dataclass
class BatchAdsorptionResult(BaseOperationResult):
    pressure_after_batch_adsorption: float


@dataclass
class FlowAdsorptionResult(BaseOperationResult):
    total_pressure: float


@dataclass
class EqualizationDepressurizationResult(BaseOperationResult):
    total_pressure: float
    diff_press: float
    downflow_params: "DownstreamFlowResult"


@dataclass
class VacuumDesorptionResult(BaseOperationResult):
    mol_fraction: Optional["MoleFractionResults"]
    accum_vacuum_amt: "VacuumPumpingResult"
    pressure_after_vacuum_desorption: float


OperationResult = (
    StopModeResult
    | BatchAdsorptionResult
    | FlowAdsorptionResult
    | EqualizationDepressurizationResult
    | VacuumDesorptionResult
)
