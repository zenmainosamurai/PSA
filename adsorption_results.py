# calculation_results.py (新規ファイル)
from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class GasFlow:
    co2_volume: float
    n2_volume: float
    co2_mole_fraction: float
    n2_mole_fraction: float


@dataclass
class GasProperties:
    density: float
    specific_heat: float


@dataclass
class AdsorptionState:
    equilibrium_loading: float
    actual_uptake_volume: float
    updated_loading: float
    theoretical_loading_delta: float


@dataclass
class PressureState:
    co2_partial_pressure: float
    outlet_co2_partial_pressure: float


@dataclass
class MaterialBalanceResult:
    """マテリアルバランス計算結果"""

    inlet_gas: GasFlow
    outlet_gas: GasFlow
    gas_properties: GasProperties
    adsorption_state: AdsorptionState
    pressure_state: PressureState

    def to_dict(self) -> dict:
        return {
            "inlet_co2_volume": self.inlet_gas.co2_volume,
            "inlet_n2_volume": self.inlet_gas.n2_volume,
            "inlet_co2_mole_fraction": self.inlet_gas.co2_mole_fraction,
            "inlet_n2_mole_fraction": self.inlet_gas.n2_mole_fraction,
            "gas_density": self.gas_properties.density,
            "gas_specific_heat": self.gas_properties.specific_heat,
            "equilibrium_loading": self.adsorption_state.equilibrium_loading,
            "actual_uptake_volume": self.adsorption_state.actual_uptake_volume,
            "updated_loading": self.adsorption_state.updated_loading,
            "outlet_co2_volume": self.outlet_gas.co2_volume,
            "outlet_n2_volume": self.outlet_gas.n2_volume,
            "outlet_co2_mole_fraction": self.outlet_gas.co2_mole_fraction,
            "outlet_n2_mole_fraction": self.outlet_gas.n2_mole_fraction,
            "theoretical_loading_delta": self.adsorption_state.theoretical_loading_delta,
            "co2_partial_pressure": self.pressure_state.co2_partial_pressure,
            "outlet_co2_partial_pressure": self.pressure_state.outlet_co2_partial_pressure,
        }


@dataclass
class HeatFlux:
    adsorption: float
    from_inner_boundary: float
    to_outer_boundary: float
    to_downstream: float
    from_upstream: float


@dataclass
class HeatTransferCoefficients:
    wall_to_bed: float
    bed_to_bed: float


@dataclass
class CellTemperatures:
    bed_temperature: float
    thermocouple_temperature: float


@dataclass
class HeatBalanceResult:
    """熱バランス計算結果"""

    cell_temperatures: CellTemperatures
    heat_transfer_coefficients: HeatTransferCoefficients
    heat_flux: HeatFlux

    def to_dict(self) -> dict:
        return {
            "temp_reached": self.cell_temperatures.bed_temperature,
            "temp_thermocouple_reached": self.cell_temperatures.thermocouple_temperature,
            "hw1": self.heat_transfer_coefficients.wall_to_bed,
            "Hroof": self.heat_flux.from_upstream,
            "Hbb": self.heat_flux.to_downstream,
            "Habs": self.heat_flux.adsorption,
            "Hwin": self.heat_flux.from_inner_boundary,
            "Hwout": self.heat_flux.to_outer_boundary,
            "u1": self.heat_transfer_coefficients.bed_to_bed,
        }


@dataclass
class WallHeatFlux:
    """壁面熱流束計算結果"""

    from_inner_boundary: float
    to_outer_boundary: float
    to_downstream: float
    from_upstream: float


@dataclass
class WallHeatBalanceResult:
    temperature: float
    heat_flux: WallHeatFlux

    def to_dict(self) -> dict:
        return {
            "temp_reached": self.temperature,
            "Hbb": self.heat_flux.to_downstream,
            "Hroof": self.heat_flux.from_upstream,
            "Hwin": self.heat_flux.from_inner_boundary,
            "Hwout": self.heat_flux.to_outer_boundary,
        }


@dataclass
class LidHeatBalanceResult:
    """蓋熱バランス計算結果"""

    temperature: float

    def to_dict(self) -> dict:
        return {"temp_reached": self.temperature}


@dataclass
class DesorptionMoleFractionResult:
    """脱着時のモル分率計算結果"""

    co2_mole_fraction_after_desorption: float
    n2_mole_fraction_after_desorption: float
    total_moles_after_desorption: float

    def to_dict(self) -> dict:
        return {
            "mf_co2_after_vacuum": self.co2_mole_fraction_after_desorption,
            "mf_n2_after_vacuum": self.n2_mole_fraction_after_desorption,
            "desorp_mw_all_after_vacuum": self.total_moles_after_desorption,
        }


@dataclass
class VacuumPumpingResult:
    """真空排気計算結果"""

    pressure_loss: float
    total_co2_recovered: float
    total_n2_recovered: float
    co2_recovery_concentration: float
    volumetric_flow_rate: float
    remaining_moles: float
    final_pressure: float

    def to_dict(self) -> dict:
        return {
            "P_resist": self.pressure_loss,
            "accum_vacuum_amt_co2": self.total_co2_recovered,
            "accum_vacuum_amt_n2": self.total_n2_recovered,
            "vacuum_co2_mf": self.co2_recovery_concentration,
            "vacuum_rate_N": self.volumetric_flow_rate,
            "case_inner_mol_amt_after_vacuum": self.remaining_moles,
            "total_press_after_vacuum": self.final_pressure,
        }


@dataclass
class DepressurizationResult:
    """減圧計算結果"""

    final_pressure: float
    flow_rate: float
    pressure_differential: float

    def to_dict(self) -> dict:
        return {
            "total_press_after_depressure": self.final_pressure,
            "flow_amount_l": self.flow_rate,
            "diff_press": self.pressure_differential,
        }


@dataclass
class DownstreamFlowResult:
    """下流側流量計算結果"""

    final_pressure: float
    outlet_flows: Dict[int, GasFlow]

    def to_dict(self) -> dict:
        outlet_flows = {}
        for stream_id, gas_flow in self.outlet_flows.items():
            outlet_flows[stream_id] = {
                "outlet_co2_volume": gas_flow.co2_volume,
                "outlet_n2_volume": gas_flow.n2_volume,
            }
        return {"total_press_after_depressure_downflow": self.final_pressure, "outflow_fr": outlet_flows}
