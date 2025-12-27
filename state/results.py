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
    downstream: float
    upstream: float


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
            "wall_to_bed_heat_transfer_coef": self.heat_transfer_coefficients.wall_to_bed,
            "upstream_heat_flux": self.heat_flux.upstream,
            "downstream_heat_flux": self.heat_flux.downstream,
            "adsorption_heat": self.heat_flux.adsorption,
            "heat_flux_from_inner_boundary": self.heat_flux.from_inner_boundary,
            "heat_flux_to_outer_boundary": self.heat_flux.to_outer_boundary,
            "bed_heat_transfer_coef": self.heat_transfer_coefficients.bed_to_bed,
        }


class SectionResults:
    """セクション毎の計算結果を管理するクラス"""

    def __init__(self, section_data: Dict[int, Any]):
        self.section_data = section_data

    def get_cell_result(self, section_id: int) -> Any:
        """指定されたセクションの結果を取得"""
        return self.section_data.get(section_id)


class StreamSectionResults:
    """ストリーム毎にセクション結果を管理するクラス"""

    def __init__(self, results_by_stream_section: Dict[int, Dict[int, Any]] = None):
        if results_by_stream_section is None:
            results_by_stream_section = {}
        self.stream_data = {
            stream_id: SectionResults(sections) for stream_id, sections in results_by_stream_section.items()
        }

    def get_stream_data(self, stream_id: int) -> SectionResults:
        """指定されたストリームのセクション結果を取得"""
        if stream_id not in self.stream_data:
            self.stream_data[stream_id] = SectionResults({})
        return self.stream_data[stream_id]


class MaterialBalanceSectionResults(SectionResults):
    """マテリアルバランス結果のセクション管理"""

    def get_material_balance_result(self, section_id: int) -> MaterialBalanceResult:
        """指定されたセクションのマテリアルバランス結果を取得"""
        return super().get_cell_result(section_id)


class MaterialBalanceStreamSectionResults(StreamSectionResults):
    """マテリアルバランス結果のストリーム・セクション管理"""

    def __init__(self, results_by_stream_section: Dict[int, Dict[int, MaterialBalanceResult]] = None):
        if results_by_stream_section is None:
            results_by_stream_section = {}
        self.stream_data = {
            stream_id: MaterialBalanceSectionResults(sections)
            for stream_id, sections in results_by_stream_section.items()
        }

    def get_stream_data(self, stream_id: int) -> MaterialBalanceSectionResults:
        """指定されたストリームのセクション結果を取得"""
        if stream_id not in self.stream_data:
            self.stream_data[stream_id] = MaterialBalanceSectionResults({})
        return self.stream_data[stream_id]

    def get_material_balance_result(self, stream_id: int, section_id: int) -> MaterialBalanceResult:
        """指定されたストリーム・セクションのマテリアルバランス結果を直接取得"""
        return self.get_stream_data(stream_id).get_material_balance_result(section_id)


class HeatBalanceSectionResults(SectionResults):
    """熱バランス結果のセクション管理"""

    def get_heat_balance_result(self, section_id: int) -> HeatBalanceResult:
        """指定されたセクションの熱バランス結果を取得"""
        return super().get_cell_result(section_id)


class HeatBalanceStreamSectionResults(StreamSectionResults):
    """熱バランス結果のストリーム・セクション管理"""

    def __init__(self, heat_balance_results_dict: Dict[int, Dict[int, HeatBalanceResult]] = None):
        if heat_balance_results_dict is None:
            heat_balance_results_dict = {}
        self.stream_data = {
            stream_id: HeatBalanceSectionResults(sections) for stream_id, sections in heat_balance_results_dict.items()
        }

    def get_stream_data(self, stream_id: int) -> HeatBalanceSectionResults:
        """指定されたストリームのセクション結果を取得"""
        if stream_id not in self.stream_data:
            self.stream_data[stream_id] = HeatBalanceSectionResults({})
        return self.stream_data[stream_id]

    def get_heat_balance_result(self, stream_id: int, section_id: int) -> HeatBalanceResult:
        """指定されたストリーム・セクションの熱バランス結果を直接取得"""
        return self.get_stream_data(stream_id).get_cell_result(section_id)


class MassBalanceResults:
    """マスバランス計算結果の集合"""

    def __init__(self, material_balance_results_dict: Dict[int, Dict[int, MaterialBalanceResult]] = None):
        self.material_balance_stream_section_results = MaterialBalanceStreamSectionResults(
            material_balance_results_dict
        )

    def get_stream_data(self, stream_id: int) -> MaterialBalanceSectionResults:
        """指定されたストリームのセクション結果管理オブジェクトを取得"""
        return self.material_balance_stream_section_results.get_stream_data(stream_id)

    def get_result(self, stream_id: int, section_id: int) -> MaterialBalanceResult:
        """指定されたストリーム・セクションのマテリアルバランス結果を直接取得"""
        return self.material_balance_stream_section_results.get_material_balance_result(stream_id, section_id)


class HeatBalanceResults:
    """熱バランス計算結果の集合"""

    def __init__(self, heat_balance_results_dict: Dict[int, Dict[int, HeatBalanceResult]] = None):
        self.heat_balance_stream_section_results = HeatBalanceStreamSectionResults(heat_balance_results_dict)

    def get_result(self, stream_id: int, section_id: int) -> HeatBalanceResult:
        """指定されたストリーム・セクションの熱バランス結果を直接取得"""
        return self.heat_balance_stream_section_results.get_heat_balance_result(stream_id, section_id)


@dataclass
class DesorptionMoleFractionResult:
    """脱着時のモル分率計算結果"""

    co2_mole_fraction_after_desorption: float
    n2_mole_fraction_after_desorption: float
    total_moles_after_desorption: float


class MoleFractionSectionResults(SectionResults):
    """モル分率結果のセクション管理"""

    def get_mole_fraction_result(self, section_id: int) -> dict:
        """指定されたセクションのモル分率結果を取得"""
        return super().get_cell_result(section_id)


class MoleFractionStreamSectionResults(StreamSectionResults):
    """モル分率結果のストリーム・セクション管理"""

    def __init__(self, mole_fraction_results_dict: Dict[int, Dict[int, dict]] = None):
        if mole_fraction_results_dict is None:
            mole_fraction_results_dict = {}
        self.stream_data = {
            stream_id: MoleFractionSectionResults(sections)
            for stream_id, sections in mole_fraction_results_dict.items()
        }

    def get_stream_data(self, stream_id: int) -> MoleFractionSectionResults:
        """指定されたストリームのセクション結果を取得"""
        if stream_id not in self.stream_data:
            self.stream_data[stream_id] = MoleFractionSectionResults({})
        return self.stream_data[stream_id]

    def get_mole_fraction_result(self, stream_id: int, section_id: int) -> dict:
        """指定されたストリーム・セクションのモル分率結果を直接取得"""
        return self.get_stream_data(stream_id).get_mole_fraction_result(section_id)


class MoleFractionResults:
    """モル分率計算結果の集合（脱着時のみ）"""

    def __init__(self, mole_fraction_results_dict: Dict[int, Dict[int, dict]] = None):
        self.mole_fraction_stream_section_results = MoleFractionStreamSectionResults(mole_fraction_results_dict)

    def get_stream_data(self, stream_id: int) -> MoleFractionSectionResults:
        """指定されたストリームのセクション結果管理オブジェクトを取得"""
        return self.mole_fraction_stream_section_results.get_stream_data(stream_id)

    def get_result(self, stream_id: int, section_id: int) -> DesorptionMoleFractionResult:
        """指定されたストリーム・セクションのモル分率結果を直接取得"""
        return self.mole_fraction_stream_section_results.get_mole_fraction_result(stream_id, section_id)


@dataclass
class MassAndHeatBalanceResults:
    """マスバランス・熱バランス計算の統合結果"""

    mass_balance_results: MassBalanceResults
    heat_balance_results: HeatBalanceResults
    mole_fraction_results: Optional[MoleFractionResults] = None  # 脱着モードの場合のみ


@dataclass
class WallHeatFlux:
    """壁面熱流束計算結果"""

    from_inner_boundary: float
    to_outer_boundary: float
    downstream: float
    upstream: float


@dataclass
class WallHeatBalanceResult:
    temperature: float
    heat_flux: WallHeatFlux

    def to_dict(self) -> dict:
        return {
            "temp_reached": self.temperature,
            "downstream_heat_flux": self.heat_flux.downstream,
            "upstream_heat_flux": self.heat_flux.upstream,
            "heat_flux_from_inner_boundary": self.heat_flux.from_inner_boundary,
            "heat_flux_to_outer_boundary": self.heat_flux.to_outer_boundary,
        }


@dataclass
class LidHeatBalanceResult:
    """蓋熱バランス計算結果"""

    temperature: float


@dataclass
class VacuumPumpingResult:
    """真空排気計算結果"""

    pressure_loss: float
    cumulative_co2_recovered: float
    cumulative_n2_recovered: float
    co2_recovery_concentration: float
    volumetric_flow_rate: float
    remaining_moles: float
    final_pressure: float


@dataclass
class DepressurizationResult:
    """減圧計算結果"""

    final_pressure: float
    flow_rate: float  # L/min
    pressure_differential: float


@dataclass
class DownstreamFlowResult:
    """下流側流量計算結果"""

    final_pressure: float
    outlet_flows: Dict[int, GasFlow]


@dataclass
class MassBalanceCalculationResult:
    """
    物質収支計算の結果
    
    PSA担当者向け説明:
    1セルの物質収支計算結果です。
    脱着モードの場合はモル分率データも含まれます。
    
    Attributes:
        material_balance: 物質収支結果
        mole_fraction_data: モル分率データ（脱着モードのみ）
    """
    material_balance: MaterialBalanceResult
    mole_fraction_data: Optional[DesorptionMoleFractionResult] = None
    
    def has_mole_fraction_data(self) -> bool:
        """モル分率データを持っているか"""
        return self.mole_fraction_data is not None
