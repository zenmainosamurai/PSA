# calculation_results.py (新規ファイル)
from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class MaterialBalanceResult:
    """マテリアルバランス計算結果"""

    inlet_co2_volume: float
    inlet_n2_volume: float
    inlet_co2_mole_fraction: float
    inlet_n2_mole_fraction: float
    gas_density: float
    gas_specific_heat: float
    equilibrium_loading: float
    actual_uptake_volume: float
    updated_loading: float
    outlet_co2_volume: float
    outlet_n2_volume: float
    outlet_co2_mole_fraction: float
    outlet_n2_mole_fraction: float
    theoretical_loading_delta: float
    co2_partial_pressure: float
    outlet_co2_partial_pressure: float


@dataclass
class HeatBalanceResult:
    """熱バランス計算結果"""

    temp_reached: float
    temp_thermocouple_reached: float
    hw1: float  # 壁-層伝熱係数
    u1: float  # 層伝熱係数
    Hroof: float  # 上流セルへの熱流束
    Hbb: float  # 下流セルへの熱流束
    Habs: float  # 発生する吸着熱
    Hwin: float  # 内側境界からの熱流束
    Hwout: float  # 外側境界への熱流束


@dataclass
class WallHeatBalanceResult:
    """壁面熱バランス計算結果"""

    temp_reached: float
    Hbb: float  # 下流壁への熱流束
    Hroof: float  # 上流壁への熱流束
    Hwin: float  # 内側境界からの熱流束
    Hwout: float  # 外側境界への熱流束


@dataclass
class LidHeatBalanceResult:
    """蓋熱バランス計算結果"""

    temp_reached: float


@dataclass
class DesorptionMoleFractionResult:
    """脱着時のモル分率計算結果"""

    mf_co2_after_vacuum: float
    mf_n2_after_vacuum: float
    desorp_mw_all_after_vacuum: float


@dataclass
class VacuumPumpingResult:
    """真空排気計算結果"""

    P_resist: float
    accum_vacuum_amt_co2: float
    accum_vacuum_amt_n2: float
    vacuum_co2_mf: float
    vacuum_rate_N: float
    case_inner_mol_amt_after_vacuum: float
    total_press_after_vacuum: float


@dataclass
class DepressurizationResult:
    """減圧計算結果"""

    total_press_after_depressure: float
    flow_amount_l: float
    diff_press: float


@dataclass
class DownstreamFlowResult:
    """下流側流量計算結果"""

    total_press_after_depressure_downflow: float
    outflow_fr: Dict[int, Dict[str, float]]


@dataclass
class SectionResults:
    """各セクションの計算結果"""

    material: MaterialBalanceResult
    heat: HeatBalanceResult
    mole_fraction: Optional[DesorptionMoleFractionResult] = None


@dataclass
class TowerCalculationResult:
    """塔全体の計算結果"""

    # ストリーム・セクション毎の結果
    sections: Dict[int, Dict[int, SectionResults]]  # sections[stream][section]

    # 壁面・蓋の結果
    wall_heat: Dict[int, WallHeatBalanceResult]  # wall_heat[section]
    lid_heat: Dict[str, LidHeatBalanceResult]  # lid_heat["up" or "down"]

    # 圧力関連の結果
    pressure_after_batch_adsorption: Optional[float] = None
    pressure_after_desorption: Optional[float] = None
    total_pressure: Optional[float] = None

    # 真空排気・減圧関連
    vacuum_pumping: Optional[VacuumPumpingResult] = None
    depressurization: Optional[DepressurizationResult] = None
    downstream_flow: Optional[DownstreamFlowResult] = None

    # その他の状態変数（記録用）
    others: Optional[Dict[str, Any]] = None
