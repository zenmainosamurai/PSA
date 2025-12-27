"""物質収支計算モジュール

PSA担当者向け説明:
このモジュールはPSAプロセスにおける物質収支（マテリアルバランス）を計算します。

- 吸着モード: CO2が吸着材に吸着される際の物質収支
- 脱着モード: 真空引きでCO2が脱着される際の物質収支  
- 停止モード: バルブ閉鎖時（物質移動なし）

主要な関数:
- calculate_mass_balance(): モードに応じた物質収支計算の統一インターフェース
- calculate_equilibrium_loading(): 平衡吸着量の計算（吸着等温線）
"""

from typing import Optional, Tuple
import numpy as np
import CoolProp.CoolProp as CP

from operation_modes.mode_types import OperationMode, ADSORPTION_MODES
from common.constants import (
    CELSIUS_TO_KELVIN_OFFSET,
    STANDARD_MOLAR_VOLUME,
    STANDARD_PRESSURE,
    MPA_TO_PA,
    MPA_TO_KPA,
    MINUTE_TO_SECOND,
    L_TO_CM3,
    CM3_TO_L,
    MINIMUM_EQUILIBRIUM_LOADING,
    MINIMUM_CO2_PARTIAL_PRESSURE,
)

# 旧コードとの互換性のためインポート
from config.sim_conditions import TowerConditions
from core.state import (
    StateVariables,
    MaterialBalanceResult,
    GasFlow,
    GasProperties,
    AdsorptionState,
    PressureState,
    VacuumPumpingResult,
    MassBalanceResults,
    DesorptionMoleFractionResult,
)
from state.calculation_results import MassBalanceCalculationResult


# ============================================================
# メイン関数（統一インターフェース）
# ============================================================

def calculate_mass_balance(
    mode: OperationMode,
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    inflow_gas: Optional[GasFlow] = None,
    equalization_flow_rate: Optional[float] = None,
    residual_gas_composition: Optional[MassBalanceResults] = None,
    vacuum_pumping_results: Optional[VacuumPumpingResult] = None,
    previous_result: Optional[MaterialBalanceResult] = None,
) -> MassBalanceCalculationResult:
    """
    物質収支を計算する（統一インターフェース）
    
    PSA担当者向け説明:
    運転モードに応じて適切な物質収支計算を実行します。
    
    Args:
        mode: 運転モード（OperationMode Enum）
        tower_conds: 塔条件（sim_conds.xlsxから読み込んだパラメータ）
        stream: ストリーム番号 (1-indexed)
        section: セクション番号 (1-indexed)
        state_manager: 状態変数管理オブジェクト
        tower_num: 塔番号
        inflow_gas: 流入ガス情報（下流セクション・下流塔の場合）
        equalization_flow_rate: 均圧配管流量（均圧減圧モードの場合）[L/min]
        residual_gas_composition: 残留ガス組成（バッチ吸着下流の場合）
        vacuum_pumping_results: 真空排気結果（脱着モードの場合）
        previous_result: 上流セクションの計算結果（セクション2以降）
    
    Returns:
        MassBalanceCalculationResult: 物質収支計算結果
            - material_balance: MaterialBalanceResult（物質収支の詳細）
            - mole_fraction_data: 脱着モードのみ、モル分率情報
    """
    # モードに応じて計算を分岐
    if mode in ADSORPTION_MODES:
        # セクション2以降は上流結果から流入ガスを構築
        if section > 1 and previous_result is not None:
            inflow_gas = GasFlow(
                co2_volume=previous_result.outlet_gas.co2_volume,
                n2_volume=previous_result.outlet_gas.n2_volume,
                co2_mole_fraction=previous_result.outlet_gas.co2_mole_fraction,
                n2_mole_fraction=previous_result.outlet_gas.n2_mole_fraction,
            )
        
        result = _calculate_adsorption_mass_balance(
            tower_conds=tower_conds,
            stream=stream,
            section=section,
            state_manager=state_manager,
            tower_num=tower_num,
            inflow_gas=inflow_gas,
            equalization_flow_rate=equalization_flow_rate,
            residual_gas_composition=residual_gas_composition,
        )
        return MassBalanceCalculationResult(material_balance=result)
    
    elif mode == OperationMode.VACUUM_DESORPTION:
        if vacuum_pumping_results is None:
            raise ValueError("真空脱着モードでは vacuum_pumping_results が必要です")
        
        material_result, mole_fraction_result = _calculate_desorption_mass_balance(
            tower_conds=tower_conds,
            stream=stream,
            section=section,
            state_manager=state_manager,
            tower_num=tower_num,
            vacuum_pumping_results=vacuum_pumping_results,
        )
        return MassBalanceCalculationResult(
            material_balance=material_result,
            mole_fraction_data=mole_fraction_result,
        )
    
    elif mode == OperationMode.STOP:
        result = _calculate_valve_closed_mass_balance(
            stream=stream,
            section=section,
            state_manager=state_manager,
            tower_num=tower_num,
        )
        return MassBalanceCalculationResult(material_balance=result)
    
    else:
        raise ValueError(f"未対応の運転モード: {mode}")


# ============================================================
# 吸着モードの物質収支計算
# ============================================================

def _calculate_adsorption_mass_balance(
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    inflow_gas: Optional[GasFlow] = None,
    equalization_flow_rate: Optional[float] = None,
    residual_gas_composition: Optional[MassBalanceResults] = None,
) -> MaterialBalanceResult:
    """
    吸着モードの物質収支計算
    
    PSA担当者向け説明:
    ガスが充填層を通過する際のCO2吸着を計算します。
    流入ガス中のCO2が吸着材に吸着され、残りが下流へ流出します。
    """
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions

    # === セクション吸着材量 [g] ===
    section_adsorbent_mass = stream_conds[stream].adsorbent_mass / tower_conds.common.num_sections

    # === 流入ガス量の計算 [cm3] ===
    inlet_co2_volume, inlet_n2_volume = _calculate_inlet_gas_volumes(
        tower_conds=tower_conds,
        stream=stream,
        section=section,
        inflow_gas=inflow_gas,
        equalization_flow_rate=equalization_flow_rate,
    )

    # === 流入ガスモル分率の計算 ===
    if residual_gas_composition is None or section != 1:
        inlet_co2_mole_fraction = inlet_co2_volume / (inlet_co2_volume + inlet_n2_volume)
        inlet_n2_mole_fraction = inlet_n2_volume / (inlet_co2_volume + inlet_n2_volume)
    else:
        inlet_co2_mole_fraction = residual_gas_composition.get_result(stream, 1).inlet_gas.co2_mole_fraction
        inlet_n2_mole_fraction = residual_gas_composition.get_result(stream, 1).inlet_gas.n2_mole_fraction

    # === 現在の状態変数を取得 ===
    total_press = tower.total_press  # 全圧 [MPaA]
    co2_partial_pressure = total_press * inlet_co2_mole_fraction  # CO2分圧 [MPaA]
    temp = tower.cell(stream, section).temp  # 現在温度 [℃]
    current_loading = tower.cell(stream, section).loading  # 現在の既存吸着量 [cm3/g-abs]

    # === ガス物性の計算 ===
    gas_density = (
        tower_conds.feed_gas.co2_density * inlet_co2_mole_fraction
        + tower_conds.feed_gas.n2_density * inlet_n2_mole_fraction
    )  # [kg/m3]
    gas_specific_heat = (
        tower_conds.feed_gas.co2_specific_heat_capacity * inlet_co2_mole_fraction
        + tower_conds.feed_gas.n2_specific_heat_capacity * inlet_n2_mole_fraction
    )  # [kJ/kg/K]

    # === 平衡吸着量の計算 [cm3/g-abs] ===
    P_KPA = co2_partial_pressure * MPA_TO_KPA  # [kPaA]
    T_K = temp + CELSIUS_TO_KELVIN_OFFSET  # [K]
    equilibrium_loading = max(
        MINIMUM_EQUILIBRIUM_LOADING,
        calculate_equilibrium_loading(P_KPA, T_K)
    )

    # === 新規吸着量の計算 ===
    theoretical_loading_delta, actual_uptake_volume = _calculate_theoretical_uptake(
        tower_conds=tower_conds,
        equilibrium_loading=equilibrium_loading,
        current_loading=current_loading,
        section_adsorbent_mass=section_adsorbent_mass,
        inlet_co2_volume=inlet_co2_volume,
    )
    actual_loading_delta = actual_uptake_volume / section_adsorbent_mass
    updated_loading = current_loading + actual_loading_delta

    # === 流出ガス量の計算 [cm3] ===
    outlet_co2_volume = inlet_co2_volume - actual_uptake_volume
    outlet_n2_volume = inlet_n2_volume
    outlet_co2_mole_fraction = outlet_co2_volume / (outlet_co2_volume + outlet_n2_volume)
    outlet_n2_mole_fraction = outlet_n2_volume / (outlet_co2_volume + outlet_n2_volume)

    # === 流出CO2分圧の整合性調整 ===
    outlet_co2_partial_pressure = total_press * outlet_co2_mole_fraction
    previous_outlet_co2_partial_pressure = tower.cell(stream, section).outlet_co2_partial_pressure

    # 直前値より低い場合は調整
    outlet_co2_partial_pressure, actual_uptake_volume, updated_loading, outlet_co2_volume, outlet_n2_volume, \
        outlet_co2_mole_fraction, outlet_n2_mole_fraction = _adjust_outlet_pressure(
            co2_partial_pressure=co2_partial_pressure,
            previous_outlet_co2_partial_pressure=previous_outlet_co2_partial_pressure,
            outlet_co2_partial_pressure=outlet_co2_partial_pressure,
            total_press=total_press,
            inlet_co2_volume=inlet_co2_volume,
            inlet_n2_volume=inlet_n2_volume,
            section_adsorbent_mass=section_adsorbent_mass,
            current_loading=current_loading,
        )

    # === 結果オブジェクトの構築 ===
    return MaterialBalanceResult(
        inlet_gas=GasFlow(
            co2_volume=inlet_co2_volume,
            n2_volume=inlet_n2_volume,
            co2_mole_fraction=inlet_co2_mole_fraction,
            n2_mole_fraction=inlet_n2_mole_fraction,
        ),
        outlet_gas=GasFlow(
            co2_volume=outlet_co2_volume,
            n2_volume=outlet_n2_volume,
            co2_mole_fraction=outlet_co2_mole_fraction,
            n2_mole_fraction=outlet_n2_mole_fraction,
        ),
        gas_properties=GasProperties(
            density=gas_density,
            specific_heat=gas_specific_heat,
        ),
        adsorption_state=AdsorptionState(
            equilibrium_loading=equilibrium_loading,
            actual_uptake_volume=actual_uptake_volume,
            updated_loading=updated_loading,
            theoretical_loading_delta=theoretical_loading_delta,
        ),
        pressure_state=PressureState(
            co2_partial_pressure=co2_partial_pressure,
            outlet_co2_partial_pressure=outlet_co2_partial_pressure,
        ),
    )


# ============================================================
# 脱着モードの物質収支計算
# ============================================================

def _calculate_desorption_mass_balance(
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    vacuum_pumping_results: VacuumPumpingResult,
) -> Tuple[MaterialBalanceResult, DesorptionMoleFractionResult]:
    """
    脱着モードの物質収支計算
    
    PSA担当者向け説明:
    真空ポンプで減圧することで吸着材からCO2が脱着する過程を計算します。
    脱着したCO2は気相に放出され、真空ポンプで排気されます。
    """
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions

    # === 現在気相モル量の計算 ===
    # セクション空間割合
    space_ratio_section = (
        stream_conds[stream].area_fraction
        / tower_conds.common.num_sections
        * tower_conds.packed_bed.void_volume
        / tower_conds.vacuum_piping.space_volume
    )
    # セクション空間現在物質量 [mol]
    mol_amt_section = vacuum_pumping_results.remaining_moles * space_ratio_section
    
    # 現在気相モル量 [mol]
    inlet_co2_volume = mol_amt_section * tower.cell(stream, section).co2_mole_fraction
    inlet_n2_volume = mol_amt_section * tower.cell(stream, section).n2_mole_fraction
    
    # 現在気相ノルマル体積 [cm3]
    inlet_co2_volume *= STANDARD_MOLAR_VOLUME * L_TO_CM3
    inlet_n2_volume *= STANDARD_MOLAR_VOLUME * L_TO_CM3

    # === 気相放出後モル量の計算 ===
    T_K = tower.cell(stream, section).temp + CELSIUS_TO_KELVIN_OFFSET
    co2_mole_fraction = tower.cell(stream, section).co2_mole_fraction
    n2_mole_fraction = tower.cell(stream, section).n2_mole_fraction
    
    # CO2分圧 [MPaA]
    co2_partial_pressure = max(
        MINIMUM_CO2_PARTIAL_PRESSURE,
        vacuum_pumping_results.final_pressure * co2_mole_fraction
    )
    
    # セクション吸着材量 [g]
    section_adsorbent_mass = stream_conds[stream].adsorbent_mass / tower_conds.common.num_sections
    
    # 平衡吸着量 [cm3/g-abs]
    P_KPA = co2_partial_pressure * MPA_TO_KPA
    equilibrium_loading = max(
        MINIMUM_EQUILIBRIUM_LOADING,
        calculate_equilibrium_loading(P_KPA, T_K)
    )
    
    # 現在の既存吸着量 [cm3/g-abs]
    current_loading = tower.cell(stream, section).loading

    # 理論新規吸着量（脱着時は負値）
    theoretical_loading_delta, actual_uptake_volume = _calculate_theoretical_uptake(
        tower_conds=tower_conds,
        equilibrium_loading=equilibrium_loading,
        current_loading=current_loading,
        section_adsorbent_mass=section_adsorbent_mass,
        inlet_co2_volume=inlet_co2_volume,
    )
    
    # 時間経過後吸着量 [cm3/g-abs]
    theoretical_loading_delta = actual_uptake_volume / section_adsorbent_mass
    updated_loading = current_loading + theoretical_loading_delta
    
    # 気相放出CO2量 [mol]
    desorp_mw_co2 = -actual_uptake_volume / (L_TO_CM3 * STANDARD_MOLAR_VOLUME)
    
    # 気相放出後モル量 [mol]
    desorp_mw_co2_after_vacuum = inlet_co2_volume + desorp_mw_co2
    desorp_mw_n2_after_vacuum = inlet_n2_volume
    desorp_mw_all_after_vacuum = desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum
    
    # 気相放出後モル分率
    desorp_mf_co2_after_vacuum = desorp_mw_co2_after_vacuum / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)
    desorp_mf_n2_after_vacuum = desorp_mw_n2_after_vacuum / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)

    # === ガス物性の計算（CoolPropを使用）===
    P = vacuum_pumping_results.final_pressure * MPA_TO_PA
    P_ATM = STANDARD_PRESSURE
    
    gas_density = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * co2_mole_fraction
        + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * n2_mole_fraction
    )
    gas_specific_heat = (
        CP.PropsSI("CPMASS", "T", T_K, "P", P_ATM, "co2") * co2_mole_fraction
        + CP.PropsSI("CPMASS", "T", T_K, "P", P_ATM, "nitrogen") * n2_mole_fraction
    ) * 1e-3  # J/kg/K -> kJ/kg/K

    # === 結果オブジェクトの構築 ===
    material_balance_result = MaterialBalanceResult(
        inlet_gas=GasFlow(co2_volume=0, n2_volume=0, co2_mole_fraction=0, n2_mole_fraction=0),
        outlet_gas=GasFlow(co2_volume=0, n2_volume=0, co2_mole_fraction=0, n2_mole_fraction=0),
        gas_properties=GasProperties(density=gas_density, specific_heat=gas_specific_heat),
        adsorption_state=AdsorptionState(
            equilibrium_loading=equilibrium_loading,
            actual_uptake_volume=actual_uptake_volume,
            updated_loading=updated_loading,
            theoretical_loading_delta=theoretical_loading_delta,
        ),
        pressure_state=PressureState(
            co2_partial_pressure=co2_partial_pressure,
            outlet_co2_partial_pressure=tower.cell(stream, section).outlet_co2_partial_pressure,
        ),
    )

    desorption_mole_fraction_result = DesorptionMoleFractionResult(
        co2_mole_fraction_after_desorption=desorp_mf_co2_after_vacuum,
        n2_mole_fraction_after_desorption=desorp_mf_n2_after_vacuum,
        total_moles_after_desorption=desorp_mw_all_after_vacuum,
    )

    return material_balance_result, desorption_mole_fraction_result


# ============================================================
# 停止モードの物質収支計算
# ============================================================

def _calculate_valve_closed_mass_balance(
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
) -> MaterialBalanceResult:
    """
    停止モードの物質収支計算
    
    PSA担当者向け説明:
    バルブが閉じている状態では、ガスの流入・流出がないため
    吸着量は変化しません。現在の状態を維持します。
    """
    tower = state_manager.towers[tower_num]
    
    return MaterialBalanceResult(
        inlet_gas=GasFlow(co2_volume=0, n2_volume=0, co2_mole_fraction=0, n2_mole_fraction=0),
        outlet_gas=GasFlow(co2_volume=0, n2_volume=0, co2_mole_fraction=0, n2_mole_fraction=0),
        gas_properties=GasProperties(density=0, specific_heat=0),
        adsorption_state=AdsorptionState(
            equilibrium_loading=0,
            actual_uptake_volume=0,
            updated_loading=tower.cell(stream, section).loading,
            theoretical_loading_delta=0,
        ),
        pressure_state=PressureState(
            co2_partial_pressure=0,
            outlet_co2_partial_pressure=tower.cell(stream, section).outlet_co2_partial_pressure,
        ),
    )


# ============================================================
# 共通ヘルパー関数
# ============================================================

def calculate_equilibrium_loading(pressure_kpa: float, temperature_k: float) -> float:
    """
    平衡吸着量を計算（吸着等温線）
    
    PSA担当者向け説明:
    与えられたCO2分圧と温度における平衡吸着量を計算します。
    シンボリック回帰による近似式を使用しています。
    
    Args:
        pressure_kpa: CO2分圧 [kPaA]
        temperature_k: 温度 [K]
    
    Returns:
        平衡吸着量 [cm3/g-abs]
    """
    P = pressure_kpa
    T = temperature_k
    
    # シンボリック回帰による近似式
    equilibrium_loading = (
        P
        * (252.0724 - 0.50989705 * T)
        / (P - 3554.54819062669 * (1 - 0.0655247236249063 * np.sqrt(T)) ** 3 + 1.7354268)
    )
    return equilibrium_loading


def _calculate_inlet_gas_volumes(
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    inflow_gas: Optional[GasFlow],
    equalization_flow_rate: Optional[float],
) -> Tuple[float, float]:
    """流入ガス量を計算 [cm3]"""
    stream_conds = tower_conds.stream_conditions
    
    if section == 1 and inflow_gas is None and equalization_flow_rate is None:
        # 最上流セル: 導入ガスから計算
        inlet_co2_volume = (
            tower_conds.feed_gas.co2_flow_rate
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
            * L_TO_CM3
        )
        inlet_n2_volume = (
            tower_conds.feed_gas.n2_flow_rate
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
            * L_TO_CM3
        )
    elif section == 1 and equalization_flow_rate is not None:
        # 均圧減圧時の最上流セル
        inlet_co2_volume = (
            tower_conds.feed_gas.co2_mole_fraction
            * equalization_flow_rate
            * L_TO_CM3
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
        )
        inlet_n2_volume = (
            tower_conds.feed_gas.n2_mole_fraction
            * equalization_flow_rate
            * L_TO_CM3
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
        )
    elif inflow_gas is not None:
        # 下流セクションまたは下流塔
        inlet_co2_volume = inflow_gas.co2_volume
        inlet_n2_volume = inflow_gas.n2_volume
    else:
        raise ValueError("流入ガス情報が不足しています")
    
    return inlet_co2_volume, inlet_n2_volume


def _calculate_theoretical_uptake(
    tower_conds: TowerConditions,
    equilibrium_loading: float,
    current_loading: float,
    section_adsorbent_mass: float,
    inlet_co2_volume: float,
) -> Tuple[float, float]:
    """
    理論新規吸着量を計算
    
    PSA担当者向け説明:
    平衡吸着量と現在吸着量の差から、理論的な新規吸着量を計算します。
    物質移動係数（LDF: Linear Driving Force）モデルを使用しています。
    
    Returns:
        (theoretical_loading_delta, actual_uptake_volume)
    """
    if equilibrium_loading >= current_loading:
        # 吸着モード
        theoretical_loading_delta = (
            tower_conds.packed_bed.adsorption_mass_transfer_coef ** (current_loading / equilibrium_loading)
            / tower_conds.packed_bed.adsorbent_bulk_density
            * 6
            * (1 - tower_conds.packed_bed.average_porosity)
            * tower_conds.packed_bed.particle_shape_factor
            / tower_conds.packed_bed.average_particle_diameter
            * (equilibrium_loading - current_loading)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
            / 1e6
        )
        theoretical_uptake_volume = theoretical_loading_delta * section_adsorbent_mass
        actual_uptake_volume = min(theoretical_uptake_volume, inlet_co2_volume)
    else:
        # 脱着モード
        theoretical_loading_delta = (
            tower_conds.packed_bed.desorption_mass_transfer_coef
            / tower_conds.packed_bed.adsorbent_bulk_density
            * 6
            * (1 - tower_conds.packed_bed.average_porosity)
            * tower_conds.packed_bed.particle_shape_factor
            / tower_conds.packed_bed.average_particle_diameter
            * (equilibrium_loading - current_loading)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
            / 1e6
        )
        theoretical_uptake_volume = theoretical_loading_delta * section_adsorbent_mass
        actual_uptake_volume = max(theoretical_uptake_volume, -current_loading)

    return theoretical_loading_delta, actual_uptake_volume


def _adjust_outlet_pressure(
    co2_partial_pressure: float,
    previous_outlet_co2_partial_pressure: float,
    outlet_co2_partial_pressure: float,
    total_press: float,
    inlet_co2_volume: float,
    inlet_n2_volume: float,
    section_adsorbent_mass: float,
    current_loading: float,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    流出CO2分圧の整合性調整
    
    PSA担当者向け説明:
    流出CO2分圧が直前値より低くなる場合、物理的に不自然なため
    吸着量を逆算して調整します。
    """
    if co2_partial_pressure >= previous_outlet_co2_partial_pressure:
        if outlet_co2_partial_pressure < previous_outlet_co2_partial_pressure:
            outlet_co2_partial_pressure = previous_outlet_co2_partial_pressure
            actual_uptake_volume = inlet_co2_volume - outlet_co2_partial_pressure * inlet_n2_volume / (
                total_press - outlet_co2_partial_pressure
            )
            actual_loading_delta = actual_uptake_volume / section_adsorbent_mass
            updated_loading = current_loading + actual_loading_delta
            outlet_co2_volume = inlet_co2_volume - actual_uptake_volume
            outlet_n2_volume = inlet_n2_volume
            outlet_co2_mole_fraction = outlet_co2_volume / (outlet_co2_volume + outlet_n2_volume)
            outlet_n2_mole_fraction = outlet_n2_volume / (outlet_co2_volume + outlet_n2_volume)
            return (outlet_co2_partial_pressure, actual_uptake_volume, updated_loading, 
                    outlet_co2_volume, outlet_n2_volume, outlet_co2_mole_fraction, outlet_n2_mole_fraction)
    else:
        if outlet_co2_partial_pressure < co2_partial_pressure:
            outlet_co2_partial_pressure = co2_partial_pressure
            actual_uptake_volume = inlet_co2_volume - outlet_co2_partial_pressure * inlet_n2_volume / (
                total_press - outlet_co2_partial_pressure
            )
            actual_loading_delta = actual_uptake_volume / section_adsorbent_mass
            updated_loading = current_loading + actual_loading_delta
            outlet_co2_volume = inlet_co2_volume - actual_uptake_volume
            outlet_n2_volume = inlet_n2_volume
            outlet_co2_mole_fraction = outlet_co2_volume / (outlet_co2_volume + outlet_n2_volume)
            outlet_n2_mole_fraction = outlet_n2_volume / (outlet_co2_volume + outlet_n2_volume)
            return (outlet_co2_partial_pressure, actual_uptake_volume, updated_loading,
                    outlet_co2_volume, outlet_n2_volume, outlet_co2_mole_fraction, outlet_n2_mole_fraction)
    
    # 調整不要の場合は None を返す（呼び出し側で元の値を使用）
    outlet_co2_volume = inlet_co2_volume - (updated_loading - current_loading) * section_adsorbent_mass
    outlet_n2_volume = inlet_n2_volume
    outlet_co2_mole_fraction = outlet_co2_volume / (outlet_co2_volume + outlet_n2_volume) if (outlet_co2_volume + outlet_n2_volume) > 0 else 0
    outlet_n2_mole_fraction = outlet_n2_volume / (outlet_co2_volume + outlet_n2_volume) if (outlet_co2_volume + outlet_n2_volume) > 0 else 0
    # 元の計算値を返す
    return (outlet_co2_partial_pressure, (updated_loading - current_loading) * section_adsorbent_mass, updated_loading,
            outlet_co2_volume, outlet_n2_volume, outlet_co2_mole_fraction, outlet_n2_mole_fraction)
