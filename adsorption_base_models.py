import numpy as np
import pandas as pd
from scipy import optimize
import CoolProp.CoolProp as CP

from utils.heat_transfer import calc_heat_transfer_coef as _heat_transfer_coef
from state_variables import StateVariables
from sim_conditions import TowerConditions
from adsorption_results import (
    HeatBalanceResult,
    MaterialBalanceResult,
    DesorptionMoleFractionResult,
    GasFlow,
    GasProperties,
    AdsorptionState,
    PressureState,
    VacuumPumpingResult,
    HeatTransferCoefficients,
    HeatFlux,
    CellTemperatures,
    WallHeatFlux,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    DepressurizationResult,
    DownstreamFlowResult,
)


import warnings

warnings.simplefilter("error")


def calculate_mass_balance_for_adsorption(
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    inflow_gas=None,
    flow_amt_depress=None,
    residual_gas_composition=None,
) -> MaterialBalanceResult:
    """任意セルのマテリアルバランスを計算する
        吸着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        inflow_gas (dict): 上流セル・塔からの流出ガス情報
        flow_amt_depress (float): 上流均圧菅からの全流量

    Returns:
        MaterialBalanceResult: 対象セルのマテリアルバランスに関する計算結果
    """
    ### マテバラ計算開始 ------------------------------------------------
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions

    # セクション吸着材量 [g]
    section_adsorbent_mass = stream_conds[stream].adsorbent_mass / tower_conds.common.num_sections
    # 流入CO2, N2流量 [cm3]
    if (section == 1) & (inflow_gas is None) & (flow_amt_depress is None):  # 最上流セルの流入ガスによる吸着など
        inlet_co2_volume = (
            tower_conds.feed_gas.co2_flow_rate
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
            * 1000
        )
        inlet_n2_volume = (
            tower_conds.feed_gas.n2_flow_rate
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
            * 1000
        )
    elif (section == 1) & (flow_amt_depress is not None):  # 減圧時の最上流セルのみ対象
        inlet_co2_volume = (
            tower_conds.feed_gas.co2_mole_fraction
            * flow_amt_depress
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
            * 1000
        )
        inlet_n2_volume = (
            tower_conds.feed_gas.n2_mole_fraction
            * flow_amt_depress
            * tower_conds.common.calculation_step_time
            * stream_conds[stream].area_fraction
            * 1000
        )
    elif inflow_gas is not None:  # 下流セクションや下流塔での吸着など
        inlet_co2_volume = inflow_gas["outlet_co2_volume"]
        inlet_n2_volume = inflow_gas["outlet_n2_volume"]
    # 流入ガスモル分率
    if (residual_gas_composition is None) | (section != 1):
        inlet_co2_mole_fraction = inlet_co2_volume / (inlet_co2_volume + inlet_n2_volume)
        inlet_n2_mole_fraction = inlet_n2_volume / (inlet_co2_volume + inlet_n2_volume)
    else:
        inlet_co2_mole_fraction = residual_gas_composition[stream][1]["inlet_co2_mole_fraction"]
        inlet_n2_mole_fraction = residual_gas_composition[stream][1]["inlet_n2_mole_fraction"]
    # 全圧 [MPaA]
    total_press = tower.total_press
    # CO2分圧 [MPaA]
    co2_partial_pressure = total_press * inlet_co2_mole_fraction
    # 現在温度 [℃]
    temp = tower.temp[stream - 1, section - 1]
    # ガス密度 [kg/m3]
    gas_density = (
        tower_conds.feed_gas.co2_density * inlet_co2_mole_fraction
        + tower_conds.feed_gas.n2_density * inlet_n2_mole_fraction
    )
    # ガス比熱 [kJ/kg/K]
    gas_specific_heat = (
        tower_conds.feed_gas.co2_specific_heat_capacity * inlet_co2_mole_fraction
        + tower_conds.feed_gas.n2_specific_heat_capacity * inlet_n2_mole_fraction
    )
    # 現在雰囲気の平衡吸着量 [cm3/g-abs]
    P_KPA = co2_partial_pressure * 1000  # [kPaA]
    T_K = temp + 273.15  # [K]
    equilibrium_loading = max(0.1, _calculate_equilibrium_adsorption_amount(P_KPA, T_K))
    # 現在の既存吸着量 [cm3/g-abs]
    current_loading = tower.adsorp_amt[stream - 1, section - 1]
    # 理論新規吸着量 [cm3/g-abs]
    if equilibrium_loading >= current_loading:
        theoretical_loading_delta = (
            tower_conds.packed_bed.adsorption_mass_transfer_coef ** (current_loading / equilibrium_loading)
            / tower_conds.packed_bed.adsorbent_bulk_density
            * 6
            * (1 - tower_conds.packed_bed.average_porosity)
            * tower_conds.packed_bed.particle_shape_factor
            / tower_conds.packed_bed.average_particle_diameter
            * (equilibrium_loading - current_loading)
            * tower_conds.common.calculation_step_time
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        theoretical_uptake_volume = theoretical_loading_delta * section_adsorbent_mass
        # 実際のセクション新規吸着量 [cm3]
        actual_uptake_volume = min(theoretical_uptake_volume, inlet_co2_volume)
    else:
        theoretical_loading_delta = (
            tower_conds.packed_bed.desorption_mass_transfer_coef ** (current_loading / equilibrium_loading)
            / tower_conds.packed_bed.adsorbent_bulk_density
            * 6
            * (1 - tower_conds.packed_bed.average_porosity)
            * tower_conds.packed_bed.particle_shape_factor
            / tower_conds.packed_bed.average_particle_diameter
            * (equilibrium_loading - current_loading)
            * tower_conds.common.calculation_step_time
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        theoretical_uptake_volume = theoretical_loading_delta * section_adsorbent_mass
        # 実際のセクション新規吸着量 [cm3]
        actual_uptake_volume = max(theoretical_uptake_volume, -current_loading)
    # 実際の新規吸着量 [cm3/g-abs]
    actual_loading_delta = actual_uptake_volume / section_adsorbent_mass
    # 時間経過後吸着量 [cm3/g-abs]
    updated_loading = current_loading + actual_loading_delta
    # 下流流出CO2流量 [cm3]
    outlet_co2_volume = inlet_co2_volume - actual_uptake_volume
    # 下流流出N2流量 [cm3]
    outlet_n2_volume = inlet_n2_volume
    # 流出CO2分率
    outlet_co2_mole_fraction = outlet_co2_volume / (outlet_co2_volume + outlet_n2_volume)
    # 流出N2分率
    outlet_n2_mole_fraction = outlet_n2_volume / (outlet_co2_volume + outlet_n2_volume)

    ### 流出CO2分圧のつじつま合わせ ------------------------------

    # 流出CO2分圧 [MPaA]
    outlet_co2_partial_pressure = total_press * outlet_co2_mole_fraction
    # 直前値より低い場合は直前値と同じになるように吸着量を変更
    previous_outlet_co2_partial_pressure = tower.outlet_co2_partial_pressure[stream - 1, section - 1]
    if co2_partial_pressure >= previous_outlet_co2_partial_pressure:
        if outlet_co2_partial_pressure < previous_outlet_co2_partial_pressure:
            # 直前値に置換
            outlet_co2_partial_pressure = previous_outlet_co2_partial_pressure
            # セクション新規吸着量とそれに伴う各変数を逆算
            actual_uptake_volume = inlet_co2_volume - outlet_co2_partial_pressure * inlet_n2_volume / (
                total_press - outlet_co2_partial_pressure
            )
            # 実際の新規吸着量 [cm3/g-abs]
            actual_loading_delta = actual_uptake_volume / section_adsorbent_mass
            # 時間経過後吸着量 [cm3/g-abs]
            updated_loading = current_loading + actual_loading_delta
            # 下流流出CO2流量 [cm3]
            outlet_co2_volume = inlet_co2_volume - actual_uptake_volume
            # 下流流出N2流量 [cm3]
            outlet_n2_volume = inlet_n2_volume
            # 流出CO2分率
            outlet_co2_mole_fraction = outlet_co2_volume / (outlet_co2_volume + outlet_n2_volume)
            # 流出N2分率
            outlet_n2_mole_fraction = outlet_n2_volume / (outlet_co2_volume + outlet_n2_volume)
    else:
        if outlet_co2_partial_pressure < co2_partial_pressure:
            # co2分圧に置換
            outlet_co2_partial_pressure = co2_partial_pressure
            # セクション新規吸着量とそれに伴う各変数を逆算
            actual_uptake_volume = inlet_co2_volume - outlet_co2_partial_pressure * inlet_n2_volume / (
                total_press - outlet_co2_partial_pressure
            )
            # 実際の新規吸着量 [cm3/g-abs]
            actual_loading_delta = actual_uptake_volume / section_adsorbent_mass
            # 時間経過後吸着量 [cm3/g-abs]
            updated_loading = current_loading + actual_loading_delta
            # 下流流出CO2流量 [cm3]
            outlet_co2_volume = inlet_co2_volume - actual_uptake_volume
            # 下流流出N2流量 [cm3]
            outlet_n2_volume = inlet_n2_volume
            # 流出CO2分率
            outlet_co2_mole_fraction = outlet_co2_volume / (outlet_co2_volume + outlet_n2_volume)
            # 流出N2分率
            outlet_n2_mole_fraction = outlet_n2_volume / (outlet_co2_volume + outlet_n2_volume)

    inlet_gas = GasFlow(
        co2_volume=inlet_co2_volume,
        n2_volume=inlet_n2_volume,
        co2_mole_fraction=inlet_co2_mole_fraction,
        n2_mole_fraction=inlet_n2_mole_fraction,
    )
    outlet_gas = GasFlow(
        co2_volume=outlet_co2_volume,
        n2_volume=outlet_n2_volume,
        co2_mole_fraction=outlet_co2_mole_fraction,
        n2_mole_fraction=outlet_n2_mole_fraction,
    )
    gas_properties = GasProperties(
        density=gas_density,
        specific_heat=gas_specific_heat,
    )
    adsorption_state = AdsorptionState(
        equilibrium_loading=equilibrium_loading,
        actual_uptake_volume=actual_uptake_volume,
        updated_loading=updated_loading,
        theoretical_loading_delta=theoretical_loading_delta,
    )
    pressure_state = PressureState(
        co2_partial_pressure=co2_partial_pressure,
        outlet_co2_partial_pressure=outlet_co2_partial_pressure,
    )
    material_balance_result = MaterialBalanceResult(
        inlet_gas=inlet_gas,
        outlet_gas=outlet_gas,
        gas_properties=gas_properties,
        adsorption_state=adsorption_state,
        pressure_state=pressure_state,
    )
    output = material_balance_result.to_dict()

    return output


def calculate_mass_balance_for_desorption(
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    vacuum_pumping_results,
) -> tuple[MaterialBalanceResult, DesorptionMoleFractionResult]:
    """任意セルのマテリアルバランスを計算する
        脱着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vacuum_pumping_results (dict): 排気後圧力計算の出力

    Returns:
        MaterialBalanceResult: 対象セルのマテリアルバランスに関する計算結果
        DesorptionMoleFractionResult: 脱着後のモル分率計算結果
    """
    ### 現在気相モル量 = 流入モル量の計算 -------------------
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions
    # セクション空間割合
    space_ratio_section = (
        stream_conds[stream].area_fraction
        / tower_conds.common.num_sections
        * tower_conds.packed_bed.void_volume
        / tower_conds.vacuum_piping.space_volume
    )
    # セクション空間現在物質量 [mol]
    mol_amt_section = vacuum_pumping_results["case_inner_mol_amt_after_vacuum"] * space_ratio_section
    # 現在気相モル量 [mol]
    inlet_co2_volume = mol_amt_section * tower.mf_co2[stream - 1, section - 1]
    inlet_n2_volume = mol_amt_section * tower.mf_n2[stream - 1, section - 1]
    # 現在気相ノルマル体積(=流入量) [cm3]
    inlet_co2_volume *= 22.4 * 1000
    inlet_n2_volume *= 22.4 * 1000

    ### 気相放出後モル量の計算 -----------------------------
    # 現在温度 [℃]
    T_K = tower.temp[stream - 1, section - 1] + 273.15
    # モル分率
    mf_co2 = tower.mf_co2[stream - 1, section - 1]
    mf_n2 = tower.mf_n2[stream - 1, section - 1]
    # CO2分圧 [MPaA]
    co2_partial_pressure = max(2.5e-3, vacuum_pumping_results["total_press_after_vacuum"] * mf_co2)
    # セクション吸着材量 [g]
    section_adsorbent_mass = stream_conds[stream].adsorbent_mass / tower_conds.common.num_sections
    # 現在雰囲気の平衡吸着量 [cm3/g-abs]
    P_KPA = co2_partial_pressure * 1000  # [MPaA] → [kPaA]
    equilibrium_loading = max(0.1, _calculate_equilibrium_adsorption_amount(P_KPA, T_K))
    # 現在の既存吸着量 [cm3/g-abs]
    current_loading = tower.adsorp_amt[stream - 1, section - 1]
    # 理論新規吸着量 [cm3/g-abs]
    if equilibrium_loading >= current_loading:
        theoretical_loading_delta = (
            tower_conds.packed_bed.adsorption_mass_transfer_coef ** (current_loading / equilibrium_loading)
            / tower_conds.packed_bed.adsorbent_bulk_density
            * 6
            * (1 - tower_conds.packed_bed.average_porosity)
            * tower_conds.packed_bed.particle_shape_factor
            / tower_conds.packed_bed.average_particle_diameter
            * (equilibrium_loading - current_loading)
            * tower_conds.common.calculation_step_time
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        theoretical_uptake_volume = theoretical_loading_delta * section_adsorbent_mass
        # 実際のセクション新規吸着量 [cm3]
        actual_uptake_volume = min(theoretical_uptake_volume, inlet_co2_volume)
    else:
        theoretical_loading_delta = (
            tower_conds.packed_bed.desorption_mass_transfer_coef ** (current_loading / equilibrium_loading)
            / tower_conds.packed_bed.adsorbent_bulk_density
            * 6
            * (1 - tower_conds.packed_bed.average_porosity)
            * tower_conds.packed_bed.particle_shape_factor
            / tower_conds.packed_bed.average_particle_diameter
            * (equilibrium_loading - current_loading)
            * tower_conds.common.calculation_step_time
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        theoretical_uptake_volume = theoretical_loading_delta * section_adsorbent_mass
        # 実際のセクション新規吸着量 [cm3]
        actual_uptake_volume = max(theoretical_uptake_volume, -current_loading)
    # 実際の新規吸着量 [cm3/g-abs]
    theoretical_loading_delta = actual_uptake_volume / section_adsorbent_mass
    # 時間経過後吸着量 [cm3/g-abs]
    updated_loading = current_loading + theoretical_loading_delta
    # 気相放出CO2量 [cm3]
    desorp_mw_co2 = -1 * actual_uptake_volume
    # 気相放出CO2量 [mol]
    desorp_mw_co2 = desorp_mw_co2 / 1000 / 22.4
    # 気相放出後モル量 [mol]
    desorp_mw_co2_after_vacuum = inlet_co2_volume + desorp_mw_co2
    desorp_mw_n2_after_vacuum = inlet_n2_volume
    # 気相放出後全物質量 [mol]
    desorp_mw_all_after_vacuum = desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum
    # 気相放出後モル分率
    desorp_mf_co2_after_vacuum = desorp_mw_co2_after_vacuum / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)
    desorp_mf_n2_after_vacuum = desorp_mw_n2_after_vacuum / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)

    ### その他（熱バラ渡す用） ---------------------------------------
    P = vacuum_pumping_results["total_press_after_vacuum"] * 1e6
    P_ATM = 0.101325 * 1e6
    # ガス密度 [kg/m3]
    gas_density = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * mf_co2 + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * mf_n2
    )
    # ガス比熱 [kJ/kg/K]
    # NOTE: 比熱と熱伝導率と粘度は大気圧を使用
    gas_specific_heat = (
        CP.PropsSI("CPMASS", "T", T_K, "P", P_ATM, "co2") * mf_co2
        + CP.PropsSI("CPMASS", "T", T_K, "P", P_ATM, "nitrogen") * mf_n2
    ) * 1e-3

    inlet_gas = GasFlow(
        co2_volume=0,
        n2_volume=0,
        co2_mole_fraction=0,
        n2_mole_fraction=0,
    )
    outlet_gas = GasFlow(
        co2_volume=0,
        n2_volume=0,
        co2_mole_fraction=0,
        n2_mole_fraction=0,
    )
    gas_properties = GasProperties(
        density=gas_density,
        specific_heat=gas_specific_heat,
    )
    adsorption_state = AdsorptionState(
        equilibrium_loading=equilibrium_loading,
        actual_uptake_volume=actual_uptake_volume,
        updated_loading=updated_loading,
        theoretical_loading_delta=theoretical_loading_delta,
    )
    pressure_state = PressureState(
        co2_partial_pressure=co2_partial_pressure,
        outlet_co2_partial_pressure=tower.outlet_co2_partial_pressure[stream - 1, section - 1],
    )
    material_balance_result = MaterialBalanceResult(
        inlet_gas=inlet_gas,
        outlet_gas=outlet_gas,
        gas_properties=gas_properties,
        adsorption_state=adsorption_state,
        pressure_state=pressure_state,
    )
    output = material_balance_result.to_dict()

    desorption_mole_fraction_result = DesorptionMoleFractionResult(
        co2_mole_fraction_after_desorption=desorp_mf_co2_after_vacuum,
        n2_mole_fraction_after_desorption=desorp_mf_n2_after_vacuum,
        total_moles_after_desorption=desorp_mw_all_after_vacuum,
    )
    output2 = desorption_mole_fraction_result.to_dict()

    return output, output2


def calculate_mass_balance_for_valve_closed(stream: int, section: int, state_manager: StateVariables, tower_num: int):
    """任意セルのマテリアルバランスを計算する
        弁停止モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vacuum_pumping_results (dict): 排気後圧力計算の出力

    Returns:
        dict: 対象セルの計算結果
    """
    tower = state_manager.towers[tower_num]
    inlet_gas = GasFlow(
        co2_volume=0,
        n2_volume=0,
        co2_mole_fraction=0,
        n2_mole_fraction=0,
    )
    output_gas = GasFlow(
        co2_volume=0,
        n2_volume=0,
        co2_mole_fraction=0,
        n2_mole_fraction=0,
    )
    gas_properties = GasProperties(
        density=0,
        specific_heat=0,
    )
    adsorption_state = AdsorptionState(
        equilibrium_loading=0,
        actual_uptake_volume=0,
        updated_loading=tower.adsorp_amt[stream - 1, section - 1],
        theoretical_loading_delta=0,
    )
    pressure_state = PressureState(
        co2_partial_pressure=0,
        outlet_co2_partial_pressure=tower.outlet_co2_partial_pressure[stream - 1, section - 1],
    )
    material_balance_result = MaterialBalanceResult(
        inlet_gas=inlet_gas,
        outlet_gas=output_gas,
        gas_properties=gas_properties,
        adsorption_state=adsorption_state,
        pressure_state=pressure_state,
    )
    output = material_balance_result.to_dict()

    return output


def calculate_heat_balance_for_bed(
    tower_conds: TowerConditions,
    stream,
    section,
    state_manager: StateVariables,
    tower_num,
    mode,
    material_output=None,
    heat_output=None,
    vacuum_pumping_results=None,
):
    """対象セルの熱バランスを計算する

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 状態変数
        mode (int): 吸着・脱着等の運転モード
        material_output (dict): 対象セルのマテバラ出力
        heat_output (dict): 上流セクションの熱バラ出力
        vacuum_pumping_results (dict): 排気後圧力計算の出力（脱着時）

    Returns:
        dict: 対象セルの熱バラ出力
    """
    ### 前準備 ------------------------------------------------------
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions

    # セクション現在温度 [℃]
    temp_now = tower.temp[stream - 1, section - 1]
    # 内側セクション温度 [℃]
    if stream == 1:
        temp_inside_cell = 18
    else:
        temp_inside_cell = tower.temp[stream - 2, section - 1]
    # 外側セクション温度 [℃]
    if stream != tower_conds.common.num_streams:
        temp_outside_cell = tower.temp[stream, section - 1]
    else:
        temp_outside_cell = tower.temp_wall[section - 1]
    # 下流セクション温度 [℃]
    if section != tower_conds.common.num_sections:
        temp_below_cell = tower.temp[stream - 1, section]
    # 発生する吸着熱 [J]
    if mode == 1:
        Habs = 0  # 弁停止モードでは0
    else:
        Habs = (
            material_output["actual_uptake_volume"]
            / 1000
            / 22.4
            * tower_conds.feed_gas.co2_molecular_weight
            * tower_conds.feed_gas.co2_adsorption_heat
        )
    # 流入ガス質量 [g]
    if mode in [1, 2]:
        Mgas = 0  # 弁停止・脱着モードでは0
    else:
        Mgas = (
            material_output["inlet_co2_volume"] / 1000 / 22.4 * tower_conds.feed_gas.co2_molecular_weight
            + material_output["inlet_n2_volume"] / 1000 / 22.4 * tower_conds.feed_gas.n2_molecular_weight
        )
    # 流入ガス比熱 [J/g/K]
    if mode == 1:
        gas_specific_heat = 0  # 弁停止モードでは0
    else:
        gas_specific_heat = material_output["gas_specific_heat"]
    # 内側境界面積 [m2]
    Ain = stream_conds[stream].inner_boundary_area / tower_conds.common.num_sections
    # 外側境界面積 [m2]
    Aout = stream_conds[stream].outer_boundary_area / tower_conds.common.num_sections
    # 下流セル境界面積 [m2]
    Abb = stream_conds[stream].cross_section
    # 壁-層伝熱係数、層伝熱係数
    if mode == 0:
        hw1, u1 = _heat_transfer_coef(
            tower_conds,
            stream,
            section,
            temp_now,
            mode,
            state_manager,
            tower_num,
            material_output,
        )
    elif mode == 1:  # 弁停止モードでは直前値に置換
        hw1 = tower.heat_t_coef[stream - 1, section - 1]
        u1 = tower.heat_t_coef_wall[stream - 1, section - 1]
    elif mode == 2:  # 脱着モードでは入力に排気後圧力計算の出力を使用
        # hw1 = variables["heat_t_coef"][stream][section]
        # u1 = variables["heat_t_coef_wall"][stream][section]
        hw1, u1 = _heat_transfer_coef(
            tower_conds,
            stream,
            section,
            temp_now,
            mode,
            state_manager,
            tower_num,
            material_output,
            vacuum_pumping_results,
        )

    ### 熱流束計算 ---------------------------------------------------

    # 内側境界からの熱流束 [J]
    if stream == 1:
        Hwin = 0
    else:
        Hwin = u1 * Ain * (temp_inside_cell - temp_now) * tower_conds.common.calculation_step_time * 60
    # 外側境界への熱流束 [J]
    if stream == tower_conds.common.num_streams:
        Hwout = hw1 * Aout * (temp_now - temp_outside_cell) * tower_conds.common.calculation_step_time * 60
    else:
        Hwout = u1 * Aout * (temp_now - temp_outside_cell) * tower_conds.common.calculation_step_time * 60
    # 下流セルへの熱流束 [J]
    if section == tower_conds.common.num_sections:
        Hbb = (  # 下蓋への熱流束
            hw1
            * stream_conds[stream].cross_section
            * (temp_now - tower.temp_lid_down)
            * tower_conds.common.calculation_step_time
            * 60
        )
    else:
        Hbb = (
            u1 * Abb * (temp_now - temp_below_cell) * tower_conds.common.calculation_step_time * 60
        )  # 下流セルへの熱流束
    # 上流セルヘの熱流束 [J]
    if section == 1:  # 上蓋からの熱流束
        Hroof = (
            hw1
            * stream_conds[stream].cross_section
            * (temp_now - tower.temp_lid_up)
            * tower_conds.common.calculation_step_time
            * 60
        )
    else:  # 上流セルヘの熱流束 = -1 * 上流セルの「下流セルへの熱流束」
        Hroof = -heat_output["Hbb"]

    ### 到達温度計算 --------------------------------------------------------------

    # セクション到達温度 [℃]
    args = {
        "tower_conds": tower_conds,
        "gas_specific_heat": gas_specific_heat,
        "Mgas": Mgas,
        "temp_now": temp_now,
        "Habs": Habs,
        "Hwin": Hwin,
        "Hwout": Hwout,
        "Hbb": Hbb,
        "Hroof": Hroof,
        "stream": stream,
    }
    temp_reached = optimize.newton(_optimize_bed_temperature, temp_now, args=args.values())

    ### 熱電対温度の計算 --------------------------------------------------------------

    # 熱電対熱容量 [J/K]
    heat_capacity = tower_conds.thermocouple.specific_heat * tower_conds.thermocouple.weight
    # 熱電対側面積 [m2]
    S_side = 0.004 * np.pi * 0.1
    # 熱電対伝熱係数 [W/m2/K]
    heat_transfer = hw1
    # 熱電対熱流束 [W]
    if mode != 2:
        heat_flux = (
            heat_transfer
            * tower_conds.thermocouple.heat_transfer_correction_factor
            * S_side
            * (tower.temp[stream - 1, section - 1] - tower.temp_thermo[stream - 1, section - 1])
        )
    else:
        heat_flux = (
            heat_transfer
            * 100
            * S_side
            * (tower.temp[stream - 1, section - 1] - tower.temp_thermo[stream - 1, section - 1])
        )
    # 熱電対上昇温度 [℃]
    temp_increase = heat_flux * tower_conds.common.calculation_step_time * 60 / heat_capacity
    # 次時刻熱電対温度 [℃]
    temp_thermocouple_reached = tower.temp_thermo[stream - 1, section - 1] + temp_increase

    cell_temperatures = CellTemperatures(
        bed_temperature=temp_reached,
        thermocouple_temperature=temp_thermocouple_reached,
    )
    heat_transfer_coefficients = HeatTransferCoefficients(wall_to_bed=hw1, bed_to_bed=u1)
    heat_flux = HeatFlux(
        adsorption=Habs, from_inner_boundary=Hwin, to_outer_boundary=Hwout, to_downstream=Hbb, from_upstream=Hroof
    )
    heat_balance_result = HeatBalanceResult(
        cell_temperatures=cell_temperatures,
        heat_transfer_coefficients=heat_transfer_coefficients,
        heat_flux=heat_flux,
    )

    output = heat_balance_result.to_dict()

    return output


def calculate_heat_balance_for_wall(
    tower_conds: TowerConditions,
    section,
    state_manager: StateVariables,
    tower_num,
    heat_output,
    heat_wall_output=None,
):
    """壁面の熱バランス計算

    Args:
        section (int): 対象のセクション番号
        variables (dict): 各セルの変数
        heat_output (dict): 隣接セルの熱バラ出力
        heat_wall_output (dict, optional): 上流セルの壁面熱バラ出力. Defaults to None.

    Returns:
        dict: 壁面熱バラ出力
    """
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions
    # セクション現在温度 [℃]
    temp_now = tower.temp_wall[section - 1]
    # 内側セクション温度 [℃]
    temp_inside_cell = tower.temp[tower_conds.common.num_streams - 1, section - 1]
    # 外側セクション温度 [℃]
    temp_outside_cell = tower_conds.vessel.ambient_temperature
    # 下流セクション温度 [℃]
    if section != tower_conds.common.num_sections:
        temp_below_cell = tower.temp_wall[section]
    # 上流壁への熱流束 [J]
    if section == 1:
        Hroof = (
            tower_conds.vessel.wall_thermal_conductivity
            * stream_conds[3].cross_section
            * (temp_now - tower.temp_lid_up)
            * tower_conds.common.calculation_step_time
            * 60
        )
    else:
        Hroof = heat_wall_output["Hbb"]
    # 内側境界からの熱流束 [J]
    Hwin = (
        heat_output["hw1"]
        * stream_conds[3].inner_boundary_area
        / tower_conds.common.num_sections
        * (temp_inside_cell - temp_now)
        * tower_conds.common.calculation_step_time
        * 60
    )
    # 外側境界への熱流束 [J]
    Hwout = (
        tower_conds.vessel.external_heat_transfer_coef
        * stream_conds[3].outer_boundary_area
        / tower_conds.common.num_sections
        * (temp_now - temp_outside_cell)
        * tower_conds.common.calculation_step_time
        * 60
    )
    # 下流壁への熱流束 [J]
    if section == tower_conds.common.num_sections:
        Hbb = (
            tower_conds.vessel.wall_thermal_conductivity
            * stream_conds[3].cross_section
            * (temp_now - tower.temp_lid_down)
            * tower_conds.common.calculation_step_time
            * 60
        )
    else:
        Hbb = (
            tower_conds.vessel.wall_thermal_conductivity
            * stream_conds[3].cross_section
            * (temp_now - tower.temp_wall[section])
        )
    # セクション到達温度 [℃]
    args = {
        "tower_conds": tower_conds,
        "temp_now": temp_now,
        "Hwin": Hwin,
        "Hwout": Hwout,
        "Hbb": Hbb,
        "Hroof": Hroof,
    }
    temp_reached = optimize.newton(_optimize_wall_temperature, temp_now, args=args.values())
    heat_flux = WallHeatFlux(from_inner_boundary=Hwin, to_outer_boundary=Hwout, to_downstream=Hbb, from_upstream=Hroof)
    wall_heat_balance_result = WallHeatBalanceResult(temperature=temp_reached, heat_flux=heat_flux)
    output = wall_heat_balance_result.to_dict()

    return output


def calculate_heat_balance_for_lid(
    tower_conds: TowerConditions,
    position: str,
    state_manager: StateVariables,
    tower_num: int,
    heat_output,
    heat_wall_output,
):
    """上下蓋の熱バランス計算

    Args:
        position (str): 上と下のどちらの蓋か
        variables (dict): 各セルの変数
        heat_output (dict): 各セルの熱バラ出力

    Returns:
        dict: 熱バラ出力
    """
    tower = state_manager.towers[tower_num]
    # セクション現在温度 [℃]
    temp_now = tower.temp_lid_up if position == "up" else tower.temp_lid_down
    # 外気への熱流束 [J]
    Hout = (
        tower_conds.vessel.external_heat_transfer_coef
        * (temp_now - tower_conds.vessel.ambient_temperature)
        * tower_conds.common.calculation_step_time
        * 60
    )
    if position == "up":
        Hout *= tower_conds.bottom.outer_flange_area
    elif position == "down":
        Hout *= tower_conds.lid.outer_flange_area
    # 底が受け取る熱(熱収支基準)
    if position == "up":
        Hlidall_heat = heat_output[2][1]["Hroof"] - heat_output[1][1]["Hroof"] - Hout - heat_wall_output[1]["Hroof"]
    else:
        Hlidall_heat = (
            heat_output[2][tower_conds.common.num_sections]["Hroof"]
            - heat_output[1][tower_conds.common.num_sections]["Hroof"]
            - Hout
            - heat_wall_output[tower_conds.common.num_sections]["Hbb"]
        )
    # セクション到達温度 [℃]
    args = {
        "tower_conds": tower_conds,
        "temp_now": temp_now,
        "Hlidall_heat": Hlidall_heat,
        "position": position,
    }
    temp_reached = optimize.newton(_optimize_lid_temperature, temp_now, args=args.values())
    lid_heat_balance_result = LidHeatBalanceResult(temperature=temp_reached)
    output = lid_heat_balance_result.to_dict()

    return output


def calculate_pressure_after_vacuum_pumping(
    tower_conds: TowerConditions, state_manager: StateVariables, tower_num: int
):
    """排気後圧力とCO2回収濃度の計算
        真空脱着モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 圧損, 積算CO2・N2回収量, CO2回収濃度, 排気後圧力,
    """
    tower = state_manager.towers[tower_num]
    ### 前準備 --------------------------------------
    # 容器内平均温度 [K]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + tower_conds.common.num_streams)
                for section in range(1, 1 + tower_conds.common.num_sections)
            ]
        )
        + 273.15
    )
    # 全圧 [PaA]
    P = tower.total_press * 1e6  # [MPaA]→[Pa]
    # 平均co2分率
    _mean_mf_co2 = np.mean(
        [
            tower.mf_co2[stream - 1, section - 1]
            for stream in range(1, 1 + tower_conds.common.num_streams)
            for section in range(1, 1 + tower_conds.common.num_sections)
        ]
    )
    # 平均n2分率
    _mean_mf_n2 = np.mean(
        [
            tower.mf_n2[stream - 1, section - 1]
            for stream in range(1, 1 + tower_conds.common.num_streams)
            for section in range(1, 1 + tower_conds.common.num_sections)
        ]
    )
    # 真空ポンプ排気ガス粘度 [Pa・s]
    # NOTE: 比熱と熱伝導率と粘度は大気圧を使用
    P_ATM = 0.101325 * 1e6
    mu = (
        CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * _mean_mf_co2
        + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * _mean_mf_n2
    )
    # 真空ポンプ排気ガス密度 [kg/m3]
    rho = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * _mean_mf_co2
        + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * _mean_mf_n2
    )

    ### 圧損計算 --------------------------------------
    _max_iteration = 1000
    P_resist = 0
    tolerance = 1e-6
    for iter in range(_max_iteration):
        P_resist_old = P_resist
        # ポンプ見せかけの全圧 [PaA]
        P_PUMP = (tower.total_press - P_resist) * 1e6
        P_PUMP = max(0, P_PUMP)
        # 真空ポンプ排気速度 [m3/min]
        vacuum_rate = 25 * (tower_conds.vacuum_piping.diameter**4) * P_PUMP / 2
        # 真空ポンプ排気ノルマル流量 [m3/min]
        vacuum_rate_N = vacuum_rate / 0.1013 * P_PUMP * 1e-6
        # 真空ポンプ排気線流速 [m/3]
        linear_velocity = vacuum_rate / tower_conds.vacuum_piping.cross_section
        # 真空ポンプ排気レイノルズ数
        Re = rho * linear_velocity * tower_conds.vacuum_piping.diameter / mu
        # 真空ポンプ排気管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        # 均圧配管圧力損失 [MPaA]
        P_resist = (
            lambda_f
            * tower_conds.vacuum_piping.length
            / tower_conds.vacuum_piping.diameter
            * linear_velocity**2
            / (2 * 9.81)
        ) * 1e-6
        # 収束判定
        if np.abs(P_resist - P_resist_old) < tolerance:
            break
        if pd.isna(P_resist):
            break
    if iter == _max_iteration - 1:
        print("収束せず: 見せかけの全圧 =", np.abs(P_resist - P_resist_old))

    ### CO2回収濃度計算 --------------------------------------
    # 排気速度 [mol/min]
    vacuum_rate_mol = 101325 * vacuum_rate_N / 8.314 / T_K
    # 排気量 [mol]
    vacuum_amt = vacuum_rate_mol * tower_conds.common.calculation_step_time
    # 排気CO2量 [mol]
    vacuum_amt_co2 = vacuum_amt * _mean_mf_co2
    # 排気N2量 [mol]
    vacuum_amt_n2 = vacuum_amt * _mean_mf_n2
    # 積算排気CO2量 [mol]
    accum_vacuum_amt_co2 = tower.vacuum_amt_co2 + vacuum_amt_co2
    # 積算排気N2量 [mol]
    accum_vacuum_amt_n2 = tower.vacuum_amt_n2 + vacuum_amt_n2
    # CO2回収濃度 [%]
    vacuum_co2_mf = (accum_vacuum_amt_co2 / (accum_vacuum_amt_co2 + accum_vacuum_amt_n2)) * 100

    ### 排気後圧力計算 --------------------------------
    # 排気"前"の真空排気空間の現在物質量 [mol]
    case_inner_mol_amt = (
        # P_PUMP * tower_conds.vacuum_piping["space_volume"]
        (P_PUMP + P_resist * 1e6)
        * tower_conds.vacuum_piping.space_volume
        / 8.314
        / T_K
    )
    # 排気"後"の現在物質量 [mol]
    case_inner_mol_amt_after_vacuum = max(0, case_inner_mol_amt - vacuum_amt)
    # 排気"後"の容器内部圧力 [MPaA]
    total_press_after_vacuum = (
        case_inner_mol_amt_after_vacuum * 8.314 * T_K / tower_conds.vacuum_piping.space_volume * 1e-6
    )
    vacuum_pumping_result = VacuumPumpingResult(
        pressure_loss=P_resist,
        total_co2_recovered=accum_vacuum_amt_co2,
        total_n2_recovered=accum_vacuum_amt_n2,
        co2_recovery_concentration=vacuum_co2_mf,
        volumetric_flow_rate=vacuum_rate_N,
        remaining_moles=case_inner_mol_amt_after_vacuum,
        final_pressure=total_press_after_vacuum,
    )
    output = vacuum_pumping_result.to_dict()

    return output


def calculate_pressure_after_depressurization(
    tower_conds: TowerConditions, state_manager: StateVariables, tower_num: int, downstream_tower_pressure
):
    """減圧時の上流からの均圧配管流量計算
        バッチ均圧(上流側)モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 排気後圧力と排気後CO2・N2モル量
    """
    tower = state_manager.towers[tower_num]
    ### 上流均圧配管流量の計算 ----------------------------------------
    # 容器内平均温度 [℃]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + tower_conds.common.num_streams)
                for section in range(1, 1 + tower_conds.common.num_sections)
            ]
        )
        + 273.15
    )
    # 全圧 [PaA]
    P = tower.total_press * 1e6  # [MPaA]→[Pa]
    # 平均co2分率
    _mean_mf_co2 = np.mean(
        [
            tower.mf_co2[stream - 1, section - 1]
            for stream in range(1, 1 + tower_conds.common.num_streams)
            for section in range(1, 1 + tower_conds.common.num_sections)
        ]
    )
    # 平均n2分率
    _mean_mf_n2 = np.mean(
        [
            tower.mf_n2[stream - 1, section - 1]
            for stream in range(1, 1 + tower_conds.common.num_streams)
            for section in range(1, 1 + tower_conds.common.num_sections)
        ]
    )
    # 上流均圧管ガス粘度 [Pa・s]
    # NOTE: 比熱と熱伝導率と粘度は大気圧を使用
    P_ATM = 0.101325 * 1e6
    mu = (
        CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * _mean_mf_co2
        + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * _mean_mf_n2
    )
    # 上流均圧管ガス密度 [kg/m3]
    rho = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * _mean_mf_co2
        + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * _mean_mf_n2
    )
    # 上流均圧配管圧力損失 [MPaA]
    _max_iteration = 1000
    P_resist = 0
    tolerance = 1e-6
    for iter in range(_max_iteration):
        P_resist_old = P_resist
        # 塔間の圧力差 [PaA]
        dP = (tower.total_press - downstream_tower_pressure - P_resist) * 1e6
        if np.abs(dP) < 1:
            dP = 0
        # 配管流速 [m/s]
        flow_rate = dP * tower_conds.equalizing_piping.diameter**2 / (32 * mu * tower_conds.equalizing_piping.length)
        flow_rate = max(1e-8, flow_rate)
        # 均圧管レイノルズ数
        Re = rho * abs(flow_rate) * tower_conds.equalizing_piping.diameter / mu
        # 管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        # 均圧配管圧力損失 [MPaA]
        P_resist = (
            lambda_f
            * tower_conds.equalizing_piping.length
            / tower_conds.equalizing_piping.diameter
            * flow_rate**2
            / (2 * 9.81)
        ) * 1e-6
        # 収束判定
        if np.abs(P_resist - P_resist_old) < tolerance:
            break
        if pd.isna(P_resist):
            break
    if iter == _max_iteration - 1:
        print("収束せず: 圧力差 =", np.abs(P_resist - P_resist_old))
    # 均圧配管流量 [m3/min]
    flow_amount_m3 = (
        tower_conds.equalizing_piping.cross_section
        * flow_rate
        / 60
        * tower_conds.equalizing_piping.flow_velocity_correction_factor
        * 5
    )
    # 均圧配管ノルマル流量 [m3/min]
    flow_amount_m3_N = flow_amount_m3 * tower.total_press / 0.1013
    # 均圧配管流量 [L/min] (下流塔への入力)
    flow_amount_l = flow_amount_m3_N * 1e3

    ### 次時刻の圧力計算 ----------------------------------------
    # 容器上流空間を移動する物質量 [mol]
    mw_upper_space = flow_amount_m3_N * 1000 * tower_conds.common.calculation_step_time / 22.4
    # 上流側の合計体積 [m3]
    V_upper_tower = tower_conds.packed_bed.vessel_internal_void_volume + tower_conds.packed_bed.void_volume
    # 上流容器圧力変化 [MPaA]
    dP_upper = 8.314 * T_K / V_upper_tower * mw_upper_space * 1e-6
    # 次時刻の容器圧力 [MPaA]
    total_press_after_depressure = tower.total_press - dP_upper
    depressurization_result = DepressurizationResult(
        final_pressure=total_press_after_depressure, flow_rate=flow_amount_l, pressure_differential=dP
    )
    output = depressurization_result.to_dict()

    return output


def calculate_downstream_flow_after_depressurization(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    mass_balance_results,
    downstream_tower_pressure,
):
    """減圧計算時に下流塔の圧力・流入量を計算する
        減圧モード

    Args:
        tower_conds (dict): 全体共通パラメータ
        stream_conds (dict): ストリーム内共通パラメータ
        variables (dict): 状態変数
        mass_balance_results (dict): マテバラ計算結果
        downstream_tower_pressure (float): 下流塔の現在全圧

    Returns:
        float: 減圧後の下流塔の全圧 [MPaA]
        float: 下流塔への流出CO2流量 [cm3]
        float: 下流塔への流出N2流量 [cm3]
    """
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions
    ### 下流塔の圧力計算 ----------------------------------------------
    # 容器内平均温度 [℃]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + tower_conds.common.num_streams)
                for section in range(1, 1 + tower_conds.common.num_sections)
            ]
        )
        + 273.15
    )
    # 下流流出量合計（最下流セクションの合計）[L]
    most_down_section = tower_conds.common.num_sections
    sum_outflow_fr = (
        sum(
            [
                mass_balance_results[stream][most_down_section]["outlet_co2_volume"]
                + mass_balance_results[stream][most_down_section]["outlet_n2_volume"]
                for stream in range(1, 1 + tower_conds.common.num_streams)
            ]
        )
        / 1e3
    )
    # 下流流出物質量 [mol]
    sum_outflow_mol = sum_outflow_fr / 22.4
    # 均圧下流側空間体積 [m3]
    V_downflow = (
        tower_conds.equalizing_piping.volume
        + tower_conds.packed_bed.void_volume
        + tower_conds.packed_bed.vessel_internal_void_volume
    )
    # 下流容器圧力変化 [MPaA]
    dP = 8.314 * T_K / V_downflow * sum_outflow_mol / 1e6
    # 次時刻の下流容器全圧 [MPaA]
    total_press_after_depressure_downflow = downstream_tower_pressure + dP

    ### 下流容器への流入量 --------------------------------------------
    # 下流塔への合計流出CO2流量 [cm3]
    sum_outflow_fr_co2 = sum(
        [
            mass_balance_results[stream][most_down_section]["outlet_co2_volume"]
            for stream in range(1, 1 + tower_conds.common.num_streams)
        ]
    )
    # 下流塔への合計流出N2流量 [cm3]
    sum_outflow_fr_n2 = sum(
        [
            mass_balance_results[stream][most_down_section]["outlet_n2_volume"]
            for stream in range(1, 1 + tower_conds.common.num_streams)
        ]
    )
    # 下流塔への流出CO2, N2流量 [cm3]
    outflow_fr = {}
    for stream in range(1, 1 + tower_conds.common.num_streams):
        outflow_fr[stream] = GasFlow(
            co2_volume=sum_outflow_fr_co2 * stream_conds[stream].area_fraction,
            n2_volume=sum_outflow_fr_n2 * stream_conds[stream].area_fraction,
            co2_mole_fraction=0,
            n2_mole_fraction=0,
        )
    downstream_flow_result = DownstreamFlowResult(
        final_pressure=total_press_after_depressure_downflow, outlet_flows=outflow_fr
    )
    output = downstream_flow_result.to_dict()

    return output


def calculate_pressure_after_desorption(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num,
    mole_fraction_results,
    vacuum_pumping_results,
):
    """気相放出後の全圧の計算

    Args:
        variables (dict): 状態変数
        mole_fraction_results (dict): マテバラの出力結果（モル分率）

    Returns:
        dict: 気相放出後の全圧
    """
    tower = state_manager.towers[tower_num]
    # 容器内平均温度 [℃]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + tower_conds.common.num_streams)
                for section in range(1, 1 + tower_conds.common.num_sections)
            ]
        )
        + 273.15
    )
    # 気相放出後の全物質量合計 [mol]
    sum_desorp_mw = (
        sum(
            [
                mole_fraction_results[stream][section]["desorp_mw_all_after_vacuum"]
                for stream in range(1, 1 + tower_conds.common.num_streams)
                for section in range(1, 1 + tower_conds.common.num_sections)
            ]
        )
        / 22.4
        / 1e3
    )
    # 配管上のモル量を加算
    sum_desorp_mw += (
        vacuum_pumping_results["total_press_after_vacuum"] * 1e6 * tower_conds.vacuum_piping.volume / 8.314 / T_K
    )
    # 気相放出後の全圧 [MPaA]
    pressure_after_desorption = sum_desorp_mw * 8.314 * T_K / tower_conds.vacuum_piping.space_volume * 1e-6

    return pressure_after_desorption


def calculate_pressure_after_batch_adsorption(
    tower_conds: TowerConditions, state_manager: StateVariables, tower_num: int, is_series_operation
):
    """バッチ吸着における圧力変化

    Args:
        tower_conds (dict): 実験条件
        variables (dict): 状態変数

    Return:
        float: バッチ吸着後の全圧
    """
    tower = state_manager.towers[tower_num]

    # 気体定数 [J/K/mol]
    R = 8.314
    # 平均温度 [K]
    temp_mean = []
    for stream in range(1, 1 + tower_conds.common.num_streams):
        for section in range(1, 1 + tower_conds.common.num_sections):
            temp_mean.append(tower.temp[stream - 1, section - 1])
    temp_mean = np.mean(temp_mean)
    temp_mean += 273.15
    # 空間体積（配管含む）
    if is_series_operation:
        V = (
            (tower_conds.packed_bed.void_volume + tower_conds.packed_bed.vessel_internal_void_volume) * 2
            + tower_conds.packed_bed.upstream_piping_volume
            + tower_conds.equalizing_piping.volume
        )
    else:
        V = (
            tower_conds.packed_bed.void_volume
            + tower_conds.packed_bed.vessel_internal_void_volume
            + tower_conds.packed_bed.upstream_piping_volume
        )
    # ノルマル体積流量
    F = tower_conds.feed_gas.total_flow_rate  # バッチ吸着: 導入ガスの流量
    # 圧力変化量
    diff_pressure = R * temp_mean / V * (F / 22.4) * tower_conds.common.calculation_step_time / 1e6
    # 変化後全圧
    pressure_after_batch_adsorption = tower.total_press + diff_pressure

    return pressure_after_batch_adsorption


def _calculate_equilibrium_adsorption_amount(P, T):
    """平衡吸着量を計算

    Args:
        P (float): 圧力 [kPaA]
        T (float): 温度 [K]
    Returns:
        平衡吸着量
    """
    # 吸着等温式（シンボリック回帰による近似式）
    equilibrium_loading = (
        P
        * (252.0724 - 0.50989705 * T)
        / (P - 3554.54819062669 * (1 - 0.0655247236249063 * np.sqrt(T)) ** 3 + 1.7354268)
    )
    return equilibrium_loading


def _optimize_bed_temperature(
    temp_reached,
    tower_conds: TowerConditions,
    gas_specific_heat,
    Mgas,
    temp_now,
    Habs,
    Hwin,
    Hwout,
    Hbb,
    Hroof,
    stream,
):
    """セクション到達温度算出におけるソルバー用関数

    Args:
        temp_reached (float): セクション到達温度
        args (dict): 充填層が受ける熱を計算するためのパラメータ

    Returns:
        float: 充填層が受ける熱の熱収支基準と時間基準の差分
    """
    stream_conds = tower_conds.stream_conditions

    # 流入ガスが受け取る熱 [J]
    Hgas = gas_specific_heat * Mgas * (temp_reached - temp_now)
    # 充填層が受け取る熱(ΔT基準) [J]
    Hbed_time = (
        tower_conds.packed_bed.heat_capacity
        * stream_conds[stream].area_fraction
        / tower_conds.common.num_sections
        * (temp_reached - temp_now)
    )
    # 充填層が受け取る熱(熱収支基準) [J]
    Hbed_heat_blc = Habs - Hgas + Hwin - Hwout - Hbb - Hroof
    return Hbed_heat_blc - Hbed_time


def _optimize_wall_temperature(
    temp_reached,
    tower_conds: TowerConditions,
    temp_now,
    Hwin,
    Hwout,
    Hbb,
    Hroof,
):
    stream_conds = tower_conds.stream_conditions
    """壁面の到達温度算出におけるソルバー用関数

    Args:
        temp_reached (float): セクション到達温度
        args (dict): 充填層が受ける熱を計算するためのパラメータ

    Returns:
        float: 充填層が受ける熱の熱収支基準と時間基準の差分
    """
    # 壁が受け取る熱(熱収支基準) [J]
    Hwall_heat_blc = Hwin - Hroof - Hwout - Hbb
    # 壁が受け取る熱(ΔT基準) [J]
    Hwall_time = (
        tower_conds.vessel.wall_specific_heat_capacity
        * stream_conds[1 + tower_conds.common.num_streams].wall_weight
        * (temp_reached - temp_now)
    )

    return Hwall_heat_blc - Hwall_time


def _optimize_lid_temperature(temp_reached, tower_conds: TowerConditions, temp_now, Hlidall_heat, position):
    """上下蓋の到達温度算出におけるソルバー用関数

    Args:
        temp_reached (float): セクション到達温度
        args (dict): 充填層が受ける熱を計算するためのパラメータ

    Returns:
        float: 充填層が受ける熱の熱収支基準と時間基準の差分
    """
    # 壁が受け取る熱(ΔT基準) [J]
    Hlid_time = tower_conds.vessel.wall_specific_heat_capacity * (temp_reached - temp_now)
    if position == "up":
        Hlid_time *= tower_conds.lid.flange_total_weight
    else:
        Hlid_time *= tower_conds.bottom.flange_total_weight

    return Hlidall_heat - Hlid_time
