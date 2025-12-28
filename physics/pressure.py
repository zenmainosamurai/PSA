"""圧力計算モジュール

PSA担当者向け説明:
このモジュールはPSAプロセスにおける圧力変化を計算します。

- 真空排気計算: 真空ポンプによる減圧とCO2回収量
- 均圧計算: 塔間の均圧配管を通じた圧力移動
- バッチ吸着後圧力: バッチ吸着における圧力上昇

主要な関数:
- calculate_vacuum_pumping(): 真空脱着時の排気計算
- calculate_depressurization(): 均圧減圧時の流量と圧力
- calculate_downstream_flow(): 減圧時の下流塔への流入
- calculate_pressure_after_vacuum_desorption(): 気相放出後の圧力
- calculate_pressure_after_batch_adsorption(): バッチ吸着後の圧力
"""

from typing import Dict
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

from common.constants import (
    CELSIUS_TO_KELVIN_OFFSET,
    STANDARD_MOLAR_VOLUME,
    GAS_CONSTANT,
    STANDARD_PRESSURE,
    PA_TO_MPA,
    MPA_TO_PA,
    MINUTE_TO_SECOND,
    GRAVITY_ACCELERATION,
    M3_TO_L,
    CM3_TO_L,
    L_TO_M3,
)

# 旧コードとの互換性のためインポート
from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MassBalanceResults,
    MoleFractionResults,
    VacuumPumpingResult,
    DepressurizationResult,
    DownstreamFlowResult,
    GasFlow,
)


# ============================================================
# 真空排気計算
# ============================================================

def calculate_vacuum_pumping(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
) -> VacuumPumpingResult:
    """
    真空排気計算（真空脱着モード）
    
    PSA担当者向け説明:
    真空ポンプで塔内を減圧する際の以下を計算します:
    - 圧力損失（配管抵抗）
    - CO2/N2回収量
    - 排気後の圧力
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
    
    Returns:
        VacuumPumpingResult: 真空排気計算結果
    """
    tower = state_manager.towers[tower_num]
    
    # === 前準備 ===
    # 容器内平均温度 [K]
    T_K = _calculate_average_temperature(tower, tower_conds) + CELSIUS_TO_KELVIN_OFFSET
    
    # 全圧 [PaA]
    P = tower.total_press * MPA_TO_PA
    
    # 平均モル分率
    avg_co2_mf, avg_n2_mf = _calculate_average_mole_fractions(tower, tower_conds)
    
    # ガス粘度・密度（CoolProp使用）
    P_ATM = STANDARD_PRESSURE
    viscosity = (
        CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * avg_co2_mf
        + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * avg_n2_mf
    )
    rho = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * avg_co2_mf
        + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * avg_n2_mf
    )

    # === 圧力損失計算（反復法）===
    pressure_loss, vacuum_rate_N = _calculate_pressure_loss_vacuum(
        tower_conds, tower, T_K, viscosity, rho
    )

    # === CO2回収量計算 ===
    cumulative_co2_recovered, cumulative_n2_recovered = _calculate_recovery_amounts(
        tower, tower_conds, avg_co2_mf, avg_n2_mf
    )
    
    # CO2回収濃度 [%]
    total_recovered = cumulative_co2_recovered + cumulative_n2_recovered
    co2_recovery_concentration = (
        (cumulative_co2_recovered / total_recovered) * 100
        if total_recovered > 0 else 0
    )

    # === 排気後圧力計算 ===
    # 排気速度 [mol/min]
    vacuum_rate_mol = STANDARD_PRESSURE * vacuum_rate_N / GAS_CONSTANT / T_K
    
    # 排気量 [mol]
    moles_pumped = vacuum_rate_mol * tower_conds.common.calculation_step_time
    
    # 排気前の物質量 [mol]
    P_PUMP = max(0, (tower.total_press - pressure_loss) * MPA_TO_PA)
    case_inner_mol_amt = (
        (P_PUMP + pressure_loss * MPA_TO_PA)
        * tower_conds.vacuum_piping.space_volume
        / (GAS_CONSTANT * T_K)
    )
    
    # 排気後の物質量 [mol]
    remaining_moles = max(0, case_inner_mol_amt - moles_pumped)
    
    # 排気後の圧力 [MPaA]
    final_pressure = (
        remaining_moles * GAS_CONSTANT * T_K
        / tower_conds.vacuum_piping.space_volume
        * PA_TO_MPA
    )

    return VacuumPumpingResult(
        pressure_loss=pressure_loss,
        cumulative_co2_recovered=cumulative_co2_recovered,
        cumulative_n2_recovered=cumulative_n2_recovered,
        co2_recovery_concentration=co2_recovery_concentration,
        volumetric_flow_rate=vacuum_rate_N,
        remaining_moles=remaining_moles,
        final_pressure=final_pressure,
    )


# ============================================================
# 均圧減圧計算
# ============================================================

def calculate_depressurization(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    downstream_tower_pressure: float,
) -> DepressurizationResult:
    """
    均圧減圧計算
    
    PSA担当者向け説明:
    均圧配管を通じて高圧側から低圧側へガスが移動する際の
    流量と圧力変化を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号（上流側）
        downstream_tower_pressure: 下流塔の現在圧力 [MPaA]
    
    Returns:
        DepressurizationResult: 減圧計算結果
    """
    tower = state_manager.towers[tower_num]
    
    # === 前準備 ===
    T_K = _calculate_average_temperature(tower, tower_conds) + CELSIUS_TO_KELVIN_OFFSET
    P = tower.total_press * MPA_TO_PA
    avg_co2_mf, avg_n2_mf = _calculate_average_mole_fractions(tower, tower_conds)
    
    # ガス物性
    P_ATM = STANDARD_PRESSURE
    viscosity = (
        CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * avg_co2_mf
        + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * avg_n2_mf
    )
    rho = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * avg_co2_mf
        + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * avg_n2_mf
    )

    # === 圧力損失・流量計算（反復法）===
    pressure_loss, flow_rate, dP = _calculate_pressure_loss_equalization(
        tower_conds, tower, downstream_tower_pressure, viscosity, rho
    )
    
    # 均圧配管流量 [L/min]
    flow_rate_l_min = flow_rate

    # === 次時刻の圧力計算 ===
    # 移動物質量 [mol]
    standard_flow_rate = flow_rate_l_min / M3_TO_L  # [m3/min]
    mw_upper_space = (
        standard_flow_rate * M3_TO_L
        * tower_conds.common.calculation_step_time
        / STANDARD_MOLAR_VOLUME
    )
    
    # 上流側の合計体積 [m3]
    V_upper = (
        tower_conds.packed_bed.vessel_internal_void_volume
        + tower_conds.packed_bed.void_volume
    )
    
    # 圧力変化 [MPaA]
    dP_upper = GAS_CONSTANT * T_K / V_upper * mw_upper_space * PA_TO_MPA
    
    # 次時刻の圧力 [MPaA]
    final_pressure = tower.total_press - dP_upper

    return DepressurizationResult(
        final_pressure=final_pressure,
        flow_rate=flow_rate_l_min,
        pressure_differential=dP,
    )


# ============================================================
# 下流塔への流入計算
# ============================================================

def calculate_downstream_flow(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    mass_balance_results: MassBalanceResults,
    downstream_tower_pressure: float,
) -> DownstreamFlowResult:
    """
    減圧時の下流塔への流入計算
    
    PSA担当者向け説明:
    上流塔から流出したガスが下流塔に流入する際の
    圧力変化と各ストリームへの流入量を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        mass_balance_results: 物質収支計算結果
        downstream_tower_pressure: 下流塔の現在圧力 [MPaA]
    
    Returns:
        DownstreamFlowResult: 下流塔流入結果
    """
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions
    
    # 平均温度 [K]
    T_K = _calculate_average_temperature(tower, tower_conds) + CELSIUS_TO_KELVIN_OFFSET
    
    # 最下流セクションからの流出量合計 [L]（0オリジン）
    last_section = tower_conds.common.num_sections - 1
    sum_outflow = sum(
        mass_balance_results.get_result(stream, last_section).outlet_gas.co2_volume
        + mass_balance_results.get_result(stream, last_section).outlet_gas.n2_volume
        for stream in range(tower_conds.common.num_streams)
    ) / 1e3  # cm3 -> L
    
    # 流出物質量 [mol]
    sum_outflow_mol = sum_outflow / STANDARD_MOLAR_VOLUME
    
    # 下流側空間体積 [m3]
    V_downflow = (
        tower_conds.equalizing_piping.volume
        + tower_conds.packed_bed.void_volume
        + tower_conds.packed_bed.vessel_internal_void_volume
    )
    
    # 圧力変化 [MPaA]
    dP = GAS_CONSTANT * T_K / V_downflow * sum_outflow_mol * PA_TO_MPA
    
    # 次時刻の下流塔圧力 [MPaA]
    final_pressure = downstream_tower_pressure + dP

    # === 各ストリームへの流入量（0オリジン） ===
    sum_outflow_co2 = sum(
        mass_balance_results.get_result(stream, last_section).outlet_gas.co2_volume
        for stream in range(tower_conds.common.num_streams)
    )
    sum_outflow_n2 = sum(
        mass_balance_results.get_result(stream, last_section).outlet_gas.n2_volume
        for stream in range(tower_conds.common.num_streams)
    )
    
    outlet_flows: Dict[int, GasFlow] = {}
    for stream in range(tower_conds.common.num_streams):
        # stream_condsは1オリジンなので+1してアクセス
        stream_1indexed = stream + 1
        outlet_flows[stream] = GasFlow(
            co2_volume=sum_outflow_co2 * stream_conds[stream_1indexed].area_fraction,
            n2_volume=sum_outflow_n2 * stream_conds[stream_1indexed].area_fraction,
            co2_mole_fraction=0,
            n2_mole_fraction=0,
        )

    return DownstreamFlowResult(
        final_pressure=final_pressure,
        outlet_flows=outlet_flows,
    )


# ============================================================
# 真空脱着後の圧力計算
# ============================================================

def calculate_pressure_after_vacuum_desorption(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    mole_fraction_results: MoleFractionResults,
    vacuum_pumping_results: VacuumPumpingResult,
) -> float:
    """
    気相放出後の全圧を計算
    
    PSA担当者向け説明:
    吸着材からCO2が脱着して気相に放出された後の
    塔内圧力を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        mole_fraction_results: モル分率計算結果
        vacuum_pumping_results: 真空排気計算結果
    
    Returns:
        気相放出後の全圧 [MPaA]
    """
    tower = state_manager.towers[tower_num]
    
    # 平均温度 [K]
    T_K = _calculate_average_temperature(tower, tower_conds) + CELSIUS_TO_KELVIN_OFFSET
    
    # 気相放出後の全物質量 [mol]（0オリジン）
    sum_desorp_mw = sum(
        mole_fraction_results.get_result(stream, section).total_moles_after_desorption
        for stream in range(tower_conds.common.num_streams)
        for section in range(tower_conds.common.num_sections)
    ) * CM3_TO_L / STANDARD_MOLAR_VOLUME
    
    # 配管上のモル量を加算
    sum_desorp_mw += (
        vacuum_pumping_results.final_pressure * MPA_TO_PA
        * tower_conds.vacuum_piping.volume
        / GAS_CONSTANT / T_K
    )
    
    # 気相放出後の全圧 [MPaA]
    pressure = (
        sum_desorp_mw * GAS_CONSTANT * T_K
        / tower_conds.vacuum_piping.space_volume
        * PA_TO_MPA
    )
    
    return pressure


# ============================================================
# バッチ吸着後の圧力計算
# ============================================================

def calculate_pressure_after_batch_adsorption(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    is_series_operation: bool,
    has_pressure_valve: bool,
) -> float:
    """
    バッチ吸着後の圧力を計算
    
    PSA担当者向け説明:
    バッチ吸着（密閉状態でのガス導入）における
    圧力上昇を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        is_series_operation: 直列運転かどうか
        has_pressure_valve: 圧調弁があるかどうか
    
    Returns:
        バッチ吸着後の全圧 [MPaA]
    """
    tower = state_manager.towers[tower_num]
    
    # 平均温度 [K]
    temp_mean = _calculate_average_temperature(tower, tower_conds)
    temp_mean += CELSIUS_TO_KELVIN_OFFSET
    
    # 空間体積の決定（運転条件による）
    if is_series_operation and not has_pressure_valve:
        V = (
            (tower_conds.packed_bed.void_volume + tower_conds.packed_bed.vessel_internal_void_volume) * 2
            + tower_conds.packed_bed.upstream_piping_volume
            + tower_conds.equalizing_piping.volume
        )
    elif is_series_operation and has_pressure_valve:
        V = (
            tower_conds.packed_bed.void_volume
            + tower_conds.packed_bed.vessel_internal_void_volume
            + tower_conds.equalizing_piping.volume
            + tower_conds.equalizing_piping.isolated_equalizing_volume
        )
    else:
        V = (
            tower_conds.packed_bed.void_volume
            + tower_conds.packed_bed.vessel_internal_void_volume
            + tower_conds.packed_bed.upstream_piping_volume
        )
    
    # ノルマル体積流量 [L/min]
    F = tower_conds.feed_gas.total_flow_rate
    
    # 圧力変化量 [MPaA]
    diff_pressure = (
        GAS_CONSTANT * temp_mean / V
        * (F / STANDARD_MOLAR_VOLUME)
        * tower_conds.common.calculation_step_time
        * PA_TO_MPA
    )
    
    # 変化後全圧 [MPaA]
    return tower.total_press + diff_pressure


# ============================================================
# ヘルパー関数
# ============================================================

def _calculate_average_temperature(tower, tower_conds: TowerConditions) -> float:
    """容器内平均温度を計算 [℃]（内部インデックスは0オリジン）"""
    temps = [
        tower.cell(stream, section).temp
        for stream in range(tower_conds.common.num_streams)
        for section in range(tower_conds.common.num_sections)
    ]
    return np.mean(temps)


def _calculate_average_mole_fractions(tower, tower_conds: TowerConditions):
    """平均モル分率を計算（内部インデックスは0オリジン）"""
    co2_fractions = [
        tower.cell(stream, section).co2_mole_fraction
        for stream in range(tower_conds.common.num_streams)
        for section in range(tower_conds.common.num_sections)
    ]
    n2_fractions = [
        tower.cell(stream, section).n2_mole_fraction
        for stream in range(tower_conds.common.num_streams)
        for section in range(tower_conds.common.num_sections)
    ]
    return np.mean(co2_fractions), np.mean(n2_fractions)


def _calculate_pressure_loss_vacuum(
    tower_conds: TowerConditions,
    tower,
    T_K: float,
    viscosity: float,
    rho: float,
):
    """真空排気時の圧力損失を計算（反復法）"""
    MAX_ITERATIONS = 1000
    TOLERANCE = 1e-6
    
    pressure_loss = 0.0
    vacuum_rate_N = 0.0
    
    for iteration in range(MAX_ITERATIONS):
        pressure_loss_old = pressure_loss
        
        # ポンプ見せかけの全圧 [PaA]
        P_PUMP = max(0, (tower.total_press - pressure_loss) * MPA_TO_PA)
        
        # 真空ポンプ見かけの排気速度 [m3/min]
        # コンダクタンスと排気速度の複合式
        vacuum_rate = (
            (P_PUMP + tower_conds.vacuum_piping.pump_correction_factor_2)
            / STANDARD_PRESSURE
            * tower_conds.vacuum_piping.pump_correction_factor_1
            * tower_conds.vacuum_piping.vacuum_pumping_speed
            * np.pi / 8
            * (tower_conds.vacuum_piping.diameter ** 4)
            * P_PUMP / 2
            / tower_conds.vacuum_piping.length
            / (
                tower_conds.vacuum_piping.vacuum_pumping_speed * viscosity
                + np.pi / 8
                * (tower_conds.vacuum_piping.diameter ** 4)
                * P_PUMP / 2
                / tower_conds.vacuum_piping.length
            )
        )
        
        # ノルマル流量 [m3/min]
        vacuum_rate_N = vacuum_rate / (STANDARD_PRESSURE * PA_TO_MPA) * P_PUMP * PA_TO_MPA
        
        # 線流速 [m/s]
        linear_velocity = vacuum_rate / tower_conds.vacuum_piping.cross_section
        
        # レイノルズ数
        Re = rho * linear_velocity * tower_conds.vacuum_piping.diameter / viscosity
        
        # 管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        
        # 圧力損失 [MPaA]
        pressure_loss = (
            lambda_f
            * tower_conds.vacuum_piping.length
            / tower_conds.vacuum_piping.diameter
            * linear_velocity ** 2
            / (2 * GRAVITY_ACCELERATION)
            * rho
            * GRAVITY_ACCELERATION
        ) * 1e-6
        
        # 収束判定
        if np.abs(pressure_loss - pressure_loss_old) < TOLERANCE:
            break
        if pd.isna(pressure_loss):
            break
    
    return pressure_loss, vacuum_rate_N


def _calculate_pressure_loss_equalization(
    tower_conds: TowerConditions,
    tower,
    downstream_tower_pressure: float,
    viscosity: float,
    rho: float,
):
    """均圧時の圧力損失を計算（反復法）"""
    MAX_ITERATIONS = 1000
    TOLERANCE = 1e-6
    
    pressure_loss = 0.0
    flow_rate_l_min = 0.0
    dP = 0.0
    
    for iteration in range(MAX_ITERATIONS):
        pressure_loss_old = pressure_loss
        
        # 塔間の圧力差 [PaA]
        dP = (tower.total_press - downstream_tower_pressure - pressure_loss) * MPA_TO_PA
        if np.abs(dP) < 1:
            dP = 0
        
        # 配管流速 [m/s]（コンダクタンスベース）
        flow_rate = (
            tower_conds.equalizing_piping.pipe_correction_factor
            * (tower.total_press - downstream_tower_pressure)
            / downstream_tower_pressure
            * tower_conds.equalizing_piping.diameter ** 2
            / 4
            * (tower.total_press - downstream_tower_pressure)
            / 2
            / (8 * viscosity * tower_conds.equalizing_piping.length)
        )
        flow_rate = max(1e-8, flow_rate)
        
        # レイノルズ数
        Re = rho * abs(flow_rate) * tower_conds.equalizing_piping.diameter / viscosity
        
        # 管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        
        # 圧力損失 [MPaA]
        pressure_loss = (
            lambda_f
            * tower_conds.equalizing_piping.length
            / tower_conds.equalizing_piping.diameter
            * flow_rate ** 2
            / (2 * GRAVITY_ACCELERATION)
            * rho
            * GRAVITY_ACCELERATION
        ) * 1e-6
        
        # 収束判定
        if np.abs(pressure_loss - pressure_loss_old) < TOLERANCE:
            break
        if pd.isna(pressure_loss):
            break
    
    # 均圧配管流量 [m3/min]
    volumetric_flow_rate = (
        tower_conds.equalizing_piping.cross_section
        * flow_rate
        * MINUTE_TO_SECOND
        * tower_conds.equalizing_piping.flow_velocity_correction_factor
    )
    
    # ノルマル流量 [m3/min]
    standard_flow_rate = volumetric_flow_rate * tower.total_press / (STANDARD_PRESSURE * PA_TO_MPA)
    
    # [L/min]
    flow_rate_l_min = standard_flow_rate * M3_TO_L
    
    return pressure_loss, flow_rate_l_min, dP


def _calculate_recovery_amounts(
    tower,
    tower_conds: TowerConditions,
    avg_co2_mf: float,
    avg_n2_mf: float,
):
    """CO2/N2回収量を計算（内部インデックスは0オリジン）"""
    # 吸着量の差分からCO2回収量を計算
    total_desorption_volume = 0.0  # [Ncm3]
    stream_conds = tower_conds.stream_conditions
    
    for stream in range(tower_conds.common.num_streams):
        for section in range(tower_conds.common.num_sections):
            current_loading = tower.cell(stream, section).loading
            previous_loading = tower.cell(stream, section).previous_loading
            # stream_condsは1オリジンなので+1してアクセス
            section_adsorbent_mass = (
                stream_conds[stream + 1].adsorbent_mass
                / tower_conds.common.num_sections
            )
            
            # 脱着量（差分）[Ncm3/g-abs]
            loading_delta = previous_loading - current_loading
            
            # セクション全体での脱着量 [Ncm3]
            section_desorption = loading_delta * section_adsorbent_mass
            total_desorption_volume += section_desorption
    
    # CO2回収量 [Nm3]
    cumulative_co2 = total_desorption_volume * CM3_TO_L * L_TO_M3
    
    # N2回収量を平均モル分率から計算 [Nm3]
    if avg_co2_mf > 0:
        cumulative_n2 = cumulative_co2 * avg_n2_mf / avg_co2_mf
    else:
        cumulative_n2 = 0.0
    
    # 累積回収量に加算
    cumulative_co2 = tower.cumulative_co2_recovered + cumulative_co2
    cumulative_n2 = tower.cumulative_n2_recovered + cumulative_n2
    
    return cumulative_co2, cumulative_n2
