"""均圧計算モジュール

均圧モードにおける塔間のガス移動計算を行います。

主な計算内容:
- 均圧減圧時の流量と圧力変化
- 下流塔への流入量と圧力変化
- 配管の圧力損失（反復法）

主要な関数:
- calculate_depressurization(): 均圧減圧計算
- calculate_downstream_flow(): 下流塔への流入計算
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
)

from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MassBalanceResults,
    DepressurizationResult,
    DownstreamFlowResult,
    GasFlow,
)

from physics.pressure import (
    _calculate_average_temperature,
    _calculate_average_mole_fractions,
)


def calculate_depressurization(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    downstream_tower_pressure: float,
) -> DepressurizationResult:
    """
    均圧減圧計算
    
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


def calculate_downstream_flow(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    mass_balance_results: MassBalanceResults,
    downstream_tower_pressure: float,
) -> DownstreamFlowResult:
    """
    減圧時の下流塔への流入計算
    
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
        outlet_flows[stream] = GasFlow(
            co2_volume=sum_outflow_co2 * stream_conds[stream].area_fraction,
            n2_volume=sum_outflow_n2 * stream_conds[stream].area_fraction,
            co2_mole_fraction=0,
            n2_mole_fraction=0,
        )

    return DownstreamFlowResult(
        final_pressure=final_pressure,
        outlet_flows=outlet_flows,
    )


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
