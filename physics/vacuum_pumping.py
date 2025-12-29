"""真空排気計算モジュール

真空脱着モードにおける真空ポンプによる排気計算を行います。

主な計算内容:
- 配管の圧力損失（反復法）
- 排気後の圧力
- CO2/N2回収量

主要な関数:
- calculate_vacuum_pumping(): 真空排気の総合計算
"""

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

from common.constants import (
    CELSIUS_TO_KELVIN_OFFSET,
    GAS_CONSTANT,
    STANDARD_PRESSURE,
    PA_TO_MPA,
    MPA_TO_PA,
    GRAVITY_ACCELERATION,
    CM3_TO_L,
    L_TO_M3,
)

from config.sim_conditions import TowerConditions
from state import StateVariables, VacuumPumpingResult

from physics.pressure import (
    _calculate_average_temperature,
    _calculate_average_mole_fractions,
)


def calculate_vacuum_pumping(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
) -> VacuumPumpingResult:
    """
    真空排気計算（真空脱着モード）
    
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


def _calculate_recovery_amounts(
    tower,
    tower_conds: TowerConditions,
    avg_co2_mf: float,
    avg_n2_mf: float,
):
    """CO2/N2回収量を計算"""
    # 吸着量の差分からCO2回収量を計算
    total_desorption_volume = 0.0  # [Ncm3]
    stream_conds = tower_conds.stream_conditions
    
    for stream in range(tower_conds.common.num_streams):
        for section in range(tower_conds.common.num_sections):
            current_loading = tower.cell(stream, section).loading
            previous_loading = tower.cell(stream, section).previous_loading
            section_adsorbent_mass = (
                stream_conds[stream].adsorbent_mass
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
