"""圧力計算モジュール

純粋な圧力計算を行います。

主な計算内容:
- 真空脱着後の圧力（気相放出後）
- バッチ吸着後の圧力（密閉状態での圧力上昇）

主要な関数:
- calculate_pressure_after_vacuum_desorption(): 気相放出後の圧力
- calculate_pressure_after_batch_adsorption(): バッチ吸着後の圧力

ヘルパー関数（他モジュールからも使用）:
- _calculate_average_temperature(): 容器内平均温度
- _calculate_average_mole_fractions(): 平均モル分率
"""

import numpy as np

from common.constants import (
    CELSIUS_TO_KELVIN_OFFSET,
    STANDARD_MOLAR_VOLUME,
    GAS_CONSTANT,
    PA_TO_MPA,
    MPA_TO_PA,
    CM3_TO_L,
)

from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MoleFractionResults,
    VacuumPumpingResult,
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
    
    # 気相放出後の全物質量 [mol]
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
# ヘルパー関数（他モジュールからも使用）
# ============================================================

def _calculate_average_temperature(tower, tower_conds: TowerConditions) -> float:
    """容器内平均温度を計算 [℃]"""
    temps = [
        tower.cell(stream, section).temp
        for stream in range(tower_conds.common.num_streams)
        for section in range(tower_conds.common.num_sections)
    ]
    return np.mean(temps)


def _calculate_average_mole_fractions(tower, tower_conds: TowerConditions):
    """平均モル分率を計算"""
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
