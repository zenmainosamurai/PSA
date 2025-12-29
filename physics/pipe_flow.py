"""配管流量・圧力損失計算モジュール

配管内のガス流れに関する物理計算を行います。

主な計算内容:
- 真空ポンプの排気速度と圧力損失
- 均圧配管の流量と圧力損失
- 理想気体の状態方程式に基づく圧力変化

主要な関数:
- calculate_vacuum_pump_flow(): 真空ポンプ流量・圧力損失
- calculate_equalization_flow(): 均圧配管流量・圧力損失
- calculate_pressure_change_from_moles(): 物質量変化からの圧力変化
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

from common.constants import (
    GAS_CONSTANT,
    STANDARD_PRESSURE,
    PA_TO_MPA,
    MPA_TO_PA,
    MINUTE_TO_SECOND,
    GRAVITY_ACCELERATION,
    M3_TO_L,
    STANDARD_MOLAR_VOLUME,
)

from config.sim_conditions import TowerConditions


@dataclass
class VacuumPumpFlowResult:
    """真空ポンプ流量計算結果"""
    pressure_loss: float          # 圧力損失 [MPa]
    volumetric_flow_rate: float   # 体積流量（ノルマル）[m³/min]
    molar_flow_rate: float        # モル流量 [mol/min]


@dataclass
class EqualizationFlowResult:
    """均圧配管流量計算結果"""
    pressure_loss: float          # 圧力損失 [MPa]
    volumetric_flow_rate: float   # 体積流量（ノルマル）[L/min]
    pressure_differential: float  # 塔間圧力差 [Pa]


def calculate_vacuum_pump_flow(
    tower_conds: TowerConditions,
    current_pressure: float,
    T_K: float,
    viscosity: float,
    density: float,
) -> VacuumPumpFlowResult:
    """
    真空ポンプの流量と圧力損失を計算
    
    真空ポンプの特性曲線と配管抵抗から、
    実際の排気速度と配管の圧力損失を反復法で計算します。
    
    Args:
        tower_conds: 塔条件
        current_pressure: 現在の塔内圧力 [MPaA]
        T_K: 温度 [K]
        viscosity: ガス粘度 [Pa·s]
        density: ガス密度 [kg/m³]
    
    Returns:
        VacuumPumpFlowResult: 流量・圧力損失計算結果
    """
    MAX_ITERATIONS = 1000
    TOLERANCE = 1e-6
    
    piping = tower_conds.vacuum_piping
    
    pressure_loss = 0.0
    vacuum_rate_N = 0.0
    
    for iteration in range(MAX_ITERATIONS):
        pressure_loss_old = pressure_loss
        
        # ポンプ吸入口での見かけ圧力 [PaA]
        P_PUMP = max(0, (current_pressure - pressure_loss) * MPA_TO_PA)
        
        # 真空ポンプ見かけの排気速度 [m³/min]
        # コンダクタンスと排気速度の複合式
        vacuum_rate = (
            (P_PUMP + piping.pump_correction_factor_2)
            / STANDARD_PRESSURE
            * piping.pump_correction_factor_1
            * piping.vacuum_pumping_speed
            * np.pi / 8
            * (piping.diameter ** 4)
            * P_PUMP / 2
            / piping.length
            / (
                piping.vacuum_pumping_speed * viscosity
                + np.pi / 8
                * (piping.diameter ** 4)
                * P_PUMP / 2
                / piping.length
            )
        )
        
        # ノルマル流量 [m³/min]
        vacuum_rate_N = vacuum_rate / (STANDARD_PRESSURE * PA_TO_MPA) * P_PUMP * PA_TO_MPA
        
        # 線流速 [m/s]
        linear_velocity = vacuum_rate / piping.cross_section
        
        # レイノルズ数
        Re = density * linear_velocity * piping.diameter / viscosity
        
        # 管摩擦係数（層流）
        lambda_f = 64 / Re if Re != 0 else 0
        
        # 圧力損失 [MPaA]（Darcy-Weisbachの式）
        pressure_loss = (
            lambda_f
            * piping.length
            / piping.diameter
            * linear_velocity ** 2
            / (2 * GRAVITY_ACCELERATION)
            * density
            * GRAVITY_ACCELERATION
        ) * 1e-6
        
        # 収束判定
        if np.abs(pressure_loss - pressure_loss_old) < TOLERANCE:
            break
        if pd.isna(pressure_loss):
            break
    
    # モル流量 [mol/min]
    molar_flow_rate = STANDARD_PRESSURE * vacuum_rate_N / GAS_CONSTANT / T_K
    
    return VacuumPumpFlowResult(
        pressure_loss=pressure_loss,
        volumetric_flow_rate=vacuum_rate_N,
        molar_flow_rate=molar_flow_rate,
    )


def calculate_equalization_flow(
    tower_conds: TowerConditions,
    upstream_pressure: float,
    downstream_pressure: float,
    viscosity: float,
    density: float,
) -> EqualizationFlowResult:
    """
    均圧配管の流量と圧力損失を計算
    
    2塔間の圧力差に基づき、均圧配管を通る流量と
    配管の圧力損失を反復法で計算します。
    
    Args:
        tower_conds: 塔条件
        upstream_pressure: 上流塔圧力 [MPaA]
        downstream_pressure: 下流塔圧力 [MPaA]
        viscosity: ガス粘度 [Pa·s]
        density: ガス密度 [kg/m³]
    
    Returns:
        EqualizationFlowResult: 流量・圧力損失計算結果
    """
    MAX_ITERATIONS = 1000
    TOLERANCE = 1e-6
    
    piping = tower_conds.equalizing_piping
    
    pressure_loss = 0.0
    flow_rate_l_min = 0.0
    dP = 0.0
    
    for iteration in range(MAX_ITERATIONS):
        pressure_loss_old = pressure_loss
        
        # 塔間の圧力差 [PaA]
        dP = (upstream_pressure - downstream_pressure - pressure_loss) * MPA_TO_PA
        if np.abs(dP) < 1:
            dP = 0
        
        # 配管流速 [m/s]（コンダクタンスベース）
        flow_rate = (
            piping.pipe_correction_factor
            * (upstream_pressure - downstream_pressure)
            / downstream_pressure
            * piping.diameter ** 2
            / 4
            * (upstream_pressure - downstream_pressure)
            / 2
            / (8 * viscosity * piping.length)
        )
        flow_rate = max(1e-8, flow_rate)
        
        # レイノルズ数
        Re = density * abs(flow_rate) * piping.diameter / viscosity
        
        # 管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        
        # 圧力損失 [MPaA]
        pressure_loss = (
            lambda_f
            * piping.length
            / piping.diameter
            * flow_rate ** 2
            / (2 * GRAVITY_ACCELERATION)
            * density
            * GRAVITY_ACCELERATION
        ) * 1e-6
        
        # 収束判定
        if np.abs(pressure_loss - pressure_loss_old) < TOLERANCE:
            break
        if pd.isna(pressure_loss):
            break
    
    # 均圧配管流量 [m³/min]
    volumetric_flow_rate = (
        piping.cross_section
        * flow_rate
        * MINUTE_TO_SECOND
        * piping.flow_velocity_correction_factor
    )
    
    # ノルマル流量 [m³/min]
    standard_flow_rate = volumetric_flow_rate * upstream_pressure / (STANDARD_PRESSURE * PA_TO_MPA)
    
    # [L/min]
    flow_rate_l_min = standard_flow_rate * M3_TO_L
    
    return EqualizationFlowResult(
        pressure_loss=pressure_loss,
        volumetric_flow_rate=flow_rate_l_min,
        pressure_differential=dP,
    )


def calculate_pressure_change_from_moles(
    moles_transferred: float,
    T_K: float,
    volume: float,
) -> float:
    """
    物質量変化から圧力変化を計算
    
    理想気体の状態方程式 PV = nRT に基づき、
    物質量の変化による圧力変化を計算します。
    
    Args:
        moles_transferred: 移動した物質量 [mol]（流入なら正、流出なら負）
        T_K: 温度 [K]
        volume: 系の体積 [m³]
    
    Returns:
        圧力変化 [MPaA]（増加なら正、減少なら負）
    """
    return GAS_CONSTANT * T_K / volume * moles_transferred * PA_TO_MPA


def calculate_pressure_from_moles(
    total_moles: float,
    T_K: float,
    volume: float,
) -> float:
    """
    物質量から圧力を計算
    
    理想気体の状態方程式 PV = nRT に基づき、
    物質量から圧力を計算します。
    
    Args:
        total_moles: 全物質量 [mol]
        T_K: 温度 [K]
        volume: 系の体積 [m³]
    
    Returns:
        圧力 [MPaA]
    """
    return total_moles * GAS_CONSTANT * T_K / volume * PA_TO_MPA


def calculate_moles_from_pressure(
    pressure: float,
    T_K: float,
    volume: float,
) -> float:
    """
    圧力から物質量を計算
    
    理想気体の状態方程式 PV = nRT に基づき、
    圧力から物質量を計算します。
    
    Args:
        pressure: 圧力 [MPaA]
        T_K: 温度 [K]
        volume: 系の体積 [m³]
    
    Returns:
        物質量 [mol]
    """
    return pressure * MPA_TO_PA * volume / (GAS_CONSTANT * T_K)
