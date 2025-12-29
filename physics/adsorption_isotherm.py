"""吸着平衡線モジュール

このモジュールは吸着平衡線（平衡吸着量）の計算を提供します。

吸着平衡線とは:
与えられた温度・圧力条件下で、吸着材がどれだけのCO2を吸着できるかを
表す関係式です。この平衡吸着量と現在の吸着量の差が、
吸着・脱着の駆動力となります。

現在使用している吸着平衡式:
シンボリック回帰により実験データから導出した近似式を使用しています。
"""

import numpy as np


def calculate_equilibrium_loading(pressure_kpa: float, temperature_k: float) -> float:
    """
    平衡吸着量を計算
    
    与えられたCO2分圧と温度における平衡吸着量を計算します。
    
    物理的意味:
    - 圧力が高いほど、平衡吸着量は大きくなる（より多く吸着できる）
    - 温度が高いほど、平衡吸着量は小さくなる（吸着が弱まる）
    
    吸着平衡式（シンボリック回帰で導出）:
        q = P * (a1 - a2*T) / (P - b1*(1 - b2*sqrt(T))^3 + b3)
        係数: a1=252.0724, a2=0.50989705, b1=3554.548, b2=0.06552, b3=1.7354
        有効範囲: P=0.1-100 kPaA, T=273-373 K
    
    Args:
        pressure_kpa: CO2分圧 [kPaA]
        temperature_k: 温度 [K]
    
    Returns:
        平衡吸着量 [cm3/g-abs]（標準状態換算）
    """
    P = pressure_kpa
    T = temperature_k
    
    # シンボリック回帰による近似式（実験データから導出）
    equilibrium_loading = (
        P
        * (252.0724 - 0.50989705 * T)
        / (P - 3554.54819062669 * (1 - 0.0655247236249063 * np.sqrt(T)) ** 3 + 1.7354268)
    )
    
    return equilibrium_loading


def calculate_loading_at_conditions(
    co2_partial_pressure_mpa: float,
    temperature_celsius: float,
) -> float:
    """
    指定条件での平衡吸着量を計算（単位変換込み）
    
    より直感的な単位（MPa、℃）で平衡吸着量を計算できます。
    
    Args:
        co2_partial_pressure_mpa: CO2分圧 [MPaA]
        temperature_celsius: 温度 [℃]
    
    Returns:
        平衡吸着量 [cm3/g-abs]
    """
    # 単位変換
    pressure_kpa = co2_partial_pressure_mpa * 1000  # MPa -> kPa
    temperature_k = temperature_celsius + 273.15   # ℃ -> K
    
    return calculate_equilibrium_loading(pressure_kpa, temperature_k)


def calculate_driving_force(
    equilibrium_loading: float,
    current_loading: float,
) -> tuple[float, str]:
    """
    吸着・脱着の駆動力を計算
    
    平衡吸着量と現在吸着量の差から、吸着・脱着の駆動力を計算します。
    
    Args:
        equilibrium_loading: 平衡吸着量 [cm3/g-abs]
        current_loading: 現在の吸着量 [cm3/g-abs]
    
    Returns:
        (駆動力, モード)
        - 駆動力: 平衡吸着量と現在吸着量の差 [cm3/g-abs]
        - モード: "adsorption"（吸着）または "desorption"（脱着）
    """
    driving_force = equilibrium_loading - current_loading
    
    if driving_force >= 0:
        mode = "adsorption"
    else:
        mode = "desorption"
    
    return driving_force, mode
