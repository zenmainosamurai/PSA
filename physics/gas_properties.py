"""ガス物性計算モジュール

混合ガスの物性（粘度・密度など）を計算します。

主な計算内容:
- CO2/N2混合ガスの粘度
- CO2/N2混合ガスの密度

主要な関数:
- calculate_mixed_gas_viscosity(): 混合ガス粘度
- calculate_mixed_gas_density(): 混合ガス密度
"""

import CoolProp.CoolProp as CP

from common.constants import STANDARD_PRESSURE


def calculate_mixed_gas_viscosity(
    T_K: float,
    co2_mole_fraction: float,
    n2_mole_fraction: float,
) -> float:
    """
    混合ガスの粘度を計算
    
    CO2とN2の混合ガスの粘度を、モル分率に基づく
    線形混合則で計算します。
    
    Args:
        T_K: 温度 [K]
        co2_mole_fraction: CO2モル分率 [-]
        n2_mole_fraction: N2モル分率 [-]
    
    Returns:
        混合ガスの粘度 [Pa·s]
    
    Note:
        粘度は大気圧での値を使用（圧力依存性は小さいため）
    """
    P_ATM = STANDARD_PRESSURE
    
    viscosity_co2 = CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2")
    viscosity_n2 = CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen")
    
    return viscosity_co2 * co2_mole_fraction + viscosity_n2 * n2_mole_fraction


def calculate_mixed_gas_density(
    T_K: float,
    P_Pa: float,
    co2_mole_fraction: float,
    n2_mole_fraction: float,
) -> float:
    """
    混合ガスの密度を計算
    
    CO2とN2の混合ガスの密度を、モル分率に基づく
    線形混合則で計算します。
    
    Args:
        T_K: 温度 [K]
        P_Pa: 圧力 [Pa]
        co2_mole_fraction: CO2モル分率 [-]
        n2_mole_fraction: N2モル分率 [-]
    
    Returns:
        混合ガスの密度 [kg/m³]
    """
    density_co2 = CP.PropsSI("D", "T", T_K, "P", P_Pa, "co2")
    density_n2 = CP.PropsSI("D", "T", T_K, "P", P_Pa, "nitrogen")
    
    return density_co2 * co2_mole_fraction + density_n2 * n2_mole_fraction
