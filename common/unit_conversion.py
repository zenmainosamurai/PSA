"""単位変換ユーティリティ

PSA担当者向け説明:
シミュレーション結果の単位変換を行う関数を提供します。
主にcm³からNm³（標準状態での体積）への変換に使用します。

使用例:
    from common.unit_conversion import convert_cm3_to_nm3
    
    # 現在の状態での体積をノルマル体積に変換
    nm3_volume = convert_cm3_to_nm3(
        volume_cm3=1000,      # 1000 cm³
        pressure_mpa=0.5,     # 0.5 MPa
        temperature_c=25      # 25℃
    )
"""

from .constants import STANDARD_PRESSURE


def convert_cm3_to_nm3(
    volume_cm3: float,
    pressure_mpa: float,
    temperature_c: float
) -> float:
    """
    体積をcm³からNm³（標準状態での体積）に変換
    
    理想気体の状態方程式に基づき、現在の圧力・温度での体積を
    標準状態（0℃, 101.325 kPa）での体積に換算します。
    
    計算式:
        V_n = V × (P / P_0) × (T_0 / T)
        
        V_n: 標準状態での体積
        V: 現在の体積
        P: 現在の圧力
        P_0: 標準圧力 (101325 Pa)
        T: 現在の温度 [K]
        T_0: 標準温度 (273.15 K)
    
    Args:
        volume_cm3: 体積 [cm³]
        pressure_mpa: 圧力 [MPa]
        temperature_c: 温度 [℃]
        
    Returns:
        標準状態での体積 [Nm³]
    """
    # 定数
    STANDARD_TEMPERATURE_K = 273.15  # 標準温度 [K]
    
    # 単位変換
    pressure_pa = pressure_mpa * 1e6  # MPa → Pa
    temperature_k = temperature_c + 273.15  # ℃ → K
    
    # 標準状態での体積を計算 [Ncm³]
    volume_ncm3 = (
        volume_cm3 
        * (pressure_pa / STANDARD_PRESSURE) 
        * (STANDARD_TEMPERATURE_K / temperature_k)
    )
    
    # cm³ → m³ に変換
    volume_nm3 = volume_ncm3 * 1e-6
    
    return volume_nm3


def convert_nm3_to_cm3(
    volume_nm3: float,
    pressure_mpa: float,
    temperature_c: float
) -> float:
    """
    体積をNm³（標準状態）からcm³（現在の状態）に変換
    
    Args:
        volume_nm3: 標準状態での体積 [Nm³]
        pressure_mpa: 現在の圧力 [MPa]
        temperature_c: 現在の温度 [℃]
        
    Returns:
        現在の状態での体積 [cm³]
    """
    STANDARD_TEMPERATURE_K = 273.15
    
    pressure_pa = pressure_mpa * 1e6
    temperature_k = temperature_c + 273.15
    
    # Nm³ → Ncm³
    volume_ncm3 = volume_nm3 * 1e6
    
    # 現在の状態での体積を計算
    volume_cm3 = (
        volume_ncm3 
        * (STANDARD_PRESSURE / pressure_pa) 
        * (temperature_k / STANDARD_TEMPERATURE_K)
    )
    
    return volume_cm3


def convert_l_per_min_to_m3_per_min(flow_rate_l_per_min: float) -> float:
    """
    流量をL/minからm³/minに変換
    
    Args:
        flow_rate_l_per_min: 流量 [L/min]
        
    Returns:
        流量 [m³/min]
    """
    return flow_rate_l_per_min * 1e-3


def convert_m3_per_min_to_l_per_min(flow_rate_m3_per_min: float) -> float:
    """
    流量をm³/minからL/minに変換
    
    Args:
        flow_rate_m3_per_min: 流量 [m³/min]
        
    Returns:
        流量 [L/min]
    """
    return flow_rate_m3_per_min * 1e3


def convert_celsius_to_kelvin(temperature_c: float) -> float:
    """
    温度を摂氏から絶対温度に変換
    
    Args:
        temperature_c: 温度 [℃]
        
    Returns:
        温度 [K]
    """
    return temperature_c + 273.15


def convert_kelvin_to_celsius(temperature_k: float) -> float:
    """
    温度を絶対温度から摂氏に変換
    
    Args:
        temperature_k: 温度 [K]
        
    Returns:
        温度 [℃]
    """
    return temperature_k - 273.15


def convert_mpa_to_pa(pressure_mpa: float) -> float:
    """
    圧力をMPaからPaに変換
    
    Args:
        pressure_mpa: 圧力 [MPa]
        
    Returns:
        圧力 [Pa]
    """
    return pressure_mpa * 1e6


def convert_pa_to_mpa(pressure_pa: float) -> float:
    """
    圧力をPaからMPaに変換
    
    Args:
        pressure_pa: 圧力 [Pa]
        
    Returns:
        圧力 [MPa]
    """
    return pressure_pa * 1e-6
