"""回収量計算モジュール

吸着・脱着に伴う物質の回収量を計算します。

主な計算内容:
- 脱着によるCO2回収量
- N2回収量（気相組成から）

主要な関数:
- calculate_desorption_amount(): 脱着による回収量計算
"""

from typing import Tuple

from common.constants import CM3_TO_L, L_TO_M3

from config.sim_conditions import TowerConditions


def calculate_desorption_amount(
    tower_conds: TowerConditions,
    tower,
    avg_co2_mole_fraction: float,
    avg_n2_mole_fraction: float,
) -> Tuple[float, float]:
    """
    脱着による回収量を計算
    
    吸着量の差分（前ステップとの差）から脱着量を計算し、
    累積回収量を返します。
    
    Args:
        tower_conds: 塔条件
        tower: タワー状態（TowerStateArrays）
        avg_co2_mole_fraction: 平均CO2モル分率 [-]
        avg_n2_mole_fraction: 平均N2モル分率 [-]
    
    Returns:
        (cumulative_co2_recovered, cumulative_n2_recovered): 
        累積CO2回収量 [Nm³], 累積N2回収量 [Nm³]
    """
    stream_conds = tower_conds.stream_conditions
    
    # 全セルの脱着量を合計
    total_desorption_volume = 0.0  # [Ncm³]
    
    for stream in range(tower_conds.common.num_streams):
        for section in range(tower_conds.common.num_sections):
            current_loading = tower.cell(stream, section).loading
            previous_loading = tower.cell(stream, section).previous_loading
            section_adsorbent_mass = (
                stream_conds[stream].adsorbent_mass
                / tower_conds.common.num_sections
            )
            
            # 脱着量（差分）[Ncm³/g-abs]
            loading_delta = previous_loading - current_loading
            
            # セクション全体での脱着量 [Ncm³]
            section_desorption = loading_delta * section_adsorbent_mass
            total_desorption_volume += section_desorption
    
    # CO2回収量 [Nm³]
    co2_this_step = total_desorption_volume * CM3_TO_L * L_TO_M3
    
    # N2回収量を平均モル分率から計算 [Nm³]
    if avg_co2_mole_fraction > 0:
        n2_this_step = co2_this_step * avg_n2_mole_fraction / avg_co2_mole_fraction
    else:
        n2_this_step = 0.0
    
    # 累積回収量に加算
    cumulative_co2 = tower.cumulative_co2_recovered + co2_this_step
    cumulative_n2 = tower.cumulative_n2_recovered + n2_this_step
    
    return cumulative_co2, cumulative_n2


def calculate_co2_recovery_concentration(
    cumulative_co2: float,
    cumulative_n2: float,
) -> float:
    """
    CO2回収濃度を計算
    
    回収ガス中のCO2濃度を計算します。
    
    Args:
        cumulative_co2: 累積CO2回収量 [Nm³]
        cumulative_n2: 累積N2回収量 [Nm³]
    
    Returns:
        CO2回収濃度 [%]
    """
    total_recovered = cumulative_co2 + cumulative_n2
    
    if total_recovered > 0:
        return (cumulative_co2 / total_recovered) * 100
    else:
        return 0.0
