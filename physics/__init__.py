"""物理計算モジュール

PSAプロセスで使用する純粋な物理計算を提供します。

設計方針:
- 各モジュールは単一の物理量/物理式を計算
- モード固有のオーケストレーションはoperation_modes/で行う
- 状態に依存しない純粋な関数として実装

主要なモジュール:
- mass_balance.py: 物質収支計算（吸着・脱着・停止モード）
- heat_balance.py: 熱収支計算（層・壁・蓋）
- pressure.py: 圧力計算（状態方程式ベース）
- gas_properties.py: ガス物性計算（粘度・密度）
- pipe_flow.py: 配管流量・圧力損失計算
- recovery.py: 回収量計算（脱着量）
- adsorption_isotherm.py: 吸着平衡線（平衡吸着量）

使用例:
    # 物質収支計算
    from physics.mass_balance import calculate_mass_balance
    result = calculate_mass_balance(mode, tower_conds, stream, section, ...)
    
    # 熱収支計算
    from physics.heat_balance import calculate_bed_heat_balance
    result = calculate_bed_heat_balance(tower_conds, stream, section, ...)
    
    # ガス物性計算
    from physics.gas_properties import calculate_mixed_gas_viscosity
    viscosity = calculate_mixed_gas_viscosity(T_K, co2_mf, n2_mf)
    
    # 配管流量計算
    from physics.pipe_flow import calculate_vacuum_pump_flow
    result = calculate_vacuum_pump_flow(tower_conds, pressure, T_K, viscosity, density)
    
    # 圧力計算
    from physics.pressure import calculate_pressure_after_batch_adsorption
    pressure = calculate_pressure_after_batch_adsorption(tower_conds, state_manager, tower_num, ...)
    
    # 吸着平衡線
    from physics.adsorption_isotherm import calculate_equilibrium_loading
    q_eq = calculate_equilibrium_loading(pressure_kpa, temperature_k)
"""

# 吸着平衡線（外部依存なし、常にインポート可能）
from .adsorption_isotherm import calculate_equilibrium_loading

# 以下のモジュールは CoolProp 等の外部依存があるため、
# インポートエラーを許容する（実際の使用時にインポート）
try:
    # 物質収支計算
    from .mass_balance import calculate_mass_balance
    
    # 熱収支計算
    from .heat_balance import (
        calculate_bed_heat_balance,
        calculate_wall_heat_balance,
        calculate_lid_heat_balance,
    )
    
    # 圧力計算
    from .pressure import (
        calculate_pressure_after_vacuum_desorption,
        calculate_pressure_after_batch_adsorption,
    )
    
    # ガス物性計算
    from .gas_properties import (
        calculate_mixed_gas_viscosity,
        calculate_mixed_gas_density,
    )
    
    # 配管流量・圧力損失計算
    from .pipe_flow import (
        calculate_vacuum_pump_flow,
        calculate_equalization_flow,
        calculate_pressure_change_from_moles,
        calculate_pressure_from_moles,
        calculate_moles_from_pressure,
    )
    
    # 回収量計算
    from .recovery import (
        calculate_desorption_amount,
        calculate_co2_recovery_concentration,
    )
except ImportError as e:
    # CoolProp等がインストールされていない環境向け
    import warnings
    warnings.warn(f"一部の物理計算モジュールがインポートできません: {e}")

__all__ = [
    # 物質収支
    "calculate_mass_balance",
    "calculate_equilibrium_loading",
    
    # 熱収支
    "calculate_bed_heat_balance",
    "calculate_wall_heat_balance",
    "calculate_lid_heat_balance",
    
    # 圧力計算
    "calculate_pressure_after_vacuum_desorption",
    "calculate_pressure_after_batch_adsorption",
    
    # ガス物性
    "calculate_mixed_gas_viscosity",
    "calculate_mixed_gas_density",
    
    # 配管流量・圧力損失
    "calculate_vacuum_pump_flow",
    "calculate_equalization_flow",
    "calculate_pressure_change_from_moles",
    "calculate_pressure_from_moles",
    "calculate_moles_from_pressure",
    
    # 回収量
    "calculate_desorption_amount",
    "calculate_co2_recovery_concentration",
]
