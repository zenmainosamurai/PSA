"""物理計算モジュール

PSAプロセスで使用する物理計算を提供します。

主要なモジュール:
- mass_balance.py: 物質収支計算（吸着・脱着・停止モード）
- heat_balance.py: 熱収支計算（層・壁・蓋）
- pressure.py: 圧力計算（真空排気・均圧・バッチ後）
- adsorption_isotherm.py: 吸着平衡線（平衡吸着量）

使用例:
    # 物質収支計算
    from physics.mass_balance import calculate_mass_balance
    result = calculate_mass_balance(mode, tower_conds, stream, section, ...)
    
    # 熱収支計算
    from physics.heat_balance import calculate_bed_heat_balance
    result = calculate_bed_heat_balance(tower_conds, stream, section, ...)
    
    # 圧力計算
    from physics.pressure import calculate_vacuum_pumping
    result = calculate_vacuum_pumping(tower_conds, state_manager, tower_num)
    
    # 吸着平衡線
    from physics.adsorption_isotherm import calculate_equilibrium_loading
    q_eq = calculate_equilibrium_loading(pressure_kpa, temperature_k)
"""

# 吸着平衡線（外部依存なし、常にインポート可能）
from .adsorption_isotherm import (
    calculate_equilibrium_loading,
    calculate_loading_at_conditions,
    calculate_driving_force,
)

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
        calculate_vacuum_pumping,
        calculate_depressurization,
        calculate_downstream_flow,
        calculate_pressure_after_vacuum_desorption,
        calculate_pressure_after_batch_adsorption,
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
    
    # 圧力
    "calculate_vacuum_pumping",
    "calculate_depressurization",
    "calculate_downstream_flow",
    "calculate_pressure_after_vacuum_desorption",
    "calculate_pressure_after_batch_adsorption",
    
    # 吸着平衡線
    "calculate_loading_at_conditions",
    "calculate_driving_force",
]
