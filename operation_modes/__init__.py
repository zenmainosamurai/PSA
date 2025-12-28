"""運転モード

各運転モード（流通吸着、バッチ吸着、真空脱着など）の定義と計算ロジックを提供します。

使用例:
    from operation_modes import (
        OperationMode,
        execute_flow_adsorption_upstream,
        execute_vacuum_desorption,
    )
    
    # 流通吸着の実行
    result = execute_flow_adsorption_upstream(tower_conds, state_manager, tower_num=1)
    
    # 真空脱着の実行
    result = execute_vacuum_desorption(tower_conds, state_manager, tower_num=1)

モジュール構成:
- mode_types.py: 運転モードのEnum定義とグループ分け
- common.py: 全モード共通の計算処理
- stop.py: 停止モード
- flow_adsorption.py: 流通吸着モード
- batch_adsorption.py: バッチ吸着モード
- equalization.py: 均圧モード
- vacuum_desorption.py: 真空脱着モード
- initial_gas_introduction.py: 初回ガス導入モード
"""

# モード定義
from .mode_types import (
    OperationMode,
    HeatCalculationMode,
    ADSORPTION_MODES,
    UPSTREAM_MODES,
    DOWNSTREAM_MODES,
    UPSTREAM_DOWNSTREAM_PAIRS,
    EQUALIZATION_MODES,
    PRESSURE_UPDATE_MODES,
    MOLE_FRACTION_UPDATE_MODES,
    BATCH_PRESSURE_CALCULATION_MODES,
    FLOW_PRESSURE_MODES,
    get_mode_category,
)

# 共通処理
from .common import (
    CellCalculationResults,
    FullTowerResults,
    calculate_all_cells,
    calculate_wall_heat,
    calculate_lid_heat,
    calculate_full_tower,
    distribute_inflow_gas,
)

# 停止モード
from .stop import (
    StopModeResult,
    execute_stop_mode,
)

# 流通吸着モード
from .flow_adsorption import (
    FlowAdsorptionResult,
    execute_flow_adsorption_upstream,
    execute_flow_adsorption_downstream,
)

# バッチ吸着モード
from .batch_adsorption import (
    BatchAdsorptionResult,
    execute_batch_adsorption_upstream,
    execute_batch_adsorption_upstream_with_valve,
    execute_batch_adsorption_downstream,
    execute_batch_adsorption_downstream_with_valve,
)

# 均圧モード
from .equalization import (
    EqualizationDepressurizationResult,
    EqualizationPressurizationResult,
    execute_equalization_depressurization,
    execute_equalization_pressurization,
)

# 真空脱着モード
from .vacuum_desorption import (
    VacuumDesorptionResult,
    execute_vacuum_desorption,
)

# 初回ガス導入モード
from .initial_gas_introduction import (
    InitialGasIntroductionResult,
    execute_initial_gas_introduction,
)

__all__ = [
    # モード定義
    "OperationMode",
    "HeatCalculationMode",
    "ADSORPTION_MODES",
    "UPSTREAM_MODES",
    "DOWNSTREAM_MODES",
    "UPSTREAM_DOWNSTREAM_PAIRS",
    "EQUALIZATION_MODES",
    "PRESSURE_UPDATE_MODES",
    "MOLE_FRACTION_UPDATE_MODES",
    "BATCH_PRESSURE_CALCULATION_MODES",
    "FLOW_PRESSURE_MODES",
    "get_mode_category",
    # 共通処理
    "CellCalculationResults",
    "FullTowerResults",
    "calculate_all_cells",
    "calculate_wall_heat",
    "calculate_lid_heat",
    "calculate_full_tower",
    "distribute_inflow_gas",
    # 停止モード
    "StopModeResult",
    "execute_stop_mode",
    # 流通吸着モード
    "FlowAdsorptionResult",
    "execute_flow_adsorption_upstream",
    "execute_flow_adsorption_downstream",
    # バッチ吸着モード
    "BatchAdsorptionResult",
    "execute_batch_adsorption_upstream",
    "execute_batch_adsorption_upstream_with_valve",
    "execute_batch_adsorption_downstream",
    "execute_batch_adsorption_downstream_with_valve",
    # 均圧モード
    "EqualizationDepressurizationResult",
    "EqualizationPressurizationResult",
    "execute_equalization_depressurization",
    "execute_equalization_pressurization",
    # 真空脱着モード
    "VacuumDesorptionResult",
    "execute_vacuum_desorption",
    # 初回ガス導入モード
    "InitialGasIntroductionResult",
    "execute_initial_gas_introduction",
]
