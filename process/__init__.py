"""プロセス制御モジュール

PSA担当者向け説明:
PSA工程（稼働工程表）に従ってシミュレーションを進行させる部分です。

使用例:
    from process import GasAdsorptionBreakthroughSimulator
    
    simulator = GasAdsorptionBreakthroughSimulator("5_08_mod_logging2")
    simulator.execute_simulation()

モジュール構成:
- simulator.py: シミュレーター本体
- process_executor.py: 工程実行ロジック（塔間依存の制御）
- termination_conditions.py: 終了条件判定（圧力到達・温度到達・時間経過など）
- simulation_results.py: シミュレーション結果データクラス
"""

# シミュレーター
from .simulator import GasAdosorptionBreakthroughsimulator, ProcessResults

# シミュレーション結果
from .simulation_results import SimulationResults

# 工程実行
from .process_executor import (
    TowerCalculationOutput,
    execute_mode_list,
    prepare_batch_adsorption_pressure,
)

# 終了条件
from .termination_conditions import (
    TerminationConditionType,
    TerminationCondition,
    parse_termination_condition,
    check_termination_condition,
    should_continue_process,
)

__all__ = [
    # シミュレーター
    "GasAdosorptionBreakthroughsimulator",
    "ProcessResults",
    "SimulationResults",
    # 工程実行
    "TowerCalculationOutput",
    "execute_mode_list",
    "prepare_batch_adsorption_pressure",
    # 終了条件
    "TerminationConditionType",
    "TerminationCondition",
    "parse_termination_condition",
    "check_termination_condition",
    "should_continue_process",
]
