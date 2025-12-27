"""プロセス制御モジュール

PSA担当者向け説明:
PSA工程（稼働工程表）に従ってシミュレーションを進行させる部分です。

使用例:
    from process import PSASimulator
    
    # シミュレーション実行
    simulator = PSASimulator("test_condition")
    simulator.run()

モジュール構成:
- simulator.py: シミュレーター本体（軽量化版）
- process_executor.py: 工程実行ロジック（塔間依存の制御）
- termination_conditions.py: 終了条件判定（圧力到達・温度到達・時間経過など）
"""

# シミュレーター
from .simulator import PSASimulator, ProcessResult

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
    "PSASimulator",
    "ProcessResult",
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
