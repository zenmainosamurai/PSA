"""プロセス制御モジュール

PSA工程（稼働工程表）に従ってシミュレーションを進行させる部分です。

使用例:
    # 簡単な使い方（従来通り）
    from process import GasAdsorptionBreakthroughSimulator
    
    simulator = GasAdsorptionBreakthroughSimulator("5_08_mod_logging2")
    simulator.execute_simulation()
    
    # 責務分離した使い方（推奨）
    from process import SimulationIO, SimulationRunner, ResultExporter
    from state import StateVariables
    
    io = SimulationIO()
    sim_conds = io.load_conditions("5_08_mod_logging2")
    df_operation = io.load_operation_schedule("5_08_mod_logging2")
    df_obs = io.load_observation_data(dt=0.01)
    
    state_manager = StateVariables(...)
    
    runner = SimulationRunner(sim_conds, state_manager, df_operation)
    output = runner.run()
    
    exporter = ResultExporter(sim_conds)
    exporter.export_all(output_dir, output.results, ...)

モジュール構成:
- simulator.py: シミュレーター（ファサード、後方互換性維持）
- simulation_io.py: 入出力（条件/観測データ/稼働工程表の読み込み）
- simulation_runner.py: シミュレーション実行（計算ロジック）
- result_exporter.py: 結果出力（CSV/PNG/XLSX）
- process_executor.py: 工程実行ロジック（塔間依存の制御）
- termination_conditions.py: 終了条件判定
- simulation_results.py: シミュレーション結果データクラス
"""

# シミュレーター（ファサード）
from .simulator import (
    GasAdsorptionBreakthroughSimulator,
    ProcessResults,
)

# 入出力
from .simulation_io import SimulationIO

# シミュレーション実行
from .simulation_runner import (
    SimulationRunner,
    ProcessResult,
    SimulationOutput,
)

# 結果出力
from .result_exporter import ResultExporter

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
    # シミュレーター（ファサード）
    "GasAdsorptionBreakthroughSimulator",
    "ProcessResults",
    # 入出力
    "SimulationIO",
    # シミュレーション実行
    "SimulationRunner",
    "ProcessResult",
    "SimulationOutput",
    # 結果出力
    "ResultExporter",
    # シミュレーション結果
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
