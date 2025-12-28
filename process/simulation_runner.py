"""シミュレーション実行モジュール

シミュレーションの計算ロジックを実行するクラスを提供します。
入出力処理は含まず、純粋な計算のみを担当します。

使用例:
    from process.simulation_runner import SimulationRunner
    from process.simulation_io import SimulationIO
    from state import StateVariables
    
    # 条件読み込み
    io = SimulationIO()
    sim_conds = io.load_conditions("5_08_mod_logging2")
    df_operation = io.load_operation_schedule("5_08_mod_logging2")
    
    # 状態変数初期化
    state_manager = StateVariables(...)
    
    # シミュレーション実行
    runner = SimulationRunner(sim_conds, state_manager, df_operation)
    results, process_log, timestamp = runner.run()
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.sim_conditions import SimulationConditions
from state import StateVariables
from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
from process.simulation_results import SimulationResults
import logger as log


@dataclass
class ProcessResult:
    """プロセス実行結果
    
    Attributes:
        timestamp: 終了時点のタイムスタンプ（分）
        simulation_results: シミュレーション結果
        success: 成功フラグ
        error_message: エラーメッセージ（失敗時）
    """
    timestamp: float
    simulation_results: SimulationResults
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SimulationOutput:
    """シミュレーション出力
    
    Attributes:
        results: シミュレーション結果
        process_completion_log: 各工程の終了時刻
        final_timestamp: 最終タイムスタンプ（分）
        success: 成功フラグ
    """
    results: SimulationResults
    process_completion_log: Dict[int, float]
    final_timestamp: float
    success: bool = True


class SimulationRunner:
    """シミュレーション実行クラス
    
    シミュレーションの計算ロジックを実行します。
    条件ファイルの読み込みや結果の出力は行いません。
    """
    
    def __init__(
        self,
        sim_conds: SimulationConditions,
        state_manager: StateVariables,
        df_operation: pd.DataFrame,
    ):
        """
        初期化
        
        Args:
            sim_conds: シミュレーション条件
            state_manager: 状態変数管理
            df_operation: 稼働工程表
        """
        self.logger = log.logger.getChild(__name__)
        
        self.sim_conds = sim_conds
        self.state_manager = state_manager
        self.df_operation = df_operation
        
        # 基本パラメータ
        self.num_tower = sim_conds.num_towers
        self.dt = sim_conds.get_tower(1).common.calculation_step_time
        self.num_str = sim_conds.get_tower(1).common.num_streams
        self.num_sec = sim_conds.get_tower(1).common.num_sections
        
        # 塔内の残留ガス情報
        self.residual_gas_composition: Optional[dict] = None
    
    def run(self) -> SimulationOutput:
        """
        シミュレーションを実行
        
        Returns:
            SimulationOutput: シミュレーション出力
        """
        # 結果格納用
        simulation_results = SimulationResults()
        for tower_num in range(1, 1 + self.num_tower):
            simulation_results.initialize_tower(tower_num)
        
        # プロセス終了時刻記録用
        process_completion_log = {key: 0.0 for key in range(1, 1 + len(self.df_operation))}
        
        # 全工程を実行
        timestamp = 0.0
        simulation_success = True
        
        for process_index in self.df_operation.index:
            mode_list = list(self.df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
            termination_cond_str = self.df_operation.loc[process_index, "終了条件"]
            
            process_result = self._execute_process(
                mode_list=mode_list,
                termination_cond_str=termination_cond_str,
                simulation_results=simulation_results,
                timestamp=timestamp,
            )
            
            timestamp = process_result.timestamp
            simulation_results = process_result.simulation_results
            process_completion_log[process_index] = round(timestamp, 2)
            
            if not process_result.success:
                self.logger.warning(f"工程 {process_index} でエラーが発生したため、後続処理をスキップします")
                simulation_success = False
                break
            
            self.logger.info(f"プロセス{process_index}終了 timestamp: {round(timestamp, 2)}")
        
        return SimulationOutput(
            results=simulation_results,
            process_completion_log=process_completion_log,
            final_timestamp=timestamp,
            success=simulation_success,
        )
    
    def _execute_process(
        self,
        mode_list: List[str],
        termination_cond_str: str,
        simulation_results: SimulationResults,
        timestamp: float,
    ) -> ProcessResult:
        """
        1工程を実行
        
        Args:
            mode_list: 各塔のモードリスト
            termination_cond_str: 終了条件文字列
            simulation_results: シミュレーション結果
            timestamp: 開始タイムスタンプ
            
        Returns:
            ProcessResult: 工程実行結果
        """
        elapsed_time = 0.0
        
        # 初回限定処理（バッチ吸着の圧力平均化）
        prepare_batch_adsorption_pressure(self.state_manager, self.sim_conds, mode_list)
        
        # 逐次計算
        while self._check_termination_condition(termination_cond_str, timestamp, elapsed_time):
            # 各塔の吸着計算
            outputs, new_residual = execute_mode_list(
                sim_conds=self.sim_conds,
                mode_list=mode_list,
                state_manager=self.state_manager,
                residual_gas_composition=self.residual_gas_composition,
            )
            
            # residual_gas_composition更新
            if new_residual is not None:
                self.residual_gas_composition = new_residual
            
            # タイムスタンプ更新
            elapsed_time += self.dt
            
            # 結果記録
            for tower_num, output in outputs.items():
                current_timestamp = timestamp + elapsed_time
                simulation_results.add_tower_result(
                    tower_id=tower_num,
                    timestamp=current_timestamp,
                    material=output.material,
                    heat=output.heat,
                    heat_wall=output.heat_wall,
                    heat_lid=output.heat_lid,
                    others=output.others,
                )
            
            # 時間超過による強制終了
            time_threshold = 20
            if elapsed_time >= time_threshold:
                self.logger.warning(f"{time_threshold}分以内に終了しなかったため強制終了")
                return ProcessResult(
                    timestamp=timestamp + elapsed_time,
                    simulation_results=simulation_results,
                    success=False,
                    error_message=f"時間超過（{time_threshold}分）",
                )
        
        return ProcessResult(
            timestamp=timestamp + elapsed_time,
            simulation_results=simulation_results,
            success=True,
        )
    
    def _check_termination_condition(
        self,
        termination_cond_str: str,
        timestamp: float,
        elapsed_time: float,
    ) -> bool:
        """
        終了条件をチェック
        
        Args:
            termination_cond_str: 終了条件文字列
            timestamp: 工程開始時のタイムスタンプ
            elapsed_time: 工程内経過時間
            
        Returns:
            bool: 継続する場合True、終了する場合False
        """
        cond_list = termination_cond_str.split("_")
        
        if cond_list[0] == "圧力到達":
            tower_num = int(cond_list[1][-1])
            target_press = float(cond_list[2])
            return self.state_manager.towers[tower_num].total_press < target_press
        
        elif cond_list[0] == "温度到達":
            tower_num = int(cond_list[1][-1])
            target_temp = float(cond_list[2])
            target_section = self.num_sec
            temp_now = np.mean(self.state_manager.towers[tower_num].temp[:, target_section - 1])
            return temp_now < target_temp
        
        elif cond_list[0] == "時間経過":
            time = float(cond_list[1])
            unit = cond_list[2]
            if unit == "s":
                time /= 60
            return elapsed_time < time
        
        elif cond_list[0] == "時間到達":
            time = float(cond_list[1])
            return timestamp + elapsed_time < time
        
        return False
