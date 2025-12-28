"""シミュレーター（ファサード）

PSA担当者向け説明:
PSAシミュレーションを実行するためのメインクラスです。
内部的には以下の3つのクラスに責務を分離しています：

- SimulationIO: 入出力（条件/観測データ/稼働工程表の読み込み）
- SimulationRunner: シミュレーション実行（計算ロジック）
- ResultExporter: 結果出力（CSV/PNG/XLSX）

使用例:
    from process import GasAdsorptionBreakthroughSimulator
    
    simulator = GasAdsorptionBreakthroughSimulator("5_08_mod_logging2")
    simulator.execute_simulation()

旧コードとの互換性:
    GasAdosorptionBreakthroughsimulator は GasAdsorptionBreakthroughSimulator のエイリアスです。
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from common.paths import OUTPUT_DIR
from config.sim_conditions import SimulationConditions
from state import StateVariables
from process.simulation_io import SimulationIO
from process.simulation_runner import SimulationRunner, SimulationOutput
from process.result_exporter import ResultExporter
from process.simulation_results import SimulationResults
import logger as log


@dataclass
class ProcessResults:
    """プロセス毎の結果を保持するデータクラス
    
    後方互換性のために残しています。
    新規コードでは ProcessResult（simulation_runner.py）を使用してください。
    """
    timestamp: float
    simulation_results: SimulationResults
    success: bool = True
    error_message: Optional[str] = None


class GasAdsorptionBreakthroughSimulator:
    """ガス吸着モデル（バッチプロセス）を実行するクラス
    
    PSA担当者向け説明:
    PSAシミュレーションを実行するメインクラスです。
    条件ファイルの読み込み、シミュレーション実行、結果出力を行います。
    
    属性:
        cond_id: 条件ID（例: "5_08_mod_logging2"）
        sim_conds: シミュレーション条件
        state_manager: 状態変数管理
        df_operation: 稼働工程表
        df_obs: 観測データ（存在しない場合はNone）
    """
    
    def __init__(self, cond_id: str):
        """
        初期化
        
        Args:
            cond_id: 条件ID（例: "5_08_mod_logging2"）
        """
        self.logger = log.logger.getChild(__name__)
        self.cond_id = cond_id
        
        # 入出力クラス
        self._io = SimulationIO()
        
        # 条件読み込み
        self.sim_conds = self._io.load_conditions(cond_id)
        self.num_tower = self.sim_conds.num_towers
        self.dt = self.sim_conds.get_tower(1).common.calculation_step_time
        self.num_str = self.sim_conds.get_tower(1).common.num_streams
        self.num_sec = self.sim_conds.get_tower(1).common.num_sections
        
        # 観測データ読み込み
        self.df_obs = self._io.load_observation_data(self.dt)
        
        # 稼働工程表読み込み
        self.df_operation = self._io.load_operation_schedule(cond_id)
        
        # 状態変数初期化
        self.state_manager = StateVariables(
            self.num_tower, self.num_str, self.num_sec, self.sim_conds
        )
        
        # 塔内の残留ガス情報（後方互換性のため）
        self.residual_gas_composition: Optional[dict] = None
    
    def execute_simulation(
        self,
        filtered_states=None,
        output_folderpath: Optional[str] = None,
    ) -> None:
        """
        シミュレーションを実行
        
        Args:
            filtered_states: データ同化で得られた状態変数（省略可）
            output_folderpath: 出力先フォルダパス（省略時はデフォルト）
        """
        # 出力先フォルダの決定
        if output_folderpath is None:
            output_folderpath = f"{OUTPUT_DIR}{self.cond_id}/"
        os.makedirs(output_folderpath, exist_ok=True)
        
        # シミュレーション実行
        runner = SimulationRunner(
            sim_conds=self.sim_conds,
            state_manager=self.state_manager,
            df_operation=self.df_operation,
        )
        output = runner.run()
        
        # residual_gas_compositionを同期（後方互換性のため）
        self.residual_gas_composition = runner.residual_gas_composition
        
        # 結果出力
        if output.success:
            exporter = ResultExporter(self.sim_conds)
            exporter.export_all(
                output_dir=output_folderpath,
                simulation_results=output.results,
                df_operation=self.df_operation,
                process_completion_log=output.process_completion_log,
                df_obs=self.df_obs,
                timestamp=output.final_timestamp,
            )
    
    # ============================================================
    # 後方互換性のためのメソッド
    # 新規コードではSimulationRunner/ResultExporterを直接使用してください
    # ============================================================
    
    def calc_adsorption_process(
        self,
        mode_list: List[str],
        termination_cond_str: str,
        simulation_results: SimulationResults,
        timestamp: float,
        filtered_x=None,
    ):
        """
        プロセスのガス吸着計算を行う
        
        後方互換性のために残しています。
        新規コードでは SimulationRunner を直接使用してください。
        """
        runner = SimulationRunner(
            sim_conds=self.sim_conds,
            state_manager=self.state_manager,
            df_operation=self.df_operation,
        )
        runner.residual_gas_composition = self.residual_gas_composition
        
        # 1工程を実行
        result = runner._execute_process(
            mode_list=mode_list,
            termination_cond_str=termination_cond_str,
            simulation_results=simulation_results,
            timestamp=timestamp,
        )
        
        # residual_gas_compositionを同期
        self.residual_gas_composition = runner.residual_gas_composition
        
        return result.timestamp, result.simulation_results, result.success
    
    def calc_adsorption_mode_list(self, sim_conds: SimulationConditions, mode_list: List[str]):
        """
        モードリストに基づく各塔のガス吸着計算
        
        後方互換性のために残しています。
        新規コードでは process_executor.execute_mode_list を直接使用してください。
        """
        from process.process_executor import execute_mode_list
        
        outputs, new_residual = execute_mode_list(
            sim_conds=sim_conds,
            mode_list=mode_list,
            state_manager=self.state_manager,
            residual_gas_composition=self.residual_gas_composition,
        )
        
        if new_residual is not None:
            self.residual_gas_composition = new_residual
        
        # TowerCalculationOutputをrecord_items形式に変換
        record_outputs_tower = {}
        for tower_num, output in outputs.items():
            record_outputs_tower[tower_num] = {
                "material": output.material,
                "heat": output.heat,
                "heat_wall": output.heat_wall,
                "heat_lid": output.heat_lid,
                "others": output.others,
            }
        
        return record_outputs_tower


# 後方互換性のためのエイリアス（タイポを含む旧名称）
GasAdosorptionBreakthroughsimulator = GasAdsorptionBreakthroughSimulator
