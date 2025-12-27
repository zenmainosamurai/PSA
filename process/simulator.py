"""シミュレーター本体

PSA担当者向け説明:
PSAシミュレーションのメインクラスです。
稼働工程表に従ってシミュレーションを実行し、結果を出力します。

使用方法:
    simulator = PSASimulator("test_condition")
    simulator.run()

シミュレーションの流れ:
1. 条件ファイルの読み込み（sim_conds.xlsx, 稼働工程表.xlsx）
2. 状態変数の初期化
3. 各工程の実行（終了条件まで繰り返し）
4. 結果の出力（CSV, PNG, Excel）

ログ出力:
- 各工程の開始・終了時刻
- エラー発生時の警告
- 出力ファイルの生成状況
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from config.sim_conditions import SimulationConditions
from core.state import StateVariables
from core.simulation_results import SimulationResults
from process.process_executor import (
    execute_mode_list,
    prepare_batch_adsorption_pressure,
    TowerCalculationOutput,
)
from process.termination_conditions import should_continue_process
from utils import const, plot_csv, plot_xlsx, other_utils
import log


@dataclass
class ProcessResult:
    """
    工程実行結果
    
    PSA担当者向け説明:
    1つの工程（稼働工程表の1行）の実行結果を保持します。
    """
    timestamp: float  # 工程終了時の時刻 [min]
    simulation_results: SimulationResults  # 累積シミュレーション結果
    success: bool = True  # 正常終了したか
    error_message: Optional[str] = None  # エラーメッセージ


class PSASimulator:
    """
    PSAシミュレーター
    
    PSA担当者向け説明:
    このクラスがシミュレーション全体を制御します。
    
    主な機能:
    - 条件ファイルの読み込み
    - 稼働工程表に従った計算の実行
    - 結果のCSV/PNG/Excel出力
    
    使用例:
        simulator = PSASimulator("condition_001")
        simulator.run()
    """
    
    def __init__(self, cond_id: str):
        """
        初期化
        
        Args:
            cond_id: 条件ID（conditionsフォルダ内のサブフォルダ名）
        """
        self.logger = log.logger.getChild(__name__)
        self.cond_id = cond_id
        
        # 条件読み込み
        self.sim_conds = SimulationConditions(cond_id)
        self.num_towers = self.sim_conds.num_towers
        self.dt = self.sim_conds.get_tower(1).common.calculation_step_time
        self.num_streams = self.sim_conds.get_tower(1).common.num_streams
        self.num_sections = self.sim_conds.get_tower(1).common.num_sections
        
        # 観測データ読み込み（オプション）
        self.df_obs = self._load_observation_data()
        
        # 稼働工程表読み込み
        self.df_operation = self._load_operation_schedule()
        
        # 状態変数初期化
        self.state_manager = StateVariables(
            self.num_towers, self.num_streams, self.num_sections, self.sim_conds
        )
        
        # 残留ガス組成（バッチ吸着下流で使用）
        self.residual_gas_composition = None
    
    def run(self, output_path: Optional[str] = None) -> None:
        """
        シミュレーション実行
        
        PSA担当者向け説明:
        稼働工程表に従ってシミュレーションを実行し、
        結果をCSV/PNG/Excelに出力します。
        
        Args:
            output_path: 出力先パス（省略時はデフォルト）
        
        使用例:
            simulator.run()  # デフォルト出力先
            simulator.run("/path/to/output")  # 指定パス
        """
        simulation_results = SimulationResults()
        
        # 塔の初期化
        for tower_num in range(1, self.num_towers + 1):
            simulation_results.initialize_tower(tower_num)
        
        # 出力先フォルダ
        if output_path is None:
            output_path = const.OUTPUT_DIR + f"{self.cond_id}/"
        os.makedirs(output_path, exist_ok=True)
        
        # 工程完了時刻記録
        process_completion_log = {
            key: 0 for key in range(1, len(self.df_operation) + 1)
        }
        
        # 各工程の実行
        timestamp = 0
        simulation_success = True
        
        for process_index in self.df_operation.index:
            mode_list = list(self.df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
            termination_cond = self.df_operation.loc[process_index, "終了条件"]
            
            process_result = self._execute_process(
                process_index=process_index,
                mode_list=mode_list,
                termination_cond_str=termination_cond,
                simulation_results=simulation_results,
                timestamp=timestamp,
            )
            
            timestamp = process_result.timestamp
            simulation_results = process_result.simulation_results
            process_completion_log[process_index] = round(timestamp, 2)
            
            if not process_result.success:
                self.logger.warning(
                    f"工程 {process_index} でエラーが発生したため、後続処理をスキップします"
                )
                simulation_success = False
                break
            
            self.logger.info(f"プロセス{process_index}終了 timestamp: {round(timestamp, 2)}")
        
        # 結果出力
        if simulation_success:
            self._output_results(
                output_path, simulation_results, process_completion_log, timestamp
            )
    
    def _execute_process(
        self,
        process_index: int,
        mode_list: List[str],
        termination_cond_str: str,
        simulation_results: SimulationResults,
        timestamp: float,
    ) -> ProcessResult:
        """単一工程の実行"""
        timestamp_in_process = 0
        
        # バッチ吸着の圧力平均化（必要な場合）
        prepare_batch_adsorption_pressure(
            self.state_manager, self.sim_conds, mode_list
        )
        
        # 終了条件判定しながら繰り返し計算
        while should_continue_process(
            termination_cond_str,
            self.state_manager,
            timestamp,
            timestamp_in_process,
            self.num_sections,
        ):
            # 各塔の計算実行
            outputs, self.residual_gas_composition = execute_mode_list(
                sim_conds=self.sim_conds,
                mode_list=mode_list,
                state_manager=self.state_manager,
                residual_gas_composition=self.residual_gas_composition,
            )
            
            # 時間更新
            timestamp_in_process += self.dt
            current_timestamp = timestamp + timestamp_in_process
            
            # 結果記録
            for tower_num in range(1, self.num_towers + 1):
                tower_output = outputs[tower_num]
                simulation_results.add_tower_result(
                    tower_id=tower_num,
                    timestamp=current_timestamp,
                    material=tower_output.material,
                    heat=tower_output.heat,
                    heat_wall=tower_output.heat_wall,
                    heat_lid=tower_output.heat_lid,
                    others=tower_output.others,
                )
            
            # タイムアウトチェック
            time_threshold = 20  # 20分
            if timestamp_in_process >= time_threshold:
                self.logger.warning(
                    f"{time_threshold}分以内に終了しなかったため強制終了"
                )
                return ProcessResult(
                    timestamp=timestamp + timestamp_in_process,
                    simulation_results=simulation_results,
                    success=False,
                    error_message="タイムアウト",
                )
        
        return ProcessResult(
            timestamp=timestamp + timestamp_in_process,
            simulation_results=simulation_results,
            success=True,
        )
    
    def _load_observation_data(self) -> Optional[pd.DataFrame]:
        """観測データの読み込み"""
        filepath = const.DATA_DIR + "3塔データ.csv"
        try:
            if filepath.lower().endswith("csv"):
                df = pd.read_csv(filepath, index_col=0)
            else:
                df = pd.read_excel(
                    filepath,
                    sheet_name=self.sim_conds[1]["sheet_name"],
                    index_col="time",
                )
            return other_utils.resample_obs_data(df, self.dt)
        except FileNotFoundError:
            self.logger.warning(
                f"観測データファイルが存在しないため比較をスキップします: {filepath}"
            )
            return None
        except Exception as exc:
            self.logger.error(
                f"観測データの読み込みに失敗しましたが処理を継続します: {exc}"
            )
            return None
    
    def _load_operation_schedule(self) -> pd.DataFrame:
        """稼働工程表の読み込み"""
        filepath = const.CONDITIONS_DIR + self.cond_id + "/" + "稼働工程表.xlsx"
        return pd.read_excel(filepath, index_col="工程", sheet_name="工程")
    
    def _output_results(
        self,
        output_path: str,
        simulation_results: SimulationResults,
        process_completion_log: Dict,
        timestamp: float,
    ) -> None:
        """結果出力"""
        # 単位変換
        self._apply_unit_conversion(simulation_results)
        
        # グラフ対象セクション
        plot_target_sec = self.sim_conds.get_tower(1).common.get_sections_for_graph()
        
        # Excel出力（オプション）
        use_xlsx = self.sim_conds.get_tower(1).common.use_xlsx
        if use_xlsx == 1:
            self._output_xlsx(output_path, simulation_results, plot_target_sec, timestamp)
        
        # CSV出力
        self._output_csv(output_path, simulation_results)
        
        # 工程終了時刻出力
        self.df_operation["終了時刻(min)"] = list(process_completion_log.values())
        self.df_operation.to_csv(
            output_path + "/プロセス終了時刻.csv", encoding="shift-jis"
        )
        
        # PNG出力
        self._output_png(output_path, plot_target_sec, timestamp)
    
    def _output_xlsx(
        self,
        output_path: str,
        simulation_results: SimulationResults,
        plot_target_sec: List[int],
        timestamp: float,
    ) -> None:
        """Excel出力"""
        self.logger.info("xlsx出力開始")
        
        for tower_num in range(1, self.num_towers + 1):
            xlsx_path = output_path + f"/xlsx/tower_{tower_num}/"
            os.makedirs(xlsx_path, exist_ok=True)
            tower_results = simulation_results.tower_simulation_results[tower_num]
            plot_xlsx.outputs_to_xlsx(
                xlsx_path,
                tower_results,
                self.sim_conds.get_tower(tower_num).common,
                plot_target_sec,
                self.df_obs,
                tower_num,
            )
        
        self.logger.info("xlsxグラフ出力開始")
        
        for tower_num in range(1, self.num_towers + 1):
            plot_xlsx.plot_xlsx_outputs(
                tgt_foldapath=output_path,
                df_obs=self.df_obs,
                tgt_sections=plot_target_sec,
                tower_num=tower_num,
                timestamp=timestamp,
                df_p_end=self.df_operation,
            )
    
    def _output_csv(
        self,
        output_path: str,
        simulation_results: SimulationResults,
    ) -> None:
        """CSV出力"""
        self.logger.info("csv出力開始")
        
        for tower_num in range(1, self.num_towers + 1):
            csv_path = output_path + f"/csv/tower_{tower_num}/"
            os.makedirs(csv_path, exist_ok=True)
            tower_results = simulation_results.tower_simulation_results[tower_num]
            plot_csv.outputs_to_csv(
                csv_path,
                tower_results,
                self.sim_conds.get_tower(tower_num).common,
            )
    
    def _output_png(
        self,
        output_path: str,
        plot_target_sec: List[int],
        timestamp: float,
    ) -> None:
        """PNG出力"""
        self.logger.info("png出力開始")
        
        for tower_num in range(1, self.num_towers + 1):
            plot_csv.plot_csv_outputs(
                tgt_foldapath=output_path,
                df_obs=self.df_obs,
                tgt_sections=plot_target_sec,
                tower_num=tower_num,
                timestamp=timestamp,
                df_p_end=self.df_operation,
            )
    
    def _apply_unit_conversion(self, simulation_results: SimulationResults) -> None:
        """単位変換の適用"""
        for tower_num in range(1, self.num_towers + 1):
            tower_results = simulation_results.tower_simulation_results[tower_num]
            time_series_data = tower_results.time_series_data
            
            for i, material_balance_results in enumerate(time_series_data.material):
                pressure_mpa = time_series_data.others[i]["total_pressure"]
                
                for stream_id in range(1, self.num_streams + 1):
                    for section_id in range(1, self.num_sections + 1):
                        material_result = material_balance_results.get_result(
                            stream_id, section_id
                        )
                        heat_result = time_series_data.heat[i].get_result(
                            stream_id, section_id
                        )
                        temperature = heat_result.cell_temperatures.bed_temperature
                        
                        # cm3 → Nm3 変換
                        material_result.inlet_gas.co2_volume = self._convert_cm3_to_nm3(
                            material_result.inlet_gas.co2_volume, pressure_mpa, temperature
                        )
                        material_result.inlet_gas.n2_volume = self._convert_cm3_to_nm3(
                            material_result.inlet_gas.n2_volume, pressure_mpa, temperature
                        )
                        material_result.outlet_gas.co2_volume = self._convert_cm3_to_nm3(
                            material_result.outlet_gas.co2_volume, pressure_mpa, temperature
                        )
                        material_result.outlet_gas.n2_volume = self._convert_cm3_to_nm3(
                            material_result.outlet_gas.n2_volume, pressure_mpa, temperature
                        )
    
    @staticmethod
    def _convert_cm3_to_nm3(
        volume_cm3: float,
        pressure_mpa: float,
        temperature: float,
    ) -> float:
        """
        cm3からNm3への変換
        
        Args:
            volume_cm3: 体積 [cm3]
            pressure_mpa: 圧力 [MPa]
            temperature: 温度 [℃]
        
        Returns:
            float: 標準状態での体積 [Nm3]
        """
        STANDARD_PRESSURE_PA = 101325
        STANDARD_TEMPERATURE_K = 273.15
        
        pressure_pa = pressure_mpa * 1e6
        volume_ncm3 = (
            volume_cm3
            * (pressure_pa / STANDARD_PRESSURE_PA)
            * (STANDARD_TEMPERATURE_K / (temperature + 273.15))
        )
        return volume_ncm3 * 1e-6
