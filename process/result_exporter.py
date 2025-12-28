"""結果出力モジュール

PSA担当者向け説明:
シミュレーション結果の出力処理を提供します。

- CSV出力
- PNG出力（グラフ）
- XLSX出力（Excel）

使用例:
    from process.result_exporter import ResultExporter
    
    exporter = ResultExporter(sim_conds)
    exporter.export_all(
        output_dir="output/5_08_mod_logging2/",
        simulation_results=results,
        df_operation=df_operation,
        df_obs=df_obs,
        timestamp=20.4,
    )
"""

import os
from typing import Dict, Optional

import pandas as pd

from config.sim_conditions import SimulationConditions
from process.simulation_results import SimulationResults
from common.constants import STANDARD_PRESSURE
from utils import plot_csv, plot_xlsx
import logger as log


class ResultExporter:
    """結果出力クラス
    
    PSA担当者向け説明:
    シミュレーション結果をCSV、PNG、XLSXファイルに出力します。
    """
    
    def __init__(self, sim_conds: SimulationConditions):
        """
        初期化
        
        Args:
            sim_conds: シミュレーション条件
        """
        self.logger = log.logger.getChild(__name__)
        self.sim_conds = sim_conds
        
        # 基本パラメータ
        self.num_tower = sim_conds.num_towers
        self.num_streams = sim_conds.get_tower(1).common.num_streams
        self.num_sections = sim_conds.get_tower(1).common.num_sections
    
    def export_all(
        self,
        output_dir: str,
        simulation_results: SimulationResults,
        df_operation: pd.DataFrame,
        process_completion_log: Dict[int, float],
        df_obs: Optional[pd.DataFrame] = None,
        timestamp: float = 0.0,
    ) -> None:
        """
        全出力を実行
        
        Args:
            output_dir: 出力先ディレクトリ
            simulation_results: シミュレーション結果
            df_operation: 稼働工程表
            process_completion_log: 各工程の終了時刻
            df_obs: 観測データ（省略可）
            timestamp: 最終タイムスタンプ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 単位変換
        self._apply_unit_conversion(simulation_results)
        
        # グラフ対象セクション
        plot_target_sec = self.sim_conds.get_tower(1).common.get_sections_for_graph()
        
        # 終了時刻をDataFrameに追加
        df_operation_with_time = df_operation.copy()
        df_operation_with_time["終了時刻(min)"] = list(process_completion_log.values())
        
        # XLSX出力
        use_xlsx = self.sim_conds.get_tower(1).common.use_xlsx
        if use_xlsx == 1:
            self.export_xlsx(output_dir, simulation_results, plot_target_sec, df_obs, timestamp, df_operation_with_time)
        
        # CSV出力
        self.export_csv(output_dir, simulation_results)
        
        # プロセス終了時刻出力
        df_operation_with_time.to_csv(
            os.path.join(output_dir, "プロセス終了時刻.csv"),
            encoding="shift-jis",
        )
        
        # PNG出力
        self.export_png(output_dir, df_obs, plot_target_sec, timestamp, df_operation_with_time)
    
    def export_csv(
        self,
        output_dir: str,
        simulation_results: SimulationResults,
    ) -> None:
        """
        CSV出力
        
        Args:
            output_dir: 出力先ディレクトリ
            simulation_results: シミュレーション結果
        """
        self.logger.info("csv出力開始")
        
        for tower_num in range(1, 1 + self.num_tower):
            tower_folder = os.path.join(output_dir, f"csv/tower_{tower_num}/")
            os.makedirs(tower_folder, exist_ok=True)
            
            tower_results = simulation_results.tower_simulation_results[tower_num]
            plot_csv.outputs_to_csv(
                tower_folder,
                tower_results,
                self.sim_conds.get_tower(tower_num).common,
            )
    
    def export_png(
        self,
        output_dir: str,
        df_obs: Optional[pd.DataFrame],
        target_sections: list,
        timestamp: float,
        df_operation: pd.DataFrame,
    ) -> None:
        """
        PNG出力（グラフ）
        
        Args:
            output_dir: 出力先ディレクトリ
            df_obs: 観測データ
            target_sections: 対象セクション
            timestamp: 最終タイムスタンプ
            df_operation: 稼働工程表
        """
        self.logger.info("png出力開始")
        
        for tower_num in range(1, 1 + self.num_tower):
            plot_csv.plot_csv_outputs(
                output_dir=output_dir,
                df_obs=df_obs,
                target_sections=target_sections,
                tower_num=tower_num,
                timestamp=timestamp,
                df_schedule=df_operation,
            )
    
    def export_xlsx(
        self,
        output_dir: str,
        simulation_results: SimulationResults,
        target_sections: list,
        df_obs: Optional[pd.DataFrame],
        timestamp: float,
        df_operation: pd.DataFrame,
    ) -> None:
        """
        XLSX出力
        
        Args:
            output_dir: 出力先ディレクトリ
            simulation_results: シミュレーション結果
            target_sections: 対象セクション
            df_obs: 観測データ
            timestamp: 最終タイムスタンプ
            df_operation: 稼働工程表
        """
        self.logger.info("xlsx出力開始")
        
        for tower_num in range(1, 1 + self.num_tower):
            tower_folder = os.path.join(output_dir, f"xlsx/tower_{tower_num}/")
            os.makedirs(tower_folder, exist_ok=True)
            
            tower_results = simulation_results.tower_simulation_results[tower_num]
            plot_xlsx.outputs_to_xlsx(
                tower_folder,
                tower_results,
                self.sim_conds.get_tower(tower_num).common,
                target_sections,
                df_obs,
                tower_num,
            )
        
        self.logger.info("xlsxグラフ出力開始")
        
        for tower_num in range(1, 1 + self.num_tower):
            plot_xlsx.plot_xlsx_outputs(
                output_dir=output_dir,
                df_obs=df_obs,
                target_sections=target_sections,
                tower_num=tower_num,
                timestamp=timestamp,
                df_schedule=df_operation,
            )
    
    def _apply_unit_conversion(self, simulation_results: SimulationResults) -> None:
        """
        単位変換を適用
        
        PSA担当者向け説明:
        流入CO2流量、流入N2流量、下流流出CO2流量、下流流出N2流量の単位を
        cm³からNm³（標準状態での体積）に変換します。
        """
        for tower_num in range(1, self.num_tower + 1):
            tower_results = simulation_results.tower_simulation_results[tower_num]
            time_series_data = tower_results.time_series_data
            
            for i, material_balance_results in enumerate(time_series_data.material):
                pressure_mpa = time_series_data.others[i]["total_pressure"]
                
                for stream_id in range(1, self.num_streams + 1):
                    for section_id in range(1, self.num_sections + 1):
                        material_balance_result = material_balance_results.get_result(
                            stream_id, section_id
                        )
                        heat_result = time_series_data.heat[i].get_result(
                            stream_id, section_id
                        )
                        temperature = heat_result.cell_temperatures.bed_temperature
                        
                        # 流入ガス
                        material_balance_result.inlet_gas.co2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.inlet_gas.co2_volume,
                            pressure_mpa,
                            temperature,
                        )
                        material_balance_result.inlet_gas.n2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.inlet_gas.n2_volume,
                            pressure_mpa,
                            temperature,
                        )
                        
                        # 流出ガス
                        material_balance_result.outlet_gas.co2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.outlet_gas.co2_volume,
                            pressure_mpa,
                            temperature,
                        )
                        material_balance_result.outlet_gas.n2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.outlet_gas.n2_volume,
                            pressure_mpa,
                            temperature,
                        )
    
    @staticmethod
    def _convert_cm3_to_nm3(
        volume_cm3: float,
        pressure_mpa: float,
        temperature: float,
    ) -> float:
        """
        単位をcm³からNm³（標準状態での体積）に変換
        
        Args:
            volume_cm3: 体積 [cm³]
            pressure_mpa: 圧力 [MPa]
            temperature: 温度 [℃]
            
        Returns:
            float: 標準状態での体積 [Nm³]
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
