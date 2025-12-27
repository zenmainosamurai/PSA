# TODO: ビジネスロジックとログ処理の分離

from copy import deepcopy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


import numpy as np
import pandas as pd

from utils import const, plot_csv, plot_xlsx, other_utils
from .state import StateVariables
from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
from .simulation_results import SimulationResults
from config.sim_conditions import SimulationConditions, TowerConditions
import log


import warnings

warnings.simplefilter("ignore")


@dataclass
class ProcessResults:
    """プロセス毎の結果を保持するデータクラス"""

    timestamp: float
    simulation_results: SimulationResults
    success: bool = True
    error_message: Optional[str] = None


class GasAdosorptionBreakthroughsimulator:
    """ガス吸着モデル(バッチプロセス)を実行するクラス"""

    def __init__(self, cond_id: str):
        """初期化関数

        Args:
            cond_id (str): 実験条件の名前(ex. test1)
        """
        # 子loggerを使用（規約に従った実装）
        self.logger = log.logger.getChild(__name__)

        # クラス変数初期化
        self.cond_id = cond_id

        # 実験条件(conditions)の読み込み
        self.sim_conds = SimulationConditions(self.cond_id)
        self.num_tower = self.sim_conds.num_towers
        self.dt = self.sim_conds.get_tower(1).common.calculation_step_time
        self.num_str = self.sim_conds.get_tower(1).common.num_streams  # ストリーム数
        self.num_sec = self.sim_conds.get_tower(1).common.num_sections  # セクション数

        # 観測値(data)の読み込み（存在しない場合は警告を出して継続）
        filepath = const.DATA_DIR + "3塔データ.csv"
        self.df_obs: Optional[pd.DataFrame] = None
        try:
            if filepath.lower().endswith("csv"):
                df = pd.read_csv(filepath, index_col=0)
            else:
                df = pd.read_excel(filepath, sheet_name=self.sim_conds[1]["sheet_name"], index_col="time")

            # リサンプリング
            self.df_obs = other_utils.resample_obs_data(df, self.dt)
        except FileNotFoundError:
            self.logger.warning(f"観測データファイルが存在しないため比較をスキップします: {filepath}")
        except Exception as exc:
            self.logger.error(f"観測データの読み込みに失敗しましたが処理を継続します: {exc}")

        # 稼働表の読み込み
        filepath = const.CONDITIONS_DIR + self.cond_id + "/" + "稼働工程表.xlsx"
        self.df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")

        # 塔内の残留ガス情報
        self.residual_gas_composition: dict | None = None

        self.state_manager = StateVariables(self.num_tower, self.num_str, self.num_sec, self.sim_conds)

    def execute_simulation(self, filtered_states=None, output_folderpath=None):
        """物理計算を通しで実行"""
        simulation_results = SimulationResults()

        # 塔の初期化
        for tower_num in range(1, 1 + self.num_tower):
            simulation_results.initialize_tower(tower_num)

        # 出力先フォルダの用意
        if filtered_states is None:
            # シミュレーションの場合
            output_folderpath = const.OUTPUT_DIR + f"{self.cond_id}/"
            os.makedirs(output_folderpath, exist_ok=True)
        else:
            # データ同化の場合
            output_folderpath = output_folderpath
            os.makedirs(output_folderpath, exist_ok=True)

        # プロセス終了時刻記録用
        process_completion_log = {key: 0 for key in range(1, 1 + len(self.df_operation))}
        # 吸着計算
        timestamp = 0
        simulation_success = True
        error_message = None
        for process_index in self.df_operation.index:
            mode_list = list(self.df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
            termination_cond = self.df_operation.loc[process_index, "終了条件"]

            process_result = self._execute_process(
                process_index=process_index,
                mode_list=mode_list,
                termination_cond_str=termination_cond,
                simulation_results=simulation_results,
                timestamp=timestamp,
                filtered_x=filtered_states,
            )
            timestamp = process_result.timestamp
            simulation_results = process_result.simulation_results
            process_completion_log[process_index] = round(timestamp, 2)
            # 処理が中断された場合、次のステップに移行
            if not process_result.success:
                self.logger.warning(f"工程 {process_index} でエラーが発生したため、後続処理をスキップします")
                break
            self.logger.info(f"プロセス{process_index}終了 timestamp: {round(timestamp, 2)}")
        if simulation_success:
            self._output_results(output_folderpath, simulation_results, process_completion_log, timestamp)

    def _execute_process(
        self,
        process_index: int,
        mode_list: List[str],
        termination_cond_str: str,
        simulation_results: SimulationResults,
        timestamp: float,
        filtered_x=None,
    ) -> ProcessResults:
        timestamp_result, simulation_results_result, success = self.calc_adsorption_process(
            mode_list=mode_list,
            termination_cond_str=termination_cond_str,
            simulation_results=simulation_results,
            timestamp=timestamp,
            filtered_x=filtered_x,
        )

        return ProcessResults(
            timestamp=timestamp_result,
            simulation_results=simulation_results_result,
            success=success,
            error_message=None if success else "プロセス実行中にエラーが発生",
        )

    def calc_adsorption_process(
        self,
        mode_list: List[str],
        termination_cond_str: str,
        simulation_results: SimulationResults,
        timestamp,
        filtered_x=None,
    ):
        """プロセスpの各塔のガス吸着計算を行う

        Args:
            process (inf): プロセス番号p
            mode_list (list): 各塔の稼働モード
            termination_cond (str): プロセスの終了条件
            simulation_results (SimulationResults): 計算結果の記録用
            timestamp (float): 時刻t
            filtered_x (pd.DataFrame): データ同化で得られた状態変数の推移
        """
        # プロセス開始後経過時間
        timestamp_p = 0
        # 初回限定処理の実施（バッチ吸着の圧力平均化）
        prepare_batch_adsorption_pressure(self.state_manager, self.sim_conds, mode_list)
        # 終了条件の抽出
        termination_cond = self._create_termination_cond(
            termination_cond_str,
            timestamp,
            timestamp_p,
        )
        # 逐次吸着計算
        while termination_cond:
            # 各塔の吸着計算実施
            _record_outputs_tower = self.calc_adsorption_mode_list(self.sim_conds, mode_list)
            # timestamp_p更新
            timestamp_p += self.dt
            # 記録
            for tower_num in range(1, 1 + self.num_tower):
                current_timestamp = timestamp + timestamp_p
                tower_outputs = _record_outputs_tower[tower_num]

                simulation_results.add_tower_result(
                    tower_id=tower_num,
                    timestamp=current_timestamp,
                    material=tower_outputs["material"],
                    heat=tower_outputs["heat"],
                    heat_wall=tower_outputs["heat_wall"],
                    heat_lid=tower_outputs["heat_lid"],
                    others=tower_outputs["others"],
                )
            # 終了条件の更新
            termination_cond = self._create_termination_cond(
                termination_cond_str,
                timestamp,
                timestamp_p,
            )
            # 時間超過による強制終了
            time_threshold = 20
            if timestamp_p >= time_threshold:
                self.logger.warning(f"{time_threshold}分以内に終了しなかったため強制終了")
                return timestamp + timestamp_p, simulation_results, False
        return timestamp + timestamp_p, simulation_results, True

    def calc_adsorption_mode_list(self, sim_conds: SimulationConditions, mode_list: List[str]):
        """モード(x_1, x_2, ... x_n)の時の各塔のガス吸着計算を行う
            上流や減圧は優先するなど、計算する順番を制御する

        Args:
            mode_list (list): [塔1のモード, 塔2のモード, 塔3のモード]
            variables_tower (dict): 各塔の状態変数

        Returns:
            dict: 更新後の各塔の状態変数
            dict: 記録用の計算結果
        """
        # 新コード（process/process_executor.py）を使用
        outputs, new_residual = execute_mode_list(
            sim_conds=sim_conds,
            mode_list=mode_list,
            state_manager=self.state_manager,
            residual_gas_composition=self.residual_gas_composition,
        )
        
        # residual_gas_compositionを更新（均圧加圧後に設定される）
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

    # TODO: utils/termination_conditions.pyに移動
    def _create_termination_cond(self, termination_cond_str: str, timestamp: float, timestamp_p: float) -> bool:
        """文字列の終了条件からブール値の終了条件を作成する

        Args:
            termination_cond_str (_type_): _description_
        """
        cond_list = termination_cond_str.split("_")
        if cond_list[0] == "圧力到達":
            tower_num = int(cond_list[1][-1])  # 塔番号
            target_press = float(cond_list[2])  # 目標圧力
            return self.state_manager.towers[tower_num].total_press < target_press

        elif cond_list[0] == "温度到達":
            tower_num = int(cond_list[1][-1])  # 塔番号
            target_temp = float(cond_list[2])  # 目標温度
            target_section = self.num_sec  # 温度測定するセクション
            temp_now = np.mean(self.state_manager.towers[tower_num].temp[:, target_section - 1])
            return temp_now < target_temp

        elif cond_list[0] == "時間経過":
            time = float(cond_list[1])  # 目標経過時間
            # 単位変換（minに合わせる）
            unit = cond_list[2]  # 単位
            if unit == "s":
                time /= 60
            return timestamp_p < time

        elif cond_list[0] == "時間到達":
            time = float(cond_list[1])  # 目標到達時間
            return timestamp + timestamp_p < time

    def _output_results(
        self,
        output_folderpath: str,
        simulation_results: SimulationResults,
        process_completion_log: Dict,
        timestamp: float,
    ) -> None:
        """計算結果の出力処理"""
        # use_xlsxフラグを取得（塔1の共通設定から取得）
        use_xlsx = self.sim_conds.get_tower(1).common.use_xlsx

        self._apply_unit_conversion_to_results(simulation_results)
        plot_target_sec = self.sim_conds.get_tower(1).common.get_sections_for_graph()

        if use_xlsx == 1:
            self.logger.info("xlsx出力開始")

            for tower_num in range(1, 1 + self.num_tower):
                _tgt_foldapath = output_folderpath + f"/xlsx/tower_{tower_num}/"
                os.makedirs(_tgt_foldapath, exist_ok=True)
                tower_results = simulation_results.tower_simulation_results[tower_num]
                plot_xlsx.outputs_to_xlsx(
                    _tgt_foldapath,
                    tower_results,
                    self.sim_conds.get_tower(tower_num).common,
                    plot_target_sec,
                    self.df_obs,
                    tower_num,
                )

            self.logger.info("xlsxグラフ出力開始")

            for tower_num in range(1, 1 + self.num_tower):
                plot_xlsx.plot_xlsx_outputs(
                    tgt_foldapath=output_folderpath,
                    df_obs=self.df_obs,
                    tgt_sections=plot_target_sec,
                    tower_num=tower_num,
                    timestamp=timestamp,
                    df_p_end=self.df_operation,
                )

        self.logger.info("csv出力開始")

        for tower_num in range(1, 1 + self.num_tower):
            _tgt_foldapath = output_folderpath + f"/csv/tower_{tower_num}/"
            os.makedirs(_tgt_foldapath, exist_ok=True)
            tower_results = simulation_results.tower_simulation_results[tower_num]
            plot_csv.outputs_to_csv(_tgt_foldapath, tower_results, self.sim_conds.get_tower(tower_num).common)

        self.df_operation["終了時刻(min)"] = list(process_completion_log.values())
        self.df_operation.to_csv(output_folderpath + "/プロセス終了時刻.csv", encoding="shift-jis")

        self.logger.info("png出力開始")

        for tower_num in range(1, 1 + self.num_tower):
            plot_csv.plot_csv_outputs(
                tgt_foldapath=output_folderpath,
                df_obs=self.df_obs,
                tgt_sections=plot_target_sec,
                tower_num=tower_num,
                timestamp=timestamp,
                df_p_end=self.df_operation,
            )

    # TODO: utils/unit_converter.pyに移動
    def _convert_cm3_to_nm3(self, volume_cm3: float, pressure_mpa: float, temperature: float) -> float:
        """
        単位をcm^3からNm^3（標準状態での体積）に変換する

        Args:
            volume_cm3 (float): 体積 [cm^3]
            pressure_mpa (float): 圧力 [MPa]
            temperature (float): 温度 [℃]

        Returns:
            float: 標準状態での体積 [Nm^3]
        """
        STANDARD_PRESSURE_PA = 101325
        STANDARD_TEMPERATURE_K = 273.15

        pressure_pa = pressure_mpa * 1e6
        volume_ncm3 = (
            volume_cm3 * (pressure_pa / STANDARD_PRESSURE_PA) * (STANDARD_TEMPERATURE_K / (temperature + 273.15))
        )
        return volume_ncm3 * 1e-6

    def _apply_unit_conversion_to_results(self, simulation_results: SimulationResults) -> None:
        """
        流入CO2流量、流入N2流量、下流流出CO2流量、下流流出N2流量の単位を"_convert_cm3_to_nm3"を使って変換する
        各時点での圧力・温度を使用して正確な単位変換を行う

        Args:
            simulation_results (SimulationResults): 時系列シミュレーション結果
        """
        for tower_num in range(1, self.num_tower + 1):
            tower_results = simulation_results.tower_simulation_results[tower_num]
            time_series_data = tower_results.time_series_data

            for i, material_balance_results in enumerate(time_series_data.material):
                pressure_mpa = time_series_data.others[i]["total_pressure"]

                for stream_id in range(1, self.num_str + 1):
                    for section_id in range(1, self.num_sec + 1):
                        material_balance_result = material_balance_results.get_result(stream_id, section_id)

                        heat_result = time_series_data.heat[i].get_result(stream_id, section_id)
                        temperature = heat_result.cell_temperatures.bed_temperature

                        material_balance_result.inlet_gas.co2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.inlet_gas.co2_volume, pressure_mpa, temperature
                        )
                        material_balance_result.inlet_gas.n2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.inlet_gas.n2_volume, pressure_mpa, temperature
                        )

                        material_balance_result.outlet_gas.co2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.outlet_gas.co2_volume, pressure_mpa, temperature
                        )
                        material_balance_result.outlet_gas.n2_volume = self._convert_cm3_to_nm3(
                            material_balance_result.outlet_gas.n2_volume, pressure_mpa, temperature
                        )
