from copy import deepcopy
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


import numpy as np
import pandas as pd
from logging import getLogger

from utils import const, plot_csv, other_utils
from .physics import operation_models
from .state import StateVariables
from .simulation_results import SimulationResults
from config.sim_conditions import SimulationConditions, TowerConditions


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
        # ロガーの作成
        # set_logger(log_dir=const.OUTPUT_DIR + cond_id + "/")
        self.logger = getLogger(__name__)

        # クラス変数初期化
        self.cond_id = cond_id

        # 実験条件(conditions)の読み込み
        self.sim_conds = SimulationConditions(self.cond_id)
        self.num_tower = self.sim_conds.num_towers
        self.dt = self.sim_conds.get_tower(1).common.calculation_step_time
        self.num_str = self.sim_conds.get_tower(1).common.num_streams  # ストリーム数
        self.num_sec = self.sim_conds.get_tower(1).common.num_sections  # セクション数

        # 観測値(data)の読み込み
        filepath = const.DATA_DIR + "3塔データ.csv"
        if filepath[-3:] == "csv":
            self.df_obs = pd.read_csv(filepath, index_col=0)
        else:
            self.df_obs = pd.read_excel(filepath, sheet_name=self.sim_conds[1]["sheet_name"], index_col="time")
        self.df_obs = other_utils.resample_obs_data(self.df_obs, self.dt)  # リサンプリング

        # 稼働表の読み込み
        filepath = const.CONDITIONS_DIR + self.cond_id + "/" + "稼働工程表.xlsx"
        self.df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")

        # 塔内の残留ガス情報
        self.residual_gas_composition: dict | None = None

        self.state_manager = StateVariables(self.num_tower, self.num_str, self.num_sec, self.sim_conds)

    def _init_variables(self):
        """各塔の状態変数を初期化"""
        for tower_num in range(1, self.num_tower + 1):
            self.state_manager.towers[tower_num].total_press = self.df_obs.loc[0, f"T{tower_num}_press"]

    def execute_simulation(self, filtered_states=None, output_folderpath=None):
        """物理計算を通しで実行"""
        ### ◆(1/4) 前準備 ------------------------------------------------
        # 記録用配列の用意
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

        ### ◆(2/4) シミュレーション実行 --------------------------------------
        self.logger.info("(1/3) simulation...")
        # プロセス終了時刻記録用
        process_completion_log = {key: 0 for key in range(1, 1 + len(self.df_operation))}
        # 状態変数の初期化
        self._init_variables()
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
            self.logger.info(f"プロセス {process_index}: done. timestamp: {round(timestamp, 2)}")
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
        # 初回限定処理の実施
        if "バッチ吸着_上流" in mode_list:
            upstream_tower_number = mode_list.index("バッチ吸着_上流") + 1
            downstream_tower_number = mode_list.index("バッチ吸着_下流") + 1
            upstream_tower_state = self.state_manager.towers[upstream_tower_number]
            downstream_tower_state = self.state_manager.towers[downstream_tower_number]
            # NOTE: 圧力平均化の位置
            # 圧力の平均化
            total_press_mean = (
                upstream_tower_state.total_press
                * self.sim_conds.get_tower(upstream_tower_number).packed_bed.void_volume
                + downstream_tower_state.total_press
                * self.sim_conds.get_tower(downstream_tower_number).packed_bed.void_volume
            ) / (
                self.sim_conds.get_tower(upstream_tower_number).packed_bed.void_volume
                + self.sim_conds.get_tower(downstream_tower_number).packed_bed.void_volume
            )
            upstream_tower_state.total_press = total_press_mean
            downstream_tower_state.total_press = total_press_mean
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
        # 記録用
        record_outputs_tower = {}  # 可視化等記録用

        ### 各塔のガス吸着計算 ---------------------------------
        # NOTE: 上流があれば優先し、下流へマテバラ結果を渡す
        # NOTE: 均圧があれば減圧を優先し、加圧へ変数を渡す
        # 1. 上流と下流がある場合
        up_and_down_mode_list = [
            ["流通吸着_単独/上流", "流通吸着_下流"],  # [上流, 下流]
            ["バッチ吸着_上流", "バッチ吸着_下流"],
        ]
        has_flow_adsorption_pair = (up_and_down_mode_list[0][0] in mode_list) and (
            up_and_down_mode_list[0][1] in mode_list
        )
        has_batch_adsorption_pair = (up_and_down_mode_list[1][0] in mode_list) and (
            up_and_down_mode_list[1][1] in mode_list
        )
        # 3塔のうち2塔で上流・下流の組み合わせがある場合
        if has_flow_adsorption_pair | has_batch_adsorption_pair:
            if has_flow_adsorption_pair:
                upstream_mode = up_and_down_mode_list[0][0]
                downstream_mode = up_and_down_mode_list[0][1]
            else:
                upstream_mode = up_and_down_mode_list[1][0]
                downstream_mode = up_and_down_mode_list[1][1]
            upstream_tower_num = mode_list.index(upstream_mode) + 1
            downstream_tower_num = mode_list.index(downstream_mode) + 1
            # 上流塔のガス吸着計算
            (
                record_outputs_tower[upstream_tower_num],
                _,
            ) = self.branch_operation_mode(
                tower_conds=sim_conds.get_tower(upstream_tower_num),
                mode=upstream_mode,
                tower_num=upstream_tower_num,
                state_manager=self.state_manager,
            )
            # 下流塔の計算
            (
                record_outputs_tower[downstream_tower_num],
                _,
            ) = self.branch_operation_mode(
                tower_conds=sim_conds.get_tower(downstream_tower_num),
                mode=downstream_mode,
                tower_num=downstream_tower_num,
                state_manager=self.state_manager,
                other_tower_params=record_outputs_tower[upstream_tower_num]["material"],
            )
            # 残りの塔
            for current_tower_num in range(1, 1 + self.num_tower):
                # 上流・下流は計算済みなのでスキップ
                if current_tower_num in [upstream_tower_num, downstream_tower_num]:
                    continue
                current_mode = mode_list[current_tower_num - 1]
                (
                    record_outputs_tower[current_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    tower_conds=sim_conds.get_tower(current_tower_num),
                    mode=current_mode,
                    tower_num=current_tower_num,
                    state_manager=self.state_manager,
                )
        # 2. 均圧の加圧と減圧がある場合
        elif ("均圧_加圧" in mode_list) and ("均圧_減圧" in mode_list):
            # 減圧と加圧の塔番号取得
            depressurization_mode = "均圧_減圧"
            depressurization_tower_num = mode_list.index(depressurization_mode) + 1
            pressurization_mode = "均圧_加圧"
            pressurization_tower_num = mode_list.index(pressurization_mode) + 1
            # 減圧から実施
            # NOTE: 加圧側の全圧を引数として渡す
            pressurization_tower_pressure = self.state_manager.towers[pressurization_tower_num].total_press
            (
                record_outputs_tower[depressurization_tower_num],
                all_outputs,
            ) = self.branch_operation_mode(
                tower_conds=sim_conds.get_tower(depressurization_tower_num),
                mode=depressurization_mode,
                tower_num=depressurization_tower_num,
                state_manager=self.state_manager,
                other_tower_params=pressurization_tower_pressure,
            )
            # 加圧
            # NOTE: 減圧側の均圧配管流量を引数として渡す
            (
                record_outputs_tower[pressurization_tower_num],
                _,
            ) = self.branch_operation_mode(
                tower_conds=sim_conds.get_tower(pressurization_tower_num),
                mode=pressurization_mode,
                tower_num=pressurization_tower_num,
                state_manager=self.state_manager,
                other_tower_params=all_outputs["downflow_params"],
            )
            # 残りの塔
            for current_tower_num in range(1, 1 + self.num_tower):
                # 加圧・減圧はスキップ
                if current_tower_num in [depressurization_tower_num, pressurization_tower_num]:
                    continue
                current_mode = mode_list[current_tower_num - 1]
                (
                    record_outputs_tower[current_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    tower_conds=sim_conds.get_tower(current_tower_num),
                    mode=current_mode,
                    tower_num=current_tower_num,
                    state_manager=self.state_manager,
                )
        # 3. 独立運転
        else:
            for current_tower_num in range(1, 1 + self.num_tower):
                current_mode = mode_list[current_tower_num - 1]
                (
                    record_outputs_tower[current_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    tower_conds=sim_conds.get_tower(current_tower_num),
                    mode=current_mode,
                    tower_num=current_tower_num,
                    state_manager=self.state_manager,
                )

        return record_outputs_tower

    def branch_operation_mode(
        self,
        tower_conds: TowerConditions,
        mode: str,
        tower_num: int,
        state_manager: StateVariables,
        other_tower_params=None,
    ):
        """稼働モードxの時のガス吸着計算を行う

        Args:
            mode (int): 稼働モード
            variables (dict): 状態変数
            timestamp (float): 現在時刻t
            other_tower_params(dict): 他の塔の出力や状態変数など

        Returns:
            dict: 更新後の状態変数
            dict: 記録用の計算結果
            dict: 全計算結果
        """
        ### 1. ガス吸着計算 --------------------------------------------
        if mode == "初回ガス導入":
            # 0. 前準備
            tower_conds_copy = deepcopy(tower_conds)
            tower_conds_copy.feed_gas.co2_flow_rate = 20
            tower_conds_copy.feed_gas.n2_flow_rate = 25.2
            calc_output = operation_models.initial_adsorption(
                tower_conds=tower_conds_copy, state_manager=state_manager, tower_num=tower_num
            )
        elif mode == "停止":
            calc_output = operation_models.stop_mode(
                tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
            )
        elif mode == "流通吸着_単独/上流":
            calc_output = operation_models.flow_adsorption_single_or_upstream(
                tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
            )
        elif mode == "流通吸着_下流":
            calc_output = operation_models.flow_adsorption_downstream(
                tower_conds=tower_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                inflow_gas=other_tower_params,
            )
        elif mode == "バッチ吸着_上流":
            calc_output = operation_models.batch_adsorption_upstream(
                tower_conds=tower_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                is_series_operation=True,
            )
        elif mode == "バッチ吸着_下流":
            if self.residual_gas_composition is None:
                self.logger.warning("residual_gas_composition計算前にバッチ吸着_下流が呼ばれました")
            calc_output = operation_models.batch_adsorption_downstream(
                tower_conds=tower_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                is_series_operation=True,
                inflow_gas=other_tower_params,
                residual_gas_composition=self.residual_gas_composition,
            )
        elif mode == "均圧_減圧":
            calc_output = operation_models.equalization_pressure_depressurization(
                tower_conds=tower_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                downstream_tower_pressure=other_tower_params,
            )
        elif mode == "均圧_加圧":
            calc_output = operation_models.equalization_pressure_pressurization(
                tower_conds=tower_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                upstream_params=other_tower_params,
            )
            self.residual_gas_composition = calc_output["material"]
        elif mode == "真空脱着":
            calc_output = operation_models.desorption_by_vacuuming(
                tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
            )

        ### 2. 状態変数の更新 ----------------------------------------
        self.state_manager.update_from_calc_output(tower_num, mode, calc_output)

        ### 3. 記録項目の抽出 ----------------------------------------
        # a. マテバラ・熱バラ
        key_list = ["material", "heat", "heat_wall", "heat_lid"]
        other_outputs = {key: calc_output[key] for key in key_list}
        # b. その他状態変数
        tower = self.state_manager.towers[tower_num]

        other_outputs["others"] = {
            "total_pressure": tower.total_press,
            "co2_mole_fraction": tower.co2_mole_fraction.copy(),
            "n2_mole_fraction": tower.n2_mole_fraction.copy(),
            "cumulative_co2_recovered": tower.cumulative_co2_recovered,
            "cumulative_n2_recovered": tower.cumulative_n2_recovered,
        }

        return other_outputs, calc_output

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
        self.logger.info("(2/3) csv output...")

        # cm³からNm³への単位変換を適用
        self.logger.info("Converting cm³ to Nm³ for gas flow rates (inlet/outlet CO2 and N2 volumes)...")
        self._apply_unit_conversion_to_simulation_results(simulation_results)
        self.logger.info("Unit conversion completed.")

        for tower_num in range(1, 1 + self.num_tower):
            _tgt_foldapath = output_folderpath + f"/csv/tower_{tower_num}/"
            os.makedirs(_tgt_foldapath, exist_ok=True)
            tower_results = simulation_results.tower_simulation_results[tower_num]
            plot_csv.outputs_to_csv(_tgt_foldapath, tower_results, self.sim_conds.get_tower(tower_num).common)

        self.df_operation["終了時刻(min)"] = list(process_completion_log.values())
        self.df_operation.to_csv(output_folderpath + "/プロセス終了時刻.csv", encoding="shift-jis")

        self.logger.info("(3/3) png output...")
        plot_target_sec = [2, 10, 18]

        for tower_num in range(1, 1 + self.num_tower):
            plot_csv.plot_csv_outputs(
                tgt_foldapath=output_folderpath,
                df_obs=self.df_obs,
                tgt_sections=plot_target_sec,
                tower_num=tower_num,
                timestamp=timestamp,
                df_p_end=self.df_operation,
            )

    def _convert_cm3_to_nm3(self, volume_cm3: float, pressure_mpa: float, temperature_k: float) -> float:
        """
        cm³からNm³（標準状態での体積）に変換する

        Args:
            volume_cm3 (float): 体積 [cm³]
            pressure_mpa (float): 圧力 [MPa]
            temperature_k (float): 温度 [K]

        Returns:
            float: 標準状態での体積 [Nm³]
        """
        STANDARD_PRESSURE_PA = 101325  # Pa (1 atm)
        STANDARD_TEMPERATURE_K = 273.15  # K (0℃)

        pressure_pa = pressure_mpa * 1e6
        volume_nm3 = volume_cm3 * (pressure_pa / STANDARD_PRESSURE_PA) * (STANDARD_TEMPERATURE_K / temperature_k)

        return volume_nm3 * 1e-6

    def _apply_unit_conversion_to_simulation_results(self, simulation_results: SimulationResults) -> None:
        """
        記録されたデータに対して単位変換を適用する
        - cm³ → Nm³ の変換を対象フィールドに適用
        - 各stream/sectionの実際の温度・圧力条件を使用

        Args:
            simulation_results: シミュレーション結果
        """
        self.logger.info("Starting unit conversion (cm³ → Nm³) for gas flow rates...")

        for tower_num in range(1, self.num_tower + 1):
            if tower_num not in simulation_results.tower_simulation_results:
                continue

            tower_results = simulation_results.tower_simulation_results[tower_num]
            time_series = tower_results.time_series_data

            for i, timestamp in enumerate(time_series.timestamps):
                try:
                    tower_pressure_mpa = time_series.others[i]["total_pressure"]
                except (KeyError, IndexError, TypeError) as e:
                    self.logger.warning(
                        f"Unable to get pressure data for unit conversion at tower {tower_num}, time {i}: {e}"
                    )
                    continue

                if i < len(time_series.material) and i < len(time_series.heat):

                    material_data = time_series.material[i]
                    heat_data = time_series.heat[i]

                    for stream in range(1, self.num_str + 1):
                        for section in range(1, self.num_sec + 1):
                            try:
                                material_result = material_data.get_result(stream, section)
                                heat_result = heat_data.get_result(stream, section)

                                # 温度: 各セルの到達温度 [℃] → [K]
                                temperature_k = heat_result.cell_temperatures.bed_temperature + 273.15
                                # 圧力: 塔の全圧 [MPa]
                                pressure_mpa = tower_pressure_mpa

                                # 流入CO2流量の変換 [cm³] → [Nm³]
                                if hasattr(material_result.inlet_gas, "co2_volume"):
                                    original_value = material_result.inlet_gas.co2_volume
                                    material_result.inlet_gas.co2_volume = self._convert_cm3_to_nm3(
                                        original_value, pressure_mpa, temperature_k
                                    )

                                # 流入N2流量の変換 [cm³] → [Nm³]
                                if hasattr(material_result.inlet_gas, "n2_volume"):
                                    original_value = material_result.inlet_gas.n2_volume
                                    material_result.inlet_gas.n2_volume = self._convert_cm3_to_nm3(
                                        original_value, pressure_mpa, temperature_k
                                    )

                                # 下流流出CO2流量の変換 [cm³] → [Nm³]
                                if hasattr(material_result.outlet_gas, "co2_volume"):
                                    original_value = material_result.outlet_gas.co2_volume
                                    material_result.outlet_gas.co2_volume = self._convert_cm3_to_nm3(
                                        original_value, pressure_mpa, temperature_k
                                    )

                                # 下流流出N2流量の変換 [cm³] → [Nm³]
                                if hasattr(material_result.outlet_gas, "n2_volume"):
                                    original_value = material_result.outlet_gas.n2_volume
                                    material_result.outlet_gas.n2_volume = self._convert_cm3_to_nm3(
                                        original_value, pressure_mpa, temperature_k
                                    )

                            except (AttributeError, IndexError, KeyError) as e:
                                self.logger.debug(
                                    f"Skipping unit conversion for tower {tower_num}, stream {stream}, "
                                    f"section {section}, time {i}: {e}"
                                )
                                continue
