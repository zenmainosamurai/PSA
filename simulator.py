import os
import datetime
import yaml
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import CoolProp.CoolProp as CP

from utils import const, init_functions, plot_csv, other_utils
import models
from base_models import _equilibrium_adsorp_amt

import warnings

warnings.simplefilter("ignore")


class GasAdosorption_Breakthrough_simulator:
    """ガス吸着モデル(バッチプロセス)を実行するクラス"""

    def __init__(self, cond_id):
        """初期化関数

        Args:
            cond_id (str): 実験条件の名前(ex. test1)
        """
        # クラス変数初期化
        self.cond_id = cond_id

        # 実験条件(conditions)の読み込み
        df_sim_conds = pd.read_excel(
            const.CONDITIONS_DIR + self.cond_id + "/sim_conds.xlsx",
            sheet_name=["共通", "塔1", "塔2", "塔3"],
        )
        self.sim_conds = init_functions.read_sim_conds(df_sim_conds)
        self.num_tower = 3  # 塔数
        self.num_str = self.sim_conds[1]["NUM_STR"]  # ストリーム分割数
        self.num_sec = self.sim_conds[1]["NUM_SEC"]  # セクション分割数

        # stream条件の初期化
        self.stream_conds = {}
        for _tower_num in [1, 2, 3]:
            self.stream_conds[_tower_num] = {}
            for stream in range(1, 1 + self.num_str):
                self.stream_conds[_tower_num][stream] = (
                    init_functions.init_stream_conds(  # 各ストリーム
                        self.sim_conds[_tower_num],
                        stream,
                        self.stream_conds[_tower_num],
                    )
                )
            self.stream_conds[_tower_num][stream + 1] = (
                init_functions.init_drum_wall_conds(  # 壁面
                    self.sim_conds[_tower_num], self.stream_conds[_tower_num]
                )
            )

        # 観測値(data)の読み込み
        filepath = const.DATA_DIR + "3塔データ.csv"
        if filepath[-3:] == "csv":
            self.df_obs = pd.read_csv(filepath, index_col=0)
        else:
            self.df_obs = pd.read_excel(
                filepath, sheet_name=self.sim_conds[1]["sheet_name"], index_col="time"
            )
        self.df_obs = other_utils.resample_obs_data(
            self.df_obs, self.sim_conds[1]["dt"]
        )  # リサンプリング

        # 稼働表の読み込み
        filepath = const.CONDITIONS_DIR + self.cond_id + "/" + "稼働工程表.xlsx"
        self.df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")

        # その他初期化
        # NOTE: test記録用
        self.record_test = []
        self.record_test2 = []

    def _init_variables(self):
        """各塔の状態変数を初期化

        Returns:
            dict: 初期化した状態変数
        """
        variables_tower = {}
        # 各塔ごとに初期化
        # HINT: variables_tower[i] ... i塔の状態変数
        for _tower_num in range(1, 1 + self.num_tower):
            variables_tower[_tower_num] = {}
            # 各セルの温度
            variables_tower[_tower_num]["temp"] = {}
            for stream in range(1, 1 + self.num_str):
                variables_tower[_tower_num]["temp"][stream] = {}
                for section in range(1, 1 + self.num_sec):
                    variables_tower[_tower_num]["temp"][stream][section] = (
                        self.sim_conds[_tower_num]["DRUM_WALL_COND"]["temp_outside"]
                    )
            # 壁面温度
            variables_tower[_tower_num]["temp_wall"] = {}
            for section in range(1, 1 + self.num_sec):
                variables_tower[_tower_num]["temp_wall"][section] = self.sim_conds[
                    _tower_num
                ]["DRUM_WALL_COND"]["temp_outside"]
            # 上下蓋の温度
            variables_tower[_tower_num]["temp_lid"] = {}
            for position in ["up", "down"]:
                variables_tower[_tower_num]["temp_lid"][position] = self.sim_conds[
                    _tower_num
                ]["DRUM_WALL_COND"]["temp_outside"]
            # 各セルの熱電対温度
            variables_tower[_tower_num]["temp_thermo"] = {}
            for stream in range(1, 1 + self.num_str):
                variables_tower[_tower_num]["temp_thermo"][stream] = {}
                for section in range(1, 1 + self.num_sec):
                    variables_tower[_tower_num]["temp_thermo"][stream][section] = (
                        self.sim_conds[_tower_num]["DRUM_WALL_COND"]["temp_outside"]
                    )
            # 吸着量
            variables_tower[_tower_num]["adsorp_amt"] = {}
            for stream in range(1, 1 + self.num_str):
                variables_tower[_tower_num]["adsorp_amt"][stream] = {}
                for section in range(1, 1 + self.num_sec):
                    variables_tower[_tower_num]["adsorp_amt"][stream][section] = (
                        self.sim_conds[_tower_num]["PACKED_BED_COND"]["init_adsorp_amt"]
                    )
            # モル分率
            variables_tower[_tower_num]["mf_co2"] = {}
            variables_tower[_tower_num]["mf_n2"] = {}
            for stream in range(1, 1 + self.num_str):
                variables_tower[_tower_num]["mf_co2"][stream] = {}
                variables_tower[_tower_num]["mf_n2"][stream] = {}
                for section in range(1, 1 + self.num_sec):
                    variables_tower[_tower_num]["mf_co2"][stream][section] = (
                        self.sim_conds[_tower_num]["INFLOW_GAS_COND"]["mf_co2"]
                    )
                    variables_tower[_tower_num]["mf_n2"][stream][section] = (
                        self.sim_conds[_tower_num]["INFLOW_GAS_COND"]["mf_n2"]
                    )
            # 壁-、層伝熱係数
            variables_tower[_tower_num]["heat_t_coef"] = {}  # 層伝熱係数
            variables_tower[_tower_num]["heat_t_coef_wall"] = {}  # 壁-層伝熱係数
            for stream in range(1, 1 + self.num_str):
                variables_tower[_tower_num]["heat_t_coef"][stream] = {}
                variables_tower[_tower_num]["heat_t_coef_wall"][stream] = {}
                for section in range(1, 1 + self.num_sec):
                    variables_tower[_tower_num]["heat_t_coef"][stream][section] = 1e-5
                    variables_tower[_tower_num]["heat_t_coef_wall"][stream][
                        section
                    ] = 14
            # 全圧
            variables_tower[_tower_num]["total_press"] = self.df_obs.loc[
                0, f"T{_tower_num}_press"
            ]
            # CO2, N2回収量 [mol]
            variables_tower[_tower_num]["vacuum_amt_co2"] = 0
            variables_tower[_tower_num]["vacuum_amt_n2"] = 0
            # 流出CO2分圧
            variables_tower[_tower_num]["outflow_pco2"] = {}
            for stream in range(1, 1 + self.num_str):
                variables_tower[_tower_num]["outflow_pco2"][stream] = {}
                for section in range(1, 1 + self.num_sec):
                    variables_tower[_tower_num]["outflow_pco2"][stream][section] = (
                        variables_tower[_tower_num]["total_press"]
                    )

        return variables_tower

    def execute_simulation(self, filtered_states=None, output_foldapath=None):
        """物理計算を通しで実行"""
        ### ◆(1/4) 前準備 ------------------------------------------------
        # 記録用配列の用意
        _record_item_list = ["material", "heat", "heat_wall", "heat_lid", "others"]
        record_dict = {}
        for _tower_num in range(1, 1 + self.num_tower):
            record_dict[_tower_num] = {}
            record_dict[_tower_num]["timestamp"] = []
            for _item in _record_item_list:
                record_dict[_tower_num][_item] = []
        # 出力先フォルダの用意
        if filtered_states is None:
            # シミュレーションの場合
            output_foldapath = const.OUTPUT_DIR + f"{self.cond_id}/simulation/"
            os.makedirs(output_foldapath, exist_ok=True)
        else:
            # データ同化の場合
            output_foldapath = output_foldapath
            os.makedirs(output_foldapath, exist_ok=True)

        ### ◆(2/4) シミュレーション実行 --------------------------------------
        print("(1/3) simulation...")
        # プロセス終了時刻記録用
        p_end_dict = {}
        # a. 状態変数の初期化
        variables_tower = self._init_variables()
        # b. 吸着計算
        timestamp = 0
        for p in self.df_operation.index:
            # 各塔の稼働モード抽出
            mode_list = list(self.df_operation.loc[p, ["塔1", "塔2", "塔3"]])
            # 終了条件(文字列)の抽出
            termination_cond = self.df_operation.loc[p, "終了条件"]
            # プロセスpにおける各塔の吸着計算実施
            timestamp, variables_tower, record_dict = self.calc_adsorption_process(
                mode_list=mode_list,
                termination_cond_str=termination_cond,
                variables_tower=variables_tower,
                record_dict=record_dict,
                timestamp=timestamp,
                filtered_x=filtered_states,
            )
            print(f"プロセス {p}: done. timestamp: {round(timestamp,2)}")
            # プロセス終了時刻の記録
            p_end_dict[p] = timestamp

        ### ◆(3/4) csv出力 -------------------------------------------------
        print("(2/3) csv output...")
        # 計算結果
        for _tower_num in range(1, 1 + self.num_tower):
            _tgt_foldapath = output_foldapath + f"/csv/tower_{_tower_num}/"
            os.makedirs(_tgt_foldapath, exist_ok=True)
            plot_csv.outputs_to_csv(
                _tgt_foldapath, record_dict[_tower_num], self.sim_conds[_tower_num]
            )
        # プロセス終了時刻
        _tgt_foldapath = output_foldapath
        self.df_operation["終了時刻(min)"] = p_end_dict.values()
        self.df_operation.to_csv(
            _tgt_foldapath + "/プロセス終了時刻.csv", encoding="shift-jis"
        )
        # NOTE: test用
        pd.DataFrame(self.record_test, columns=["diff_press [MPaA]"]).to_csv(
            _tgt_foldapath + "/塔23圧力差.csv"
        )  # , encoding="shift-jis")
        pd.DataFrame(self.record_test2, columns=["排気後全圧 [MPaA]"]).to_csv(
            _tgt_foldapath + "/排気後全圧.csv"
        )  # , encoding="shift-jis")

        ### ◆(4/4) 可視化 -------------------------------------------------
        print("(3/3) png output...")
        # 可視化対象のセルを算出
        plot_target_sec = [2, 10, 18]
        # record_dictの可視化
        for _tower_num in range(1, 1 + self.num_tower):
            tgt_foldapath = output_foldapath
            plot_csv.plot_csv_outputs(
                tgt_foldapath=tgt_foldapath,
                df_obs=self.df_obs,
                tgt_sections=plot_target_sec,
                tower_num=_tower_num,
                timestamp=timestamp,
                df_p_end=self.df_operation,
            )

    def calc_adsorption_process(
        self,
        mode_list,
        termination_cond_str,
        variables_tower,
        record_dict,
        timestamp,
        filtered_x=None,
    ):
        """プロセスpの各塔のガス吸着計算を行う

        Args:
            process (inf): プロセス番号p
            mode_list (list): 各塔の稼働モード
            termination_cond (str): プロセスの終了条件
            record_dict (dict): 計算結果の記録用
            timestamp (float): 時刻t
            filtered_x (pd.DataFrame): データ同化で得られた状態変数の推移
        """
        # プロセス開始後経過時間
        timestamp_p = 0
        # 初回限定処理の実施
        if "バッチ吸着_上流" in mode_list:
            tower_num_up = mode_list.index("バッチ吸着_上流") + 1
            tower_num_dw = mode_list.index("バッチ吸着_下流") + 1
            # 圧力の平均化
            total_press_mean = (
                variables_tower[tower_num_up]["total_press"]
                * self.sim_conds[tower_num_up]["PACKED_BED_COND"]["v_space"]
                + variables_tower[tower_num_dw]["total_press"]
                * self.sim_conds[tower_num_dw]["PACKED_BED_COND"]["v_space"]
            ) / (
                self.sim_conds[tower_num_up]["PACKED_BED_COND"]["v_space"]
                + self.sim_conds[tower_num_dw]["PACKED_BED_COND"]["v_space"]
            )
            variables_tower[tower_num_up]["total_press"] = total_press_mean
            variables_tower[tower_num_dw]["total_press"] = total_press_mean
        # 終了条件の抽出
        termination_cond = self._create_termination_cond(
            termination_cond_str,
            variables_tower,
            timestamp,
            timestamp_p,
        )
        # 逐次吸着計算
        while termination_cond:
            # 各塔の吸着計算実施
            variables_tower, _record_outputs_tower = self.calc_adsorption_mode_list(
                self.sim_conds, mode_list, variables_tower
            )
            # timestamp_p更新
            timestamp_p += self.sim_conds[1]["dt"]
            # 記録
            for _tower_num in range(1, 1 + self.num_tower):
                record_dict[_tower_num]["timestamp"].append(timestamp + timestamp_p)
                for key, values in _record_outputs_tower[_tower_num].items():
                    record_dict[_tower_num][key].append(values)
            # 終了条件の更新
            termination_cond = self._create_termination_cond(
                termination_cond_str,
                variables_tower,
                timestamp,
                timestamp_p,
            )
            # 強制終了
            time_threshold = 20
            if timestamp_p >= time_threshold:
                print(f"{time_threshold}分以内に終了しなかったため強制終了")
                break

        return timestamp + timestamp_p, variables_tower, record_dict

    def calc_adsorption_mode_list(self, sim_conds, mode_list, variables_tower):
        """モード(x_1, x_2, ... x_n)の時の各塔のガス吸着計算を行う
            上流や減圧は優先するなど、計算する順番を制御する

        Args:
            mode_list (list): モード番号のリスト
            variables_tower (dict): 各塔の状態変数

        Returns:
            dict: 更新後の各塔の状態変数
            dict: 記録用の計算結果
        """
        # 記録用
        new_variables_tower = {}  # 状態変数
        record_outputs_tower = {}  # 可視化等記録用

        ### 各塔のガス吸着計算 ---------------------------------
        # NOTE: 上流があれば優先し、下流へマテバラ結果を渡す
        # NOTE: 均圧があれば減圧を優先し、加圧へ変数を渡す
        # 1. 上流と下流がある場合
        up_and_down_mode_list = [
            ["流通吸着_単独/上流", "流通吸着_下流"],  # [上流, 下流]
            ["バッチ吸着_上流", "バッチ吸着_下流"],
        ]
        cond1 = (up_and_down_mode_list[0][0] in mode_list) and (
            up_and_down_mode_list[0][1] in mode_list
        )
        cond2 = (up_and_down_mode_list[1][0] in mode_list) and (
            up_and_down_mode_list[1][1] in mode_list
        )
        if cond1 | cond2:
            # 上流から実施
            if cond1:
                # モード抽出
                _tgt_mode = up_and_down_mode_list[0][0]
            else:
                _tgt_mode = up_and_down_mode_list[1][0]
            # 塔番号
            _tgt_tower_num_up = mode_list.index(_tgt_mode) + 1
            # ガス吸着計算実施
            (
                new_variables_tower[_tgt_tower_num_up],
                record_outputs_tower[_tgt_tower_num_up],
                _,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_up],
                stream_conds=self.stream_conds[_tgt_tower_num_up],
                mode=_tgt_mode,
                variables=variables_tower[_tgt_tower_num_up],
            )
            # 下流 (上流のマテバラ出力を使用)
            if cond1:
                _tgt_mode = up_and_down_mode_list[0][1]
            else:
                _tgt_mode = up_and_down_mode_list[1][1]
            _tgt_tower_num_down = mode_list.index(_tgt_mode) + 1
            (
                new_variables_tower[_tgt_tower_num_down],
                record_outputs_tower[_tgt_tower_num_down],
                _,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_down],
                stream_conds=self.stream_conds[_tgt_tower_num_down],
                mode=_tgt_mode,
                variables=variables_tower[_tgt_tower_num_down],
                other_tower_params=record_outputs_tower[_tgt_tower_num_up]["material"],
            )
            # 残りの塔
            for tgt_tower_num in range(1, 1 + self.num_tower):
                # 上流・下流はスキップ
                if tgt_tower_num in [_tgt_tower_num_up, _tgt_tower_num_down]:
                    continue
                _tgt_mode = mode_list[tgt_tower_num - 1]
                (
                    new_variables_tower[tgt_tower_num],
                    record_outputs_tower[tgt_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    sim_conds=sim_conds[tgt_tower_num],
                    stream_conds=self.stream_conds[tgt_tower_num],
                    mode=_tgt_mode,
                    variables=variables_tower[tgt_tower_num],
                )
        # 2. 均圧の加圧と減圧がある場合
        elif ("均圧_加圧" in mode_list) and ("均圧_減圧" in mode_list):
            # 減圧と加圧の塔番号取得
            _tgt_mode_dep = "均圧_減圧"
            _tgt_tower_num_depress = mode_list.index(_tgt_mode_dep) + 1
            _tgt_mode_pre = "均圧_加圧"
            _tgt_tower_num_press = mode_list.index(_tgt_mode_pre) + 1
            # 減圧から実施
            # NOTE: 加圧側の全圧を引数として渡す
            (
                new_variables_tower[_tgt_tower_num_depress],
                record_outputs_tower[_tgt_tower_num_depress],
                all_outputs,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_depress],
                stream_conds=self.stream_conds[_tgt_tower_num_depress],
                mode=_tgt_mode_dep,
                variables=variables_tower[_tgt_tower_num_depress],
                other_tower_params=variables_tower[_tgt_tower_num_press]["total_press"],
            )
            # 加圧
            # NOTE: 減圧側の均圧配管流量を引数として渡す
            (
                new_variables_tower[_tgt_tower_num_press],
                record_outputs_tower[_tgt_tower_num_press],
                _,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_press],
                stream_conds=self.stream_conds[_tgt_tower_num_press],
                mode=_tgt_mode_pre,
                variables=variables_tower[_tgt_tower_num_press],
                other_tower_params=all_outputs["downflow_params"],
            )
            # 残りの塔
            for tgt_tower_num in range(1, 1 + self.num_tower):
                # 加圧・減圧はスキップ
                if tgt_tower_num in [_tgt_tower_num_depress, _tgt_tower_num_press]:
                    continue
                _tgt_mode = mode_list[tgt_tower_num - 1]
                (
                    new_variables_tower[tgt_tower_num],
                    record_outputs_tower[tgt_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    sim_conds=sim_conds[tgt_tower_num],
                    stream_conds=self.stream_conds[tgt_tower_num],
                    mode=_tgt_mode,
                    variables=variables_tower[tgt_tower_num],
                )
        # 3. どちらも含まれない場合
        else:
            # 残りの塔
            for tgt_tower_num in range(1, 1 + self.num_tower):
                _tgt_mode = mode_list[tgt_tower_num - 1]
                (
                    new_variables_tower[tgt_tower_num],
                    record_outputs_tower[tgt_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    sim_conds=sim_conds[tgt_tower_num],
                    stream_conds=self.stream_conds[tgt_tower_num],
                    mode=_tgt_mode,
                    variables=variables_tower[tgt_tower_num],
                )

        return new_variables_tower, record_outputs_tower

    def branch_operation_mode(
        self, sim_conds, stream_conds, mode, variables, other_tower_params=None
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
        # 初回ガス導入
        if mode == "初回ガス導入":
            # 0. 前準備
            sim_conds_copy = sim_conds.copy()
            sim_conds_copy["INFLOW_GAS_COND"]["fr_co2"] = 20
            sim_conds_copy["INFLOW_GAS_COND"]["fr_n2"] = 25.2
            calc_output = models.initial_adsorption(
                sim_conds=sim_conds_copy, stream_conds=stream_conds, variables=variables
            )
        # 停止
        elif mode == "停止":
            calc_output = models.stop_mode(
                sim_conds=sim_conds, stream_conds=stream_conds, variables=variables
            )
        # 流通吸着_単独/上流
        elif mode == "流通吸着_単独/上流":
            calc_output = models.flow_adsorption_single_or_upstream(
                sim_conds=sim_conds, stream_conds=stream_conds, variables=variables
            )
        # 流通吸着_下流
        elif mode == "流通吸着_下流":
            calc_output = models.flow_adsorption_downstream(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                variables=variables,
                inflow_gas=other_tower_params,
            )
        # バッチ吸着_上流
        elif mode == "バッチ吸着_上流":
            calc_output = models.batch_adsorption_upstream(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                variables=variables,
                series=True,
            )
        # バッチ吸着_下流
        elif mode == "バッチ吸着_下流":
            calc_output = models.batch_adsorption_downstream(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                variables=variables,
                series=True,
                inflow_gas=other_tower_params,
                stagnant_mf=self.stagnant_mf,
            )
        # 均圧_減圧
        elif mode == "均圧_減圧":
            calc_output = models.equalization_pressure_depressurization(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                variables=variables,
                downflow_total_press=other_tower_params,
            )
            # NOTE: test用
            self.record_test.append(calc_output["total_press"])
        # 均圧_加圧
        elif mode == "均圧_加圧":
            calc_output = models.equalization_pressure_pressurization(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                variables=variables,
                upstream_params=other_tower_params,
            )
            self.stagnant_mf = calc_output["material"]
        # 真空脱着
        elif mode == "真空脱着":
            calc_output = models.desorption_by_vacuuming(
                sim_conds=sim_conds, stream_conds=stream_conds, variables=variables
            )
            # NOTE: test2用
            self.record_test2.append(
                calc_output["accum_vacuum_amt"]["total_press_after_vacuum"]
            )

        ### 2. 状態変数の抽出 ----------------------------------------
        new_variables = self._extract_state_vars(mode, variables, calc_output)

        ### 3. 記録項目の抽出 ----------------------------------------
        # a. マテバラ・熱バラ
        key_list = ["material", "heat", "heat_wall", "heat_lid"]
        other_outputs = {key: calc_output[key] for key in key_list}
        # b. その他状態変数
        key_list = ["total_press", "mf_co2", "mf_n2", "vacuum_amt_co2", "vacuum_amt_n2"]
        other_outputs["others"] = {key: new_variables[key] for key in key_list}

        return new_variables, other_outputs, calc_output

    def _extract_state_vars(self, mode, variables, calc_output):
        """吸着計算結果から状態変数を抽出する

        Args:
            mode (int): 稼働モード
            variables (dict): 状態変数
            timestamp (float): 現在時刻t

        Returns:
            dict: 更新後の各塔の状態変数
            dict: 記録用の計算結果
        """
        new_variables = {}
        # 温度
        new_variables["temp"] = {}
        for stream in range(1, 1 + self.num_str):
            new_variables["temp"][stream] = {}
            for section in range(1, 1 + self.num_sec):
                new_variables["temp"][stream][section] = calc_output["heat"][stream][
                    section
                ]["temp_reached"]
        # 温度（壁面）
        new_variables["temp_wall"] = {}
        for section in range(1, 1 + self.num_sec):
            new_variables["temp_wall"][section] = calc_output["heat_wall"][section][
                "temp_reached"
            ]
        # 上下蓋の温度
        new_variables["temp_lid"] = {}
        for position in ["up", "down"]:
            new_variables["temp_lid"][position] = calc_output["heat_lid"][position][
                "temp_reached"
            ]
        # 熱電対温度
        new_variables["temp_thermo"] = {}
        for stream in range(1, 1 + self.num_str):
            new_variables["temp_thermo"][stream] = {}
            for section in range(1, 1 + self.num_sec):
                new_variables["temp_thermo"][stream][section] = calc_output["heat"][
                    stream
                ][section]["temp_thermocouple_reached"]
        # 既存吸着量
        new_variables["adsorp_amt"] = {}
        for stream in range(1, 1 + self.num_str):
            new_variables["adsorp_amt"][stream] = {}
            for section in range(1, 1 + self.num_sec):
                new_variables["adsorp_amt"][stream][section] = calc_output["material"][
                    stream
                ][section]["accum_adsorp_amt"]
        # モル分率
        new_variables["mf_co2"] = {}
        new_variables["mf_n2"] = {}
        if mode in [
            "初回ガス導入",
            "流通吸着_単独/上流",
            "バッチ吸着_上流",
            "均圧_加圧",
            "均圧_減圧",
            "バッチ吸着_下流",
            "流通吸着_下流",
        ]:
            for stream in range(1, 1 + self.num_str):
                new_variables["mf_co2"][stream] = {}
                new_variables["mf_n2"][stream] = {}
                for section in range(1, 1 + self.num_sec):  # 吸着時は「流出モル分率」
                    new_variables["mf_co2"][stream][section] = calc_output["material"][
                        stream
                    ][section]["outflow_mf_co2"]
                    new_variables["mf_n2"][stream][section] = calc_output["material"][
                        stream
                    ][section]["outflow_mf_n2"]
        elif mode in ["真空脱着"]:
            for stream in range(1, 1 + self.num_str):
                new_variables["mf_co2"][stream] = {}
                new_variables["mf_n2"][stream] = {}
                for section in range(
                    1, 1 + self.num_sec
                ):  # 脱着時は「脱着後のモル分率」
                    new_variables["mf_co2"][stream][section] = calc_output[
                        "mol_fraction"
                    ][stream][section]["mf_co2_after_vacuum"]
                    new_variables["mf_n2"][stream][section] = calc_output[
                        "mol_fraction"
                    ][stream][section]["mf_n2_after_vacuum"]
        elif mode in ["停止"]:
            for stream in range(1, 1 + self.num_str):
                new_variables["mf_co2"][stream] = {}
                new_variables["mf_n2"][stream] = {}
                for section in range(1, 1 + self.num_sec):  # 停止時は直前のモル分率
                    new_variables["mf_co2"][stream][section] = variables["mf_co2"][
                        stream
                    ][section]
                    new_variables["mf_n2"][stream][section] = variables["mf_n2"][
                        stream
                    ][section]
        # 壁―, 層伝熱係数
        new_variables["heat_t_coef"] = {}
        new_variables["heat_t_coef_wall"] = {}
        for stream in range(1, 1 + self.num_str):
            new_variables["heat_t_coef"][stream] = {}
            new_variables["heat_t_coef_wall"][stream] = {}
            for section in range(1, 1 + self.num_sec):
                new_variables["heat_t_coef"][stream][section] = calc_output["heat"][
                    stream
                ][section]["hw1"]
                new_variables["heat_t_coef_wall"][stream][section] = calc_output[
                    "heat"
                ][stream][section]["u1"]
        # 全圧
        if mode in ["初回ガス導入", "バッチ吸着_上流", "均圧_加圧", "バッチ吸着_下流"]:
            new_variables["total_press"] = calc_output["total_press_after_batch_adsorp"]
        elif mode in ["均圧_減圧", "流通吸着_単独/上流", "流通吸着_下流"]:
            new_variables["total_press"] = calc_output["total_press"]
        elif mode in ["真空脱着"]:
            new_variables["total_press"] = calc_output["total_press_after_desorp"]
        else:
            new_variables["total_press"] = variables["total_press"]
        # CO2, N2回収量 [mol]
        if mode in ["真空脱着"]:
            new_variables["vacuum_amt_co2"] = calc_output["accum_vacuum_amt"][
                "accum_vacuum_amt_co2"
            ]
            new_variables["vacuum_amt_n2"] = calc_output["accum_vacuum_amt"][
                "accum_vacuum_amt_n2"
            ]
        else:
            new_variables["vacuum_amt_co2"] = 0
            new_variables["vacuum_amt_n2"] = 0
        # 流出CO2分圧
        new_variables["outflow_pco2"] = {}
        for stream in range(1, 1 + self.num_str):
            new_variables["outflow_pco2"][stream] = {}
            for section in range(1, 1 + self.num_sec):
                new_variables["outflow_pco2"][stream][section] = calc_output[
                    "material"
                ][stream][section]["outflow_pco2"]

        return new_variables

    def _overwrite_state_vars(self, filtered_x, timestamp):
        """データ同化で得られた状態変数に上書き

        Args:
            filtered_x (pd.DataFrame): 状態変数の推移
            timestamp (float): 現在時刻
        """
        # 現在時刻に近いindexを抽出
        _tgt_index = filtered_x.index[np.abs(filtered_x.index - (timestamp)).argmin()]
        # 得られた状態変数に置換
        # NOTE: マイナスチェックも同時に実施（smoothingの過程でマイナスになる事例あり）
        for _tower_num in range(1, 1 + self.num_tower):
            self.sim_conds[_tower_num]["PACKED_BED_COND"]["ks_adsorp"] = np.max(
                [1e-8, filtered_x.loc[_tgt_index, f"T{_tower_num}_ks_adsorp"]]
            )
            self.sim_conds[_tower_num]["PACKED_BED_COND"]["ks_desorp"] = np.max(
                [1e-8, filtered_x.loc[_tgt_index, f"T{_tower_num}_ks_desorp"]]
            )
            self.sim_conds[_tower_num]["DRUM_WALL_COND"]["coef_hw1"] = np.max(
                [1e-8, filtered_x.loc[_tgt_index, f"T{_tower_num}_coef_hw1"]]
            )
            self.sim_conds[_tower_num]["INFLOW_GAS_COND"]["adsorp_heat_co2"] = np.max(
                [1e-8, filtered_x.loc[_tgt_index, f"T{_tower_num}_adsorp_heat_co2"]]
            )

    def _create_termination_cond(self, termination_cond_str, variables_tower, timestamp, timestamp_p):
        """文字列の終了条件からブール値の終了条件を作成する

        Args:
            termination_cond_str (_type_): _description_
        """
        cond_list = termination_cond_str.split("_")
        if cond_list[0] == "圧力到達":
            tower_num = int(cond_list[1][-1])  # 塔番号
            target_press = float(cond_list[2])  # 目標圧力
            return variables_tower[tower_num]["total_press"] < target_press

        elif cond_list[0] == "温度到達":
            tower_num = int(cond_list[1][-1])  # 塔番号
            target_temp = float(cond_list[2])  # 目標温度
            target_section = self.num_sec # 温度測定するセクション
            temp_now = np.mean(
                [variables_tower[tower_num]["temp"][stream][target_section]
                for stream in range(1, 1 + self.num_str)
                ]
            )
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