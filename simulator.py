from copy import deepcopy
import os

import numpy as np
import pandas as pd
from logging import getLogger

from utils import const, init_functions, plot_csv, other_utils
import operation_models
from state_variables import StateVariables, TowerStateArrays


import warnings

warnings.simplefilter("ignore")


class GasAdosorptionBreakthroughsimulator:
    """ガス吸着モデル(バッチプロセス)を実行するクラス"""

    def __init__(self, cond_id):
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
        df_sim_conds = pd.read_excel(
            const.CONDITIONS_DIR + self.cond_id + "/sim_conds.xlsx",
            sheet_name=[
                "共通",
                "触媒充填層条件",
                "導入ガス条件",
                "容器壁条件",
                "蓋条件",
                "底条件",
                "均圧配管条件",
                "真空引き配管条件",
                "熱電対条件",
            ],
            index_col=1,
        )
        self.sim_conds = init_functions.read_sim_conds(df_sim_conds)
        self.num_tower = 3  # 塔数
        self.dt = self.sim_conds[1]["COMMON_COND"]["calculation_step_time"]  # dt
        self.num_str = self.sim_conds[1]["COMMON_COND"]["num_streams"]  # ストリーム分割数
        self.num_sec = self.sim_conds[1]["COMMON_COND"]["num_sections"]  # セクション分割数

        # stream条件の初期化
        self.stream_conds = {}
        for _tower_num in [1, 2, 3]:
            self.stream_conds[_tower_num] = {}
            for stream in range(1, 1 + self.num_str):
                self.stream_conds[_tower_num][stream] = init_functions.init_stream_conds(  # 各ストリーム
                    self.sim_conds[_tower_num],
                    stream,
                    self.stream_conds[_tower_num],
                )
            self.stream_conds[_tower_num][stream + 1] = init_functions.init_drum_wall_conds(  # 壁面
                self.sim_conds[_tower_num], self.stream_conds[_tower_num]
            )

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

        # その他初期化
        self.stagnant_mf: dict | None = None

        self.state_manager = StateVariables(self.num_tower, self.num_str, self.num_sec, self.sim_conds)

    def _init_variables(self):
        """各塔の状態変数を初期化"""
        for tower_num in range(1, self.num_tower + 1):
            self.state_manager.towers[tower_num].total_press = self.df_obs.loc[0, f"T{tower_num}_press"]

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
            output_foldapath = const.OUTPUT_DIR + f"{self.cond_id}/"
            os.makedirs(output_foldapath, exist_ok=True)
        else:
            # データ同化の場合
            output_foldapath = output_foldapath
            os.makedirs(output_foldapath, exist_ok=True)

        ### ◆(2/4) シミュレーション実行 --------------------------------------
        self.logger.info("(1/3) simulation...")
        # プロセス終了時刻記録用
        p_end_dict = {key: 0 for key in range(1, 1 + len(self.df_operation))}
        # 状態変数の初期化
        self._init_variables()
        # 吸着計算
        timestamp = 0
        for p in self.df_operation.index:
            # 各塔の稼働モード抽出
            mode_list = list(self.df_operation.loc[p, ["塔1", "塔2", "塔3"]])
            # 終了条件(文字列)の抽出
            termination_cond = self.df_operation.loc[p, "終了条件"]
            # プロセスpにおける各塔の吸着計算実施
            timestamp, record_dict, success = self.calc_adsorption_process(
                mode_list=mode_list,
                termination_cond_str=termination_cond,
                record_dict=record_dict,
                timestamp=timestamp,
                filtered_x=filtered_states,
            )
            self.logger.info(f"プロセス {p}: done. timestamp: {round(timestamp,2)}")
            # プロセス終了時刻の記録
            p_end_dict[p] = round(timestamp, 2)
            # 処理が中断された場合、次のステップに移行
            if not success:
                self.logger.warning(f"工程 {p} でエラーが発生したため、後続処理をスキップします")
                break

        ### ◆(3/4) csv出力 -------------------------------------------------
        self.logger.info("(2/3) csv output...")
        # 計算結果
        for _tower_num in range(1, 1 + self.num_tower):
            _tgt_foldapath = output_foldapath + f"/csv/tower_{_tower_num}/"
            os.makedirs(_tgt_foldapath, exist_ok=True)
            plot_csv.outputs_to_csv(
                _tgt_foldapath,
                record_dict[_tower_num],
                self.sim_conds[_tower_num]["COMMON_COND"],
            )
        # プロセス終了時刻
        _tgt_foldapath = output_foldapath
        self.df_operation["終了時刻(min)"] = p_end_dict.values()
        self.df_operation.to_csv(_tgt_foldapath + "/プロセス終了時刻.csv", encoding="shift-jis")

        ### ◆(4/4) 可視化 -------------------------------------------------
        self.logger.info("(3/3) png output...")
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
        # try:
        # プロセス開始後経過時間
        timestamp_p = 0
        # 初回限定処理の実施
        if "バッチ吸着_上流" in mode_list:
            tower_num_up = mode_list.index("バッチ吸着_上流") + 1
            tower_num_dw = mode_list.index("バッチ吸着_下流") + 1
            tower_up = self.state_manager.towers[tower_num_up]
            tower_dw = self.state_manager.towers[tower_num_dw]
            # NOTE: 圧力平均化の位置
            # 圧力の平均化
            total_press_mean = (
                tower_up.total_press * self.sim_conds[tower_num_up]["PACKED_BED_COND"]["void_volume"]
                + tower_dw.total_press * self.sim_conds[tower_num_dw]["PACKED_BED_COND"]["void_volume"]
            ) / (
                self.sim_conds[tower_num_up]["PACKED_BED_COND"]["void_volume"]
                + self.sim_conds[tower_num_dw]["PACKED_BED_COND"]["void_volume"]
            )
            tower_up.total_press = total_press_mean
            tower_dw.total_press = total_press_mean
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
            for _tower_num in range(1, 1 + self.num_tower):
                record_dict[_tower_num]["timestamp"].append(timestamp + timestamp_p)
                for key, values in _record_outputs_tower[_tower_num].items():
                    record_dict[_tower_num][key].append(values)
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
                return timestamp + timestamp_p, record_dict, False
        return timestamp + timestamp_p, record_dict, True

        # except Exception as e:
        #     self.logger.warning(f"エラーが発生しました: \n{e}")
        #     return timestamp + timestamp_p, record_dict, False

    def calc_adsorption_mode_list(self, sim_conds, mode_list):
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
        record_outputs_tower = {}  # 可視化等記録用

        ### 各塔のガス吸着計算 ---------------------------------
        # NOTE: 上流があれば優先し、下流へマテバラ結果を渡す
        # NOTE: 均圧があれば減圧を優先し、加圧へ変数を渡す
        # 1. 上流と下流がある場合
        up_and_down_mode_list = [
            ["流通吸着_単独/上流", "流通吸着_下流"],  # [上流, 下流]
            ["バッチ吸着_上流", "バッチ吸着_下流"],
        ]
        cond1 = (up_and_down_mode_list[0][0] in mode_list) and (up_and_down_mode_list[0][1] in mode_list)
        cond2 = (up_and_down_mode_list[1][0] in mode_list) and (up_and_down_mode_list[1][1] in mode_list)
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
                record_outputs_tower[_tgt_tower_num_up],
                _,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_up],
                stream_conds=self.stream_conds[_tgt_tower_num_up],
                mode=_tgt_mode,
                tower_num=_tgt_tower_num_up,
                state_manager=self.state_manager,
            )
            # 下流 (上流のマテバラ出力を使用)
            if cond1:
                _tgt_mode = up_and_down_mode_list[0][1]
            else:
                _tgt_mode = up_and_down_mode_list[1][1]
            _tgt_tower_num_down = mode_list.index(_tgt_mode) + 1
            (
                record_outputs_tower[_tgt_tower_num_down],
                _,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_down],
                stream_conds=self.stream_conds[_tgt_tower_num_down],
                mode=_tgt_mode,
                tower_num=_tgt_tower_num_down,
                state_manager=self.state_manager,
                other_tower_params=record_outputs_tower[_tgt_tower_num_up]["material"],
            )
            # 残りの塔
            for tgt_tower_num in range(1, 1 + self.num_tower):
                # 上流・下流はスキップ
                if tgt_tower_num in [_tgt_tower_num_up, _tgt_tower_num_down]:
                    continue
                _tgt_mode = mode_list[tgt_tower_num - 1]
                (
                    record_outputs_tower[tgt_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    sim_conds=sim_conds[tgt_tower_num],
                    stream_conds=self.stream_conds[tgt_tower_num],
                    mode=_tgt_mode,
                    tower_num=tgt_tower_num,
                    state_manager=self.state_manager,
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
            press_tower_pressure = self.state_manager.towers[_tgt_tower_num_press].total_press
            (
                record_outputs_tower[_tgt_tower_num_depress],
                all_outputs,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_depress],
                stream_conds=self.stream_conds[_tgt_tower_num_depress],
                mode=_tgt_mode_dep,
                tower_num=_tgt_tower_num_depress,
                state_manager=self.state_manager,
                other_tower_params=press_tower_pressure,
            )
            # 加圧
            # NOTE: 減圧側の均圧配管流量を引数として渡す
            (
                record_outputs_tower[_tgt_tower_num_press],
                _,
            ) = self.branch_operation_mode(
                sim_conds=sim_conds[_tgt_tower_num_press],
                stream_conds=self.stream_conds[_tgt_tower_num_press],
                mode=_tgt_mode_pre,
                tower_num=_tgt_tower_num_press,
                state_manager=self.state_manager,
                other_tower_params=all_outputs["downflow_params"],
            )
            # 残りの塔
            for tgt_tower_num in range(1, 1 + self.num_tower):
                # 加圧・減圧はスキップ
                if tgt_tower_num in [_tgt_tower_num_depress, _tgt_tower_num_press]:
                    continue
                _tgt_mode = mode_list[tgt_tower_num - 1]
                (
                    record_outputs_tower[tgt_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    sim_conds=sim_conds[tgt_tower_num],
                    stream_conds=self.stream_conds[tgt_tower_num],
                    mode=_tgt_mode,
                    tower_num=tgt_tower_num,
                    state_manager=self.state_manager,
                )
        # 3. 独立運転
        else:
            # 残りの塔
            for tgt_tower_num in range(1, 1 + self.num_tower):
                _tgt_mode = mode_list[tgt_tower_num - 1]
                (
                    record_outputs_tower[tgt_tower_num],
                    _,
                ) = self.branch_operation_mode(
                    sim_conds=sim_conds[tgt_tower_num],
                    stream_conds=self.stream_conds[tgt_tower_num],
                    mode=_tgt_mode,
                    tower_num=tgt_tower_num,
                    state_manager=self.state_manager,
                )

        return record_outputs_tower

    def branch_operation_mode(self, sim_conds, stream_conds, mode, tower_num, state_manager, other_tower_params=None):
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
            sim_conds_copy = deepcopy(sim_conds)
            sim_conds_copy["FEED_GAS_COND"]["co2_flow_rate"] = 20
            sim_conds_copy["FEED_GAS_COND"]["n2_flow_rate"] = 25.2
            calc_output = operation_models.initial_adsorption(
                sim_conds=sim_conds_copy, stream_conds=stream_conds, state_manager=state_manager, tower_num=tower_num
            )
        # 停止
        elif mode == "停止":
            calc_output = operation_models.stop_mode(
                sim_conds=sim_conds, stream_conds=stream_conds, state_manager=state_manager, tower_num=tower_num
            )
        # 流通吸着_単独/上流
        elif mode == "流通吸着_単独/上流":
            calc_output = operation_models.flow_adsorption_single_or_upstream(
                sim_conds=sim_conds, stream_conds=stream_conds, state_manager=state_manager, tower_num=tower_num
            )
        # 流通吸着_下流
        elif mode == "流通吸着_下流":
            calc_output = operation_models.flow_adsorption_downstream(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                inflow_gas=other_tower_params,
            )
        # バッチ吸着_上流
        elif mode == "バッチ吸着_上流":
            calc_output = operation_models.batch_adsorption_upstream(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                is_series_operation=True,
            )
        # バッチ吸着_下流
        elif mode == "バッチ吸着_下流":
            if self.stagnant_mf is None:
                self.logger.warning("stagnant_mf計算前にバッチ吸着_下流が呼ばれました")
            calc_output = operation_models.batch_adsorption_downstream(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                is_series_operation=True,
                inflow_gas=other_tower_params,
                stagnant_mf=self.stagnant_mf,
            )
        # 均圧_減圧
        elif mode == "均圧_減圧":
            calc_output = operation_models.equalization_pressure_depressurization(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                downstream_tower_pressure=other_tower_params,
            )
        # 均圧_加圧
        elif mode == "均圧_加圧":
            calc_output = operation_models.equalization_pressure_pressurization(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                state_manager=state_manager,
                tower_num=tower_num,
                upstream_params=other_tower_params,
            )
            self.stagnant_mf = calc_output["material"]
        # 真空脱着
        elif mode == "真空脱着":
            calc_output = operation_models.desorption_by_vacuuming(
                sim_conds=sim_conds, stream_conds=stream_conds, state_manager=state_manager, tower_num=tower_num
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
            "co2_mole_fraction": tower.mf_co2.copy(),
            "n2_mole_fraction": tower.mf_n2.copy(),
            "vacuum_amt_co2": tower.vacuum_amt_co2,
            "vacuum_amt_n2": tower.vacuum_amt_n2,
        }

        return other_outputs, calc_output

    def _create_termination_cond(self, termination_cond_str, timestamp, timestamp_p):
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
