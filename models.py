import os
import sys
import datetime
import yaml
import math
import copy
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import optimize
import CoolProp.CoolProp as CP

from utils import const, init_functions, plot_csv, other_utils

import warnings
warnings.simplefilter('error')


class GasAdosorption_Breakthrough_simulator():
    """ ガス吸着モデル(バッチプロセス)を実行するクラス
    """

    def __init__(self, cond_id):
        """ 初期化関数

        Args:
            obs_name (str): 観測値の名前(ex. obs_case1)
            cond_id (str): 実験条件の名前(ex. test1)
        """

        # クラス変数初期化
        self.cond_id = cond_id

        # 実験条件(conditions)の読み込み
        filepath = const.CONDITIONS_DIR + self.cond_id + "/sim_conds.yml"
        with open(filepath, encoding="utf-8") as f:
            self.common_conds = yaml.safe_load(f)

        # 追加の初期化
        self.common_conds = init_functions.add_common_conds(self.common_conds)
        self.num_str = self.common_conds["CELL_SPLIT"]["num_str"]
        self.num_sec = self.common_conds["CELL_SPLIT"]["num_sec"]

        # stream条件の初期化
        self.stream_conds = {}
        for stream in range(1, 1+self.num_str):
            self.stream_conds[stream] = init_functions.init_stream_conds(
                self.common_conds, stream, self.stream_conds
            )
        self.stream_conds[stream+1] = init_functions.init_drum_wall_conds(
            self.common_conds, self.stream_conds
        )

        # 観測値(data)の読み込み
        filepath = const.DATA_DIR + self.common_conds["data_path"]
        if filepath[-3:] == "csv":
            self.df_obs = pd.read_csv(filepath, index_col="time")
        else:
            self.df_obs = pd.read_excel(filepath, sheet_name=self.common_conds["sheet_name"], index_col="time")
        self.df_obs = other_utils.resample_obs_data(self.df_obs, self.common_conds["dt"]) # リサンプリング

        # その他初期化
        self.time_threshold = 1

    def init_variables(self):
        """温度等の状態変数を初期化する

        Returns:
            dict: 初期化した状態変数
        """
        variables = {}
        # 各セルの温度
        variables["temp"] = {}
        for stream in range(1, 1+self.num_str):
            variables["temp"][stream] = {}
            for section in range(1, 1+self.num_sec):
                variables["temp"][stream][section] = \
                    self.common_conds["DRUM_WALL_COND"]["temp_outside"]
        # 壁面温度
        variables["temp_wall"] = {}
        for section in range(1, 1+self.num_sec):
            variables["temp_wall"][section] = self.common_conds["DRUM_WALL_COND"]["temp_outside"]
        # 上下蓋の温度
        variables["temp_lid"] = {}
        for position in ["up", "down"]:
            variables["temp_lid"][position] = self.common_conds["DRUM_WALL_COND"]["temp_outside"]
        # 吸着量
        variables["adsorp_amt"] = {}
        for stream in range(1, 1+self.num_str):
            variables["adsorp_amt"][stream] = {}
            for section in range(1, 1+self.num_sec):
                variables["adsorp_amt"][stream][section] = (
                    (0.0000021*self.common_conds["PACKED_BED_COND"]["vp"]**2-0.0003385
                    *self.common_conds["PACKED_BED_COND"]["vp"]+0.0145345)*(variables["temp"][stream][section]+273.15)**2
                    +(-0.0012701*self.common_conds["PACKED_BED_COND"]["vp"]**2+0.2091781
                    *self.common_conds["PACKED_BED_COND"]["vp"]-9.9261428)*(variables["temp"][stream][section]+273.15)
                    +0.1828984*self.common_conds["PACKED_BED_COND"]["vp"]**2
                    -30.8594655*self.common_conds["PACKED_BED_COND"]["vp"]+1700.5767712
                )
        # 壁-、層伝熱係数
        variables["heat_t_coef"] = {} # 層伝熱係数
        variables["heat_t_coef_wall"] = {} # 壁-層伝熱係数
        for stream in range(1, 1+self.num_str):
            variables["heat_t_coef"][stream] = {}
            variables["heat_t_coef_wall"][stream] = {}
            for section in range(1, 1+self.num_sec):
                variables["heat_t_coef"][stream][section] = 1e-5
                variables["heat_t_coef_wall"][stream][section] = 14
        # 全圧 
        variables["total_press"] = self.common_conds["INFLOW_GAS_COND"]["total_press"]
        # モル分率
        variables["mf_co2"] = self.common_conds["INFLOW_GAS_COND"]["mf_co2"]
        variables["mf_n2"] = self.common_conds["INFLOW_GAS_COND"]["mf_n2"]

        return variables

    def execute_simulation(self, filtered_states=None, output_foldapath=None):
        """ 物理計算を通しで実行
        """

        ### ◆(1/4) 前準備 ------------------------------------------------

        # 初期値用意
        # filepath = const.CONDITIONS_DIR + self.cond_id + "/assim_conds.yml"
        # with open(filepath, encoding="utf-8") as f:
        #     assim_conds = yaml.safe_load(f)
        # state_vars = assim_conds["STATE_VARS"]

        # 記録用dictの用意
        record_dict = {
            "timestamp": [],
            "params": [],
            "heat": [],
            "heat_wall": [],
            "heat_lid": [],
            "material": [],
            "vaccume": [],
        }

        # 出力用フォルダの用意
        if filtered_states is None:
            mode = "simulation"
            output_foldapath = const.OUTPUT_DIR + f"{self.cond_id}/"
            os.makedirs(output_foldapath, exist_ok=True)
        else:
            mode = "assimilation"

        ### ◆(2/4) シミュレーション実行 --------------------------------------
        print("(1/3) simulation...")

        # 初期化
        variables = self.init_variables()

        # 全体計算
        timestamp = 0
        while timestamp < self.df_obs.index[-1]:

            # 通常のマテバラ・熱バラ計算を実行
            variables, all_output = self.calc_all_cell_balance(variables, timestamp)
            # timestamp更新
            timestamp += self.common_conds["dt"]
            timestamp = round(timestamp, 2)
            # 記録
            record_dict["timestamp"].append(timestamp)
            for key, values in all_output.items():
                record_dict[key].append(values)
            record_dict["params"].append(self.common_conds["INFLOW_GAS_COND"])
        ### ◆(3/4) csv出力 -------------------------------------------------
        print("(2/3) csv output...")

        tgt_foldapath = output_foldapath + mode + "/csv/"
        os.makedirs(tgt_foldapath, exist_ok=True)
        plot_csv.outputs_to_csv(tgt_foldapath, record_dict, self.common_conds)

        ### ◆(4/4) 可視化 -------------------------------------------------
        print("(3/3) png output...")
 
        # 可視化対象のセルを算出
        plot_target_sec = []
        loc_cells = np.arange(0, self.common_conds["PACKED_BED_COND"]["Lbed"],
                              self.common_conds["PACKED_BED_COND"]["Lbed"] / self.num_sec) # 各セルの位置
        loc_cells += self.common_conds["PACKED_BED_COND"]["Lbed"] / self.num_sec / 2 # セルの半径を加算
        for value in self.common_conds["LOC_CENCER"].values(): # 温度計に最も近いセルを算出
            plot_target_sec.append(1 + np.argmin(np.abs(loc_cells - value)))
        # all_outputの可視化
        tgt_foldapath = output_foldapath + mode + "/"
        plot_csv.plot_csv_outputs(tgt_foldapath=tgt_foldapath,
                                  df_obs=self.df_obs,
                                  tgt_sections=plot_target_sec,
                                  )

    def calc_all_cell_balance(self, variables, timestamp):
        """ 全体のマテバラ・熱バラを順次計算する

        Args:
            variables (dict): 状態変数
            timestamp (float): 現在時刻

        Returns:
            dict: 更新後の状態変数
            dict: マテバラ・熱バラの全出力（参考・記録用）
        """
        mb_dict = {}
        hb_dict = {}
        hb_wall = {}
        hb_lid = {}
        ### 1. 観測値によるパラメータの更新 -------------------------------------

        # 稼働モードを抽出
        idx_obs = self.df_obs.index[np.argmin(np.abs(timestamp - self.df_obs.index))]
        mode = round(self.df_obs.loc[idx_obs, "mode"])

        # 脱着以外では実験データによる上書きを実施
        if mode != 2:
            # 現在時刻に最も近い観測値のindex
            tgt_index = self.df_obs.index[np.argmin(np.abs(self.df_obs.index - timestamp))]
            # パラメータ更新
            self.common_conds["INFLOW_GAS_COND"]["fr_co2"] = self.df_obs.loc[tgt_index, "flow_rate_co2"] # ガス流量_CO2 [L/min]
            self.common_conds["INFLOW_GAS_COND"]["fr_n2"] = self.df_obs.loc[tgt_index, "flow_rate_n2"] # ガス流量_N2 [L/min]
            variables["total_press"] = self.df_obs.loc[tgt_index, "Pressure"] / 1000 # 圧力[Mpa]
            # 上記に紐づくパラメータの更新
            init_functions.update_params_by_obs(self.common_conds)

        ### 2. マテバラ・熱バラ計算 ------------------------------------------------------

        # 吸着モード
        if mode == 0:
            # 真空ポンプによる排気後圧力等を計算
            # NOTE: 脱着以外ではほぼ欠損値を出力
            vaccume_output = self.calc_total_press_after_vacuuming(variables, mode)
            # マテバラ・熱バラ計算
            for stream in range(1, 1+self.num_str):
                mb_dict[stream] = {}
                hb_dict[stream] = {}
                # sec_1は手動で実施
                mb_dict[stream][1] = self.calc_cell_material_adsorp(stream=stream,
                                                            section=1,
                                                            variables=variables)
                hb_dict[stream][1] = self.calc_cell_heat(stream=stream,
                                                         section=1,
                                                         variables=variables,
                                                         mode=mode,
                                                         material_output=mb_dict[stream][1])
                # sec_2以降は自動で実施
                for section in range(2, 1+self.num_sec):
                    mb_dict[stream][section] = self.calc_cell_material_adsorp(stream=stream,
                                                                    section=section,
                                                                    variables=variables,
                                                                    inflow_gas=mb_dict[stream][section-1])
                    hb_dict[stream][section] = self.calc_cell_heat(stream=stream,
                                                                   section=section,
                                                                   variables=variables,
                                                                   mode=mode,
                                                                   material_output=mb_dict[stream][section],
                                                                   heat_output=hb_dict[stream][section-1],)
            # 排気後モル分率計算
            # NOTE: 脱着以外ではほぼ欠損値を出力
            mf_after_vaccume = self.calc_mf_after_vaccume(vaccume_output=vaccume_output,
                                                          mb_dict=mb_dict,
                                                          mode=mode)
        # 吸着(弁停止)モード
        elif mode == 1:
            # 真空ポンプによる排気後圧力等を計算
            # NOTE: 脱着以外ではほぼ欠損値を出力
            vaccume_output = self.calc_total_press_after_vacuuming(variables, mode)
            # マテバラ・熱バラ計算
            for stream in range(1, 1+self.num_str):
                mb_dict[stream] = {}
                hb_dict[stream] = {}
                # sec_1は手動で実施
                mb_dict[stream][1] = self.calc_cell_material_valve_stop(stream=stream, # 弁停止時のマテバラ計算
                                                                        section=1,
                                                                        variables=variables)
                hb_dict[stream][1] = self.calc_cell_heat(stream=stream,
                                                         section=1,
                                                         variables=variables,
                                                         mode=mode)
                # sec_2以降は自動で実施
                for section in range(2, 1+self.num_sec):
                    mb_dict[stream][section] = self.calc_cell_material_valve_stop(stream=stream, # 弁停止時のマテバラ計算
                                                                                section=section,
                                                                                variables=variables)
                    hb_dict[stream][section] = self.calc_cell_heat(stream=stream,
                                                                   section=section,
                                                                   variables=variables,
                                                                   mode=mode,
                                                                   heat_output=hb_dict[stream][section-1],)
            # 排気後モル分率計算
            # NOTE: 脱着以外ではほぼ欠損値を出力
            mf_after_vaccume = self.calc_mf_after_vaccume(vaccume_output=vaccume_output,
                                                          mb_dict=mb_dict,
                                                          mode=mode)
        # 脱着モード
        if mode == 2:
            # 真空ポンプによる排気後圧力等を計算
            vaccume_output = self.calc_total_press_after_vacuuming(variables, mode)
            #########################################
            # 現在時刻に最も近い観測値のindex
            tgt_index = self.df_obs.index[np.argmin(np.abs(self.df_obs.index - timestamp))]
            # パラメータ更新
            vaccume_output["total_press_after_vaccume"] = self.df_obs.loc[tgt_index, "Pressure"] / 1000 # 圧力[Mpa]
            # #########################################
            # マテバラ・熱バラ計算
            for stream in range(1, 1+self.num_str):
                mb_dict[stream] = {}
                hb_dict[stream] = {}
                # sec_1は手動で実施
                mb_dict[stream][1] = self.calc_cell_material_desorp(stream=stream, # 脱着時のマテバラ計算
                                                                    section=1,
                                                                    variables=variables,
                                                                    vaccume_output=vaccume_output)
                hb_dict[stream][1] = self.calc_cell_heat(stream=stream,
                                                         section=1,
                                                         variables=variables,
                                                         mode=mode,
                                                         material_output=mb_dict[stream][1],
                                                         vaccume_output=vaccume_output)
                # sec_2以降は自動で実施
                for section in range(2, 1+self.num_sec):
                    mb_dict[stream][section] = self.calc_cell_material_desorp(stream=stream, # 脱着時のマテバラ計算
                                                                section=section,
                                                                variables=variables,
                                                                vaccume_output=vaccume_output,)
                    hb_dict[stream][section] = self.calc_cell_heat(stream=stream,
                                                                section=section,
                                                                variables=variables,
                                                                mode=mode,
                                                                material_output=mb_dict[stream][section],
                                                                heat_output=hb_dict[stream][section-1],
                                                                vaccume_output=vaccume_output)
            # 排気後モル分率計算
            mf_after_vaccume = self.calc_mf_after_vaccume(vaccume_output=vaccume_output,
                                                          mb_dict=mb_dict,
                                                          mode=mode)

        ### 3. その他熱バランス -------------------------------------------------------------

        # 壁面熱バラ（stream = 1+self.num_str）
        hb_wall[1] = self.calc_cell_heat_wall(section=1,
                                              variables=variables,
                                              heat_output=hb_dict[self.num_str][1])
        for section in range(2, 1+self.num_sec):
            hb_wall[section] = self.calc_cell_heat_wall(section=section,
                                                        variables=variables,
                                                        heat_output=hb_dict[self.num_str][section],
                                                        heat_wall_output=hb_wall[section-1])
        # 上下蓋熱バラ
        for position in ["up", "down"]:
            hb_lid[position] = self.calc_cell_heat_lid(position=position,
                                                       variables=variables,
                                                       heat_output=hb_dict,
                                                       heat_wall_output=hb_wall)

        ### 4. 状態変数の抽出 --------------------------------------------------

        # 更新後の状態変数
        new_variables = {}
        # 温度
        new_variables["temp"] = {}
        for stream in range(1, 1+self.num_str):
            new_variables["temp"][stream] = {}
            for section in range(1, 1+self.num_sec):
                new_variables["temp"][stream][section] = hb_dict[stream][section]["temp_reached"]
        # 温度（壁面）
        new_variables["temp_wall"] = {}
        for section in range(1, 1+self.num_sec):
            new_variables["temp_wall"][section] = hb_wall[section]["temp_reached"]
        # 上下蓋の温度
        new_variables["temp_lid"] = {}
        for position in ["up", "down"]:
            new_variables["temp_lid"][position] = hb_lid[position]["temp_reached"]
        # 既存吸着量
        new_variables["adsorp_amt"] = {}
        for stream in range(1, 1+self.num_str):
            new_variables["adsorp_amt"][stream] = {}
            for section in range(1, 1+self.num_sec):
                new_variables["adsorp_amt"][stream][section] = mb_dict[stream][section]["accum_adsorp_amt"]
        # 壁―, 層伝熱係数
        new_variables["heat_t_coef"] = {}
        new_variables["heat_t_coef_wall"] = {}
        for stream in range(1, 1+self.num_str):
            new_variables["heat_t_coef"][stream] = {}
            new_variables["heat_t_coef_wall"][stream] = {}
            for section in range(1, 1+self.num_sec):
                new_variables["heat_t_coef"][stream][section] = hb_dict[stream][section]["hw1"]
                new_variables["heat_t_coef_wall"][stream][section] = hb_dict[stream][section]["u1"]
        # 全圧(脱着のみ)
        if mode == 2:
            new_variables["total_press"] = vaccume_output["total_press_after_vaccume"]
        else:
            new_variables["total_press"] = variables["total_press"]
        # モル分率(脱着のみ)
        if mode == 2:
            new_variables["mf_co2"] = mf_after_vaccume["mf_co2_after_vaccume"]
            new_variables["mf_n2"] = mf_after_vaccume["mf_n2_after_vaccume"]
        else:
            new_variables["mf_co2"] = variables["mf_co2"]
            new_variables["mf_n2"] = variables["mf_n2"]

        ### 出力 -----------------------------------------------------------------

        # 全出力結果(参考・記録用)
        output = {
            "material": mb_dict, # マテバラ計算結果
            "heat": hb_dict, # 熱バラ計算結果
            "heat_wall": hb_wall, # 熱バラ計算結果（壁面）
            "heat_lid": hb_lid, # 熱バラ計算結果（蓋）
            "vaccume": {**vaccume_output, **mf_after_vaccume}, # 真空ポンプの排気結果（脱着以外は欠損）
        }

        return new_variables, output

    def calc_cell_material_adsorp(self, stream, section, variables, inflow_gas=None):
        """ 任意セルのマテリアルバランスを計算する
            吸着モード

        Args:
            stream (int): 対象セルのstream番号
            section (int): 対象セルのsection番号
            variables (dict): 温度等の状態変数
            inflow_gas (dict): 上部セルの出力値

        Returns:
            dict: 対象セルの計算結果
        """
        ### マテバラ計算開始 ------------------------------------------------

        # セクション吸着材量 [g]
        Mabs = self.stream_conds[stream]["Mabs"] / self.num_sec
        # 流入CO2流量 [cm3]
        if section == 1:
            inflow_fr_co2 = (
                self.common_conds["INFLOW_GAS_COND"]["fr_co2"] * self.common_conds["dt"]
                * self.stream_conds[stream]["streamratio"] * 1000
            )
        else :
            inflow_fr_co2 = inflow_gas["outflow_fr_co2"]
        # 流入N2流量 [cm3]
        if section == 1:
            inflow_fr_n2 = (
                self.common_conds["INFLOW_GAS_COND"]["fr_n2"] * self.common_conds["dt"]
                * self.stream_conds[stream]["streamratio"] * 1000
            )
        else:
            inflow_fr_n2 = inflow_gas["outflow_fr_n2"]
        # 流入CO2分率
        inflow_mf_co2 = inflow_fr_co2 / (inflow_fr_co2 + inflow_fr_n2)
        # 流入N2分率
        inflow_mf_n2 = inflow_fr_n2 / (inflow_fr_co2 + inflow_fr_n2)
        # 全圧 [MPaA]
        total_press = variables["total_press"]
        # CO2分圧 [MPaA]
        p_co2 = total_press * inflow_mf_co2
        # 現在温度 [℃]
        temp = variables["temp"][stream][section]
        # ガス密度 [kg/m3]
        gas_density = (
            self.common_conds["INFLOW_GAS_COND"]["dense_co2"] * inflow_mf_co2
            + self.common_conds["INFLOW_GAS_COND"]["dense_n2"] * inflow_mf_n2
        )
        # ガス比熱 [kJ/kg/K]
        gas_cp = (
            self.common_conds["INFLOW_GAS_COND"]["cp_co2"] * inflow_mf_co2
            + self.common_conds["INFLOW_GAS_COND"]["cp_n2"] * inflow_mf_n2
        )
        # 現在雰囲気の平衡吸着量 [cm3/g-abs]
        P_kpaa = total_press * 1000 # [kPaA]
        T_K = temp + 273.15 # [K]
        adsorp_amt_equilibrium = self._calc_equilibrium_adsorp_amt(P_kpaa, T_K)
        # 現在の既存吸着量 [cm3/g-abs]
        adsorp_amt_current = variables["adsorp_amt"][stream][section]
        # 理論新規吸着量 [cm3/g-abs]
        adsorp_amt_estimate_abs = (
            self.common_conds["PACKED_BED_COND"]["ks_adsorp"] ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / self.common_conds["PACKED_BED_COND"]["rho_abs"]
            * 6 * (1 - self.common_conds["PACKED_BED_COND"]["epsilon"]) * self.common_conds["PACKED_BED_COND"]["phi"]
            / self.common_conds["PACKED_BED_COND"]["dp"] * (adsorp_amt_equilibrium - adsorp_amt_current)
            * self.common_conds["dt"] / 1e6 * 60
        )
        # セクション理論新規吸着量 [cm3]
        adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
        # 実際のセクション新規吸着量 [cm3]
        adsorp_amt_estimate = min(adsorp_amt_estimate, inflow_fr_co2)
        # 実際の新規吸着量 [cm3/g-abs]
        adsorp_amt_estimate_abs = adsorp_amt_estimate / Mabs
        # 時間経過後吸着量 [cm3/g-abs]
        accum_adsorp_amt = adsorp_amt_current + adsorp_amt_estimate_abs
        # 下流流出CO2流量 [cm3]
        outflow_fr_co2 = inflow_fr_co2 - adsorp_amt_estimate
        # 下流流出N2流量 [cm3]
        outflow_fr_n2 = inflow_fr_n2
        # 流出CO2分率
        outflow_mf_co2 = outflow_fr_co2 / (outflow_fr_co2 + outflow_fr_n2)
        # 流出N2分率
        outflow_mf_n2 = outflow_fr_n2 / (outflow_fr_co2 + outflow_fr_n2)

        output = {
            "inflow_fr_co2": inflow_fr_co2,
            "inflow_fr_n2": inflow_fr_n2,
            "inflow_mf_co2": inflow_mf_co2,
            "inflow_mf_n2": inflow_mf_n2,
            "gas_density": gas_density,
            "gas_cp": gas_cp,
            "adsorp_amt_estimate": adsorp_amt_estimate,
            "accum_adsorp_amt": accum_adsorp_amt,
            "outflow_fr_co2": outflow_fr_co2,
            "outflow_fr_n2": outflow_fr_n2,
            "outflow_mf_co2": outflow_mf_co2,
            "outflow_mf_n2": outflow_mf_n2,
            "adsorp_amt_estimate_abs": adsorp_amt_estimate_abs,
            "desorp_mw_co2": 0,
        }

        return output

    def calc_cell_material_valve_stop(self, stream, section, variables):
        """ 任意セルのマテリアルバランスを計算する
            弁停止モード

        Args:
            stream (int): 対象セルのstream番号
            section (int): 対象セルのsection番号
            variables (dict): 温度等の状態変数

        Returns:
            dict: 対象セルの計算結果
        """
        output = {
            "inflow_fr_co2": 0,
            "inflow_fr_n2": 0,
            "inflow_mf_co2": 0,
            "inflow_mf_n2": 0,
            "gas_density": 0,
            "gas_cp": 0,
            "adsorp_amt_estimate": 0,
            "accum_adsorp_amt": variables["adsorp_amt"][stream][section],
            "outflow_fr_co2": 0,
            "outflow_fr_n2": 0,
            "outflow_mf_co2": 0,
            "outflow_mf_n2": 0,
            "adsorp_amt_estimate_abs": 0,
            "desorp_mw_co2": 0,
        }

        return output

    def calc_cell_material_desorp(self, stream, section, variables, vaccume_output):
        """ 任意セルのマテリアルバランスを計算する
            脱着モード

        Args:
            stream (int): 対象セルのstream番号
            section (int): 対象セルのsection番号
            variables (dict): 温度等の状態変数
            vaccume_output (dict): 排気後圧力計算の出力

        Returns:
            dict: 対象セルの計算結果
        """
        # 現在温度 [℃]
        temp_now = variables["temp"][stream][section]
        # CoolProp用の前準備
        T_K = temp_now + 273.15 # 温度 [K]
        P = vaccume_output["total_press_after_vaccume"] * 1e6 # 圧力 [Pa]        
        # セクション吸着材量 [g]
        Mabs = self.stream_conds[stream]["Mabs"] / self.num_sec
        # CO2分圧 [MPaA]
        p_co2 = vaccume_output["total_press_after_vaccume"] * variables["mf_co2"]
        # ガス密度 [kg/m3]
        gas_density = (
            CP.PropsSI('D', 'T', T_K, 'P', P, "co2") * variables["mf_co2"]
            + CP.PropsSI('D', 'T', T_K, 'P', P, "nitrogen") * variables["mf_n2"]
        )
        # ガス比熱 [kJ/kg/K]
        gas_cp = (
            CP.PropsSI('C', 'T', T_K, 'P', P, "co2") * variables["mf_co2"]
            + CP.PropsSI('C', 'T', T_K, 'P', P, "nitrogen") * variables["mf_n2"]
        )
        # 現在雰囲気の平衡吸着量 [cm3/g-abs]
        P_kpa = P / 1000 # [kPaA]
        adsorp_amt_equilibrium = self._calc_equilibrium_adsorp_amt(P_kpa, T_K)
        # 現在の既存吸着量 [cm3/g-abs]
        adsorp_amt_current = variables["adsorp_amt"][stream][section]
        # 理論新規吸着量 [cm3/g-abs]
        adsorp_amt_estimate_abs = (
            self.common_conds["PACKED_BED_COND"]["ks_desorp"] ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / self.common_conds["PACKED_BED_COND"]["rho_abs"]
            * 6 * (1 - self.common_conds["PACKED_BED_COND"]["epsilon"]) * self.common_conds["PACKED_BED_COND"]["phi"]
            / self.common_conds["PACKED_BED_COND"]["dp"] * (adsorp_amt_equilibrium - adsorp_amt_current)
            * self.common_conds["dt"] / 1e6 * 60
        )
        # セクション理論新規吸着量 [cm3]
        adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
        # 実際のセクション新規吸着量 [cm3]
        adsorp_amt_estimate = max(adsorp_amt_estimate, -adsorp_amt_current)
        # 実際の新規吸着量 [cm3/g-abs]
        adsorp_amt_estimate_abs = adsorp_amt_estimate / Mabs
        # 時間経過後吸着量 [cm3/g-abs]
        accum_adsorp_amt = adsorp_amt_current + adsorp_amt_estimate_abs
        # 気相放出CO2量 [cm3]
        desorp_mw_co2 = -1 * adsorp_amt_estimate
        # 気相放出CO2量 [mol]
        desorp_mw_co2 = desorp_mw_co2 / 1000 / 22.4

        # 出力
        output = {
            "inflow_fr_co2": 0,
            "inflow_fr_n2": 0,
            "inflow_mf_co2": 0,
            "inflow_mf_n2": 0,
            "gas_density": gas_density,
            "gas_cp": gas_cp,
            "adsorp_amt_estimate": adsorp_amt_estimate,
            "accum_adsorp_amt": accum_adsorp_amt,
            "outflow_fr_co2": 0,
            "outflow_fr_n2": 0,
            "outflow_mf_co2": 0,
            "outflow_mf_n2": 0,
            "adsorp_amt_estimate_abs": adsorp_amt_estimate_abs,
            "desorp_mw_co2": desorp_mw_co2,
        }

        return output

    def calc_cell_heat(self, stream, section, variables, mode,
                       material_output=None, heat_output=None, vaccume_output=None):
        """ 対象セルの熱バランスを計算する

        Args:
            stream (int): 対象セルのstream番号
            section (int): 対象セルのsection番号
            variables (dict): 状態変数
            mode (int): 吸着・脱着等の運転モード
            material_output (dict): 対象セルのマテバラ出力
            heat_output (dict): 上流セクションの熱バラ出力
            vaccume_output (dict): 排気後圧力計算の出力（脱着時）

        Returns:
            dict: 対象セルの熱バラ出力
        """
        ### 前準備 ------------------------------------------------------

        # セクション現在温度 [℃]
        temp_now = variables["temp"][stream][section]
        # 内側セクション温度 [℃]
        if stream == 1:
            temp_inside_cell = 18
        else:
            temp_inside_cell = variables["temp"][stream-1][section]
        # 外側セクション温度 [℃]
        if stream != self.num_str:
            temp_outside_cell = variables["temp"][stream+1][section]
        else:
            temp_outside_cell = variables["temp_wall"][section]
        # 下流セクション温度 [℃]
        if section != self.num_sec:
            temp_below_cell = variables["temp"][stream][section+1]
        # 発生する吸着熱 [J]
        if mode == 1:
            Habs = 0 # 弁停止モードでは0
        else:
            Habs = (
                material_output["adsorp_amt_estimate"] / 1000 / 22.4
                * self.common_conds["INFLOW_GAS_COND"]["mw_co2"]
                * self.common_conds["INFLOW_GAS_COND"]["adsorp_heat_co2"]
            )
        # 流入ガス質量 [g]
        if mode in [1, 2]:
            Mgas = 0 # 弁停止・脱着モードでは0
        else:
            Mgas = (
                material_output["inflow_fr_co2"] / 1000 / 22.4
                * self.common_conds["INFLOW_GAS_COND"]["mw_co2"]
                + material_output["inflow_fr_n2"] / 1000 / 22.4
                * self.common_conds["INFLOW_GAS_COND"]["mw_n2"]
            )
        # 流入ガス比熱 [J/g/K]
        if mode == 1:
            gas_cp = 0 # 弁停止モードでは0
        else:
            gas_cp = material_output["gas_cp"]
        # 内側境界面積 [m2]
        Ain = self.stream_conds[stream]["Ain"] / self.num_sec
        # 外側境界面積 [m2]
        Aout = self.stream_conds[stream]["Aout"] / self.num_sec
        # 下流セル境界面積 [m2]
        Abb = self.stream_conds[stream]["Sstream"]
        # 壁-層伝熱係数、層伝熱係数
        if mode == 0:
            hw1, u1 = self._calc_heat_transfer_coef(stream,
                                                    temp_now,
                                                    mode,
                                                    variables,
                                                    material_output)
        elif mode == 1: # 弁停止モードでは直前値に置換
            hw1 = variables["heat_t_coef"][stream][section]
            u1 = variables["heat_t_coef_wall"][stream][section]
        elif mode == 2: # 脱着モードでは入力に排気後圧力計算の出力を使用
            hw1 = variables["heat_t_coef"][stream][section]
            u1 = variables["heat_t_coef_wall"][stream][section]
            # hw1, u1 = self._calc_heat_transfer_coef(stream,
            #                                         temp_now,
            #                                         mode,
            #                                         variables,
            #                                         material_output,
            #                                         vaccume_output)

        ### 熱流束計算 ---------------------------------------------------

        # 内側境界からの熱流束 [J]
        if stream == 1:
            Hwin = 0
        else:
            Hwin = u1 * Ain * (temp_inside_cell - temp_now) * self.common_conds["dt"] * 60
        # 外側境界への熱流束 [J]
        if stream == self.num_str:
            Hwout = hw1 * Aout * (temp_now - temp_outside_cell) * self.common_conds["dt"] * 60
        else :
            Hwout = u1 * Aout * (temp_now - temp_outside_cell) * self.common_conds["dt"] * 60
        # 下流セルへの熱流束 [J]
        if section == self.num_sec:
            Hbb = ( # 下蓋への熱流束
                hw1 * self.stream_conds[stream]["Sstream"]
                * (temp_now - variables["temp_lid"]["down"])
                * self.common_conds["dt"] * 60
            )
        else:
            Hbb = ( # 下流セルへの熱流束
                u1 * Abb * (temp_now - temp_below_cell)
                * self.common_conds["dt"] * 60
            )
        # 上流セルヘの熱流束 [J]
        if section == 1: # 上蓋からの熱流束
            Hroof = (
                hw1 * self.stream_conds[stream]["Sstream"]
                * (temp_now - variables["temp_lid"]["up"])
                * self.common_conds["dt"] * 60
            )
        else: # 上流セルヘの熱流束 = -1 * 上流セルの「下流セルへの熱流束」
            Hroof = - heat_output["Hbb"]

        ### 到達温度計算 --------------------------------------------------------------

        # セクション到達温度 [℃]
        args = {
            "gas_cp": gas_cp,
            "Mgas": Mgas,
            "temp_now": temp_now,
            "Habs": Habs,
            "Hwin": Hwin,
            "Hwout": Hwout,
            "Hbb": Hbb,
            "Hroof": Hroof,
            "stream": stream,
        }
        temp_reached = optimize.newton(self.__optimize_temp_reached, temp_now, args=args.values())

        output = {
            "temp_reached": temp_reached,
            "hw1": hw1, # 壁-層伝熱係数
            "Hroof": Hroof,
            "Hbb": Hbb, # 下流セルへの熱流束 [J]
            ### 以下確認用
            "Habs": Habs, # 発生する吸着熱 [J]
            "Hwin": Hwin, # 内側境界からの熱流束 [J]
            "Hwout": Hwout, # 外側境界への熱流束 [J]
            "u1": u1, # 層伝熱係数 [W/m2/K]
        }

        return output

    def calc_cell_heat_wall(self, section, variables, heat_output, heat_wall_output=None):
        """ 壁面の熱バランス計算

        Args:
            section (int): 対象のセクション番号
            variables (dict): 各セルの変数
            heat_output (dict): 隣接セルの熱バラ出力
            heat_wall_output (dict, optional): 上流セルの壁面熱バラ出力. Defaults to None.

        Returns:
            dict: 壁面熱バラ出力
        """
        # セクション現在温度 [℃]
        temp_now = variables["temp_wall"][section]
        # 内側セクション温度 [℃]
        temp_inside_cell = variables["temp"][self.num_str][section]
        # 外側セクション温度 [℃]
        temp_outside_cell = self.common_conds["DRUM_WALL_COND"]["temp_outside"]
        # 下流セクション温度 [℃]
        if section != self.num_sec:
            temp_below_cell = variables["temp_wall"][section+1]
        # 上流壁への熱流束 [J]
        if section == 1:
            Hroof = (
                self.common_conds["DRUM_WALL_COND"]["c_drumw"]
                * self.stream_conds[3]["Sstream"]
                * (temp_now - variables["temp_lid"]["up"])
                * self.common_conds["dt"] * 60
            )
        else:
            Hroof = heat_wall_output["Hbb"]
        # 内側境界からの熱流束 [J]
        Hwin = (
            heat_output["hw1"] * self.stream_conds[3]["Ain"] / self.num_sec
            * (temp_inside_cell - temp_now) * self.common_conds["dt"] * 60
        )
        # 外側境界への熱流束 [J]
        Hwout = (
            self.common_conds["DRUM_WALL_COND"]["coef_outair_heat"] * self.stream_conds[3]["Aout"]
            / self.num_sec * (temp_now - temp_outside_cell)
            * self.common_conds["dt"] * 60
        )
        # 下流壁への熱流束 [J]
        if section == self.num_sec:
            Hbb = (
                self.common_conds["DRUM_WALL_COND"]["c_drumw"]
                * self.stream_conds[3]["Sstream"]
                * (temp_now - variables["temp_lid"]["down"])
                * self.common_conds["dt"] * 60
            )
        else:
            Hbb = (
                self.common_conds["DRUM_WALL_COND"]["c_drumw"] * self.stream_conds[3]["Sstream"]
                * (temp_now - variables["temp_wall"][section+1])
            )
        # セクション到達温度 [℃]
        args = {
            "temp_now": temp_now,
            "Hwin": Hwin,
            "Hwout": Hwout,
            "Hbb": Hbb,
            "Hroof": Hroof,
        }
        temp_reached = optimize.newton(self.__optimize_temp_reached_wall, temp_now, args=args.values())

        output = {
            "Hbb": Hbb,
            "temp_reached": temp_reached,
            "Hroof": Hroof, # 上流壁への熱流束 [J]
            ### 以下記録用
            "Hwin": Hwin, # 内側境界からの熱流束 [J]
            "Hwout": Hwout, # 外側境界への熱流束 [J]
            "Hbb": Hbb, # 下流壁への熱流束 [J]
        }

        return output

    def calc_cell_heat_lid(self, position, variables, heat_output, heat_wall_output):
        """ 上下蓋の熱バランス計算

        Args:
            position (str): 上と下のどちらの蓋か
            variables (dict): 各セルの変数
            heat_output (dict): 各セルの熱バラ出力

        Returns:
            dict: 熱バラ出力
        """
        # セクション現在温度 [℃]
        temp_now = variables["temp_lid"][position]
        # 外気への熱流束 [J]
        Hout = (
            self.common_conds["DRUM_WALL_COND"]["coef_outair_heat"]
            * (temp_now - self.common_conds["DRUM_WALL_COND"]["temp_outside"])
            * self.common_conds["dt"] * 60
        )
        if position == "up":
            Hout *= self.common_conds["LID_COND"]["DOWN"]["Sflange_up"]
        elif position == "down":
            Hout *= self.common_conds["LID_COND"]["UP"]["Sflange_up"]
        # 底が受け取る熱(熱収支基準)
        if position == "up":
            Hlidall_heat = (
                heat_output[2][1]["Hroof"] - heat_output[1][1]["Hroof"]
                - Hout - heat_wall_output[1]["Hroof"]
            )
        else:
            Hlidall_heat = (
                heat_output[2][self.num_sec]["Hroof"] - heat_output[1][self.num_sec]["Hroof"]
                - Hout - heat_wall_output[self.num_sec]["Hbb"]
            )
        # セクション到達温度 [℃]
        args = {
            "temp_now": temp_now,
            "Hlidall_heat": Hlidall_heat,
            "position": position,
        }
        temp_reached = optimize.newton(self.__optimize_temp_reached_lid, temp_now, args=args.values())

        output = {
            "temp_reached": temp_reached,
            # "Hlidall_heat": Hlidall_heat,
            # "Hlid_time": Hlid_time,
        }

        return output

    def calc_total_press_after_vacuuming(self, variables, mode):
        """ 排気後圧力の計算

        Args:
            variables (dict): 状態変数

        Returns:
            dict: 排気後圧力と排気後CO2・N2モル量
        """
        # 脱着時のみ計算
        if mode == 2:
            # 容器内平均温度 [℃]
            temp_now_mean = np.mean([
                variables["temp"][stream][section] for stream in range(1,1+self.num_str) for section in range(1,1+self.num_sec)
            ])
            # 真空ポンプ排気速度 [m3/min]
            if variables["total_press"]*1e6 > 900:
                vacuum_rate = 0.57
            elif variables["total_press"]*1e6 > 100:
                vacuum_rate = 0.085 * np.log(variables["total_press"]*1e6 - 99)
            else:
                vacuum_rate = 0
            # 真空ポンプ排気速度 [mol]
            vacuum_rate = (
                variables["total_press"]*1e6 * vacuum_rate / 8.314 / (temp_now_mean + 273.15)
            )
            # 排気量 [mol]
            vaccume_amount = vacuum_rate * self.common_conds["dt"]
            # 容器内部空間体積 [m3]
            case_inner_volume = (
                self.common_conds["PACKED_BED_COND"]["Vbed"] * self.common_conds["PACKED_BED_COND"]["epsilon"]
            )
            # 容器内部空間物質量 [mol]
            case_inner_amount = (
                variables["total_press"]*1e6 * case_inner_volume / 8.314 /
                (temp_now_mean + 273.15)
            )
            # 排気後容器内部空間物質量 [mol]
            case_inner_amount_after_vaccume = case_inner_amount - vaccume_amount
            # 排気後容器内部圧力 [MPaA]
            total_press_after_vaccume = (
                case_inner_amount_after_vaccume * 8.314 * (temp_now_mean + 273.15)
                / case_inner_volume / 1e6
            )
            # 排気後CO2モル量 [mol]
            mw_co2_after_vaccume = variables["mf_co2"] * case_inner_amount_after_vaccume
            # 排気後N2モル量 [mol]
            mw_n2_after_vaccume = variables["mf_n2"] * case_inner_amount_after_vaccume

            # 出力
            output = {
                "case_inner_amount_after_vaccume": case_inner_amount_after_vaccume, # 排気後容器内部空間物質量
                "total_press_after_vaccume": total_press_after_vaccume, # 排気後圧力
                "mw_co2_after_vaccume": mw_co2_after_vaccume, # 排気後CO2モル量
                "mw_n2_after_vaccume": mw_n2_after_vaccume, # 排気後N2モル量
                "vacuum_rate": vacuum_rate, # 真空ポンプの排気速度
            }

        else: # 脱着以外
            output = {
                "case_inner_amount_after_vaccume": np.nan,
                "total_press_after_vaccume": variables["total_press"], # 排気後圧力
                "mw_co2_after_vaccume": np.nan, # 排気後CO2モル量
                "mw_n2_after_vaccume": np.nan, # 排気後N2モル量
                "vacuum_rate": np.nan, # 真空ポンプの排気速度
            }

        return output

    def calc_mf_after_vaccume(self, vaccume_output, mb_dict, mode):
        """ 排気後モル分率の計算

        Args:
            vaccume_output (dict): 排気後圧力計算の出力
            mb_dict (dict): マテバラ計算の出力

        Returns:
            dict: 排気後モル分率
        """
        # 脱着時のみ計算
        if mode == 2:
            # 脱着気相放出CO2モル量 [mol]
            sum_desorp_mw = sum([
                mb_dict[stream][section]["desorp_mw_co2"] for stream in range(1,1+self.num_str) for section in range(1,1+self.num_sec)
            ])
            # 脱着後CO2モル量 [mol]
            mw_co2_after_vaccume = sum_desorp_mw + vaccume_output["mw_co2_after_vaccume"]
            # 脱着後N2モル量 [mol]
            mw_n2_after_vaccume = vaccume_output["mw_n2_after_vaccume"]
            # 脱着後CO2モル分率
            mf_co2_after_vaccume = mw_co2_after_vaccume / (mw_co2_after_vaccume + mw_n2_after_vaccume)
            # 脱着後N2モル分率
            mf_n2_after_vaccume = mw_n2_after_vaccume / (mw_co2_after_vaccume + mw_n2_after_vaccume)

            output = {
                "sum_desorp_mw": sum_desorp_mw, # 脱着気相放出CO2モル量
                "mf_co2_after_vaccume": mf_co2_after_vaccume, # 脱着後CO2モル分率
                "mf_n2_after_vaccume": mf_n2_after_vaccume, # 脱着後N2モル分率
            }

        else: # 脱着以外
            output = {
                "sum_desorp_mw": np.nan,
                "mf_co2_after_vaccume": self.common_conds["INFLOW_GAS_COND"]["mf_co2"],
                "mf_n2_after_vaccume": self.common_conds["INFLOW_GAS_COND"]["mf_n2"],
            }

        return output

    def _calc_heat_transfer_coef(self, stream, temp_now, mode, variables,
                                 material_output, vaccume_output=None):
        """ 層伝熱係数、壁-層伝熱係数を算出する

        Args:
            stream (int): 対象のストリーム番号
            temp_now (float): 対象セルの現在温度
            mode (str): 吸着・脱着等の運転モード
            material_output (dict): 対象セルのマテバラ計算結果
            vaccume_output (dict): 排気後圧力計算の出力（脱着時）

        Returns:
            float: 層伝熱係数、壁-層伝熱係数
        """
        # CoolProp用の前準備
        T_K = temp_now + 273.15 # 温度 [K]
        if mode == 0: # 吸着
            P = variables["total_press"] * 1e6 # 圧力 [Pa]
            mf_co2 = material_output["inflow_mf_co2"] # 流入ガスのCO2モル分率
            mf_n2 = material_output["inflow_mf_n2"] # 流入ガスのN2モル分率
        elif mode == 2: # 脱着
            P = vaccume_output["total_press_after_vaccume"] * 1e6 # 圧力 [Pa]
            mf_co2 = variables["mf_co2"] # 気相中のCO2モル分率
            mf_n2 = variables["mf_n2"] # 気相中のN2モル分率
        # 導入気体の熱伝導率 [W/m/K]
        kf = (
            CP.PropsSI('L', 'T', T_K, 'P', P, "co2") * mf_co2
            + CP.PropsSI('L', 'T', T_K, 'P', P, "nitrogen") * mf_n2
        ) / 1000
        # 充填剤の熱伝導率 [W/m/K]
        kp = self.common_conds["PACKED_BED_COND"]["lambda_col"]
        # Yagi-Kunii式 1
        Phi_1 = 0.15
        # Yagi-Kunii式 2
        Phi_2 = 0.07
        # Yagi-Kunii式 3
        Phi = (
            Phi_2 + (Phi_1 - Phi_2)
            * (self.common_conds["PACKED_BED_COND"]["epsilon"] - 0.26)
            / 0.26
        )
        # Yagi-Kunii式 4
        hrv = (
            (0.227 / (1 + self.common_conds["PACKED_BED_COND"]["epsilon"] / 2
                      / (1 - self.common_conds["PACKED_BED_COND"]["epsilon"])
                      * (1 - self.common_conds["PACKED_BED_COND"]["epsilon_p"])
                      / self.common_conds["PACKED_BED_COND"]["epsilon_p"]))
            * ((temp_now + 273.15) / 100)**3
        )
        # Yagi-Kunii式 5
        hrp = (
            0.227 * self.common_conds["PACKED_BED_COND"]["epsilon_p"]
            / (2 - self.common_conds["PACKED_BED_COND"]["epsilon_p"])
            * ((temp_now + 273.15) / 100)**3
        )
        # Yagi-Kunii式 6
        ksi = (
            1 / Phi + hrp * self.common_conds["PACKED_BED_COND"]["dp"] / kf
        )
        # Yagi-Kunii式 7
        ke0_kf = (
            self.common_conds["PACKED_BED_COND"]["epsilon"]
            * (1 + hrv * self.common_conds["PACKED_BED_COND"]["dp"] / kf)
            + (1 - self.common_conds["PACKED_BED_COND"]["epsilon"])
            / (1 / ksi + 2 * kf / 3 / kp)
        )
        # 静止充填層有効熱伝導率 [W/m/K]
        ke0 = kf * ke0_kf        
        # ストリーム換算直径 [m]
        d1 = 2 * (self.stream_conds[stream]["Sstream"] / math.pi)**0.5
        # 気体粘度 [Pas]
        mu = (
            CP.PropsSI('V', 'T', T_K, 'P', P, "co2") * mf_co2
            + CP.PropsSI('V', 'T', T_K, 'P', P, "nitrogen") * mf_n2
        )
        # プラントル数
        Pr = mu * 1000 * material_output["gas_cp"] / kf
        # 流入ガス体積流量 [m3/s]
        if mode == 0:
            f0 = (
                (material_output["inflow_fr_co2"] + material_output["inflow_fr_n2"])
                / 1e6 / (self.common_conds["dt"] * 60)
            )
        elif mode == 2: # 脱着時は排気ガス体積流量 [m3/s]
            f0 = (
                vaccume_output["vacuum_rate"] / 60 * self.stream_conds[stream]["streamratio"]
            )
        # ストリーム空塔速度 [m/s]
        vcol = f0 / self.stream_conds[stream]["Sstream"]
        # 気体動粘度 [m2/s]
        nu = mu / material_output["gas_density"]
        # 粒子レイノルズ数
        Rep = vcol * self.common_conds["PACKED_BED_COND"]["dp"] / nu
        # 充填層有効熱伝導率 1
        psi_beta = (
            1.0985 * (self.common_conds["PACKED_BED_COND"]["dp"] / d1)**2
            - 0.5192 * (self.common_conds["PACKED_BED_COND"]["dp"] / d1) + 0.1324
        )
        # 充填層有効熱伝導率 2
        ke_kf = ke0 / kf + psi_beta * Pr * Rep
        # 充填層有効熱伝導率 3 [W/m/K]
        ke = ke_kf * kf
        # ヌッセルト数
        Nup = 0.84 * Rep
        # 粒子‐流体間熱伝達率 [W/m2/K]
        habs = Nup / self.common_conds["PACKED_BED_COND"]["dp"] * kf
        # 隙間係数
        a = 2
        # 格子長さ [m]
        l0 = self.common_conds["PACKED_BED_COND"]["dp"] * a * 2 / 2**0.5
        # 粒子代表長さ [m]
        dlat = l0 * (1 - self.common_conds["PACKED_BED_COND"]["epsilon"])
        # 代表長さ（セクション全長）1
        c0 = 4
        # 代表長さ（セクション全長）2
        Lambda_2 = c0 * Pr ** (1/3) * Rep ** (1/2)
        # 代表長さ（セクション全長）3
        knew = ke + 1 / (1 / (0.02 * Pr * Rep) + 2 / Lambda_2)
        # 代表長さ（セクション全長）4
        Lambda_1 = 2 / (kf / ke - kf / knew)
        # 代表長さ（セクション全長）5
        b0 = (
            0.5 * Lambda_1 * d1 / self.common_conds["PACKED_BED_COND"]["dp"]
            * kf / ke
        )
        # 代表長さ（セクション全長）6
        Phi_b = 0.0775 * np.log(b0)+0.028
        # 代表長さ（セクション全長）7
        a12 = 0.9107 * np.log(b0) + 2.2395
        # 代表長さ（セクション全長）8 [m]
        Lp = self.common_conds["PACKED_BED_COND"]["Lbed"] / self.num_sec
        # 粒子層-壁面伝熱ヌッセルト数 1
        y0 = (
            4 * self.common_conds["PACKED_BED_COND"]["dp"]
            / d1 * Lp / d1 * ke_kf / (Pr * Rep)
        )
        # 粒子層-壁面伝熱ヌッセルト数 1
        Nupw = (
            self.common_conds["PACKED_BED_COND"]["dp"]
            / d1 * ke_kf * (a12 + Phi_b / y0)
        )
        # 壁-層伝熱係数 [W/m2/K]
        hw1 = Nupw / self.common_conds["PACKED_BED_COND"]["dp"] * kf
        hw1 *= self.common_conds["DRUM_WALL_COND"]["coef_hw1"]
        # 層伝熱係数 [W/m2/K]
        u1 = 1 / (dlat / ke + 1 / habs)

        return hw1, u1

    def _calc_equilibrium_adsorp_amt(self, P, T):
        """ 平衡吸着量を計算

        Args:
            P (float): 圧力 [kPaA]
            T (float): 温度 [K]
        Returns:
            平衡吸着量
        """
        # 吸着等温式（シンボリック回帰による近似式）
        adsorp_amt_equilibrium = (
            P * (252.0724 - 0.50989705 * T)
            / (P - 3554.54819062669*(1 - 0.0655247236249063 * np.sqrt(T))**3 + 1.7354268)
        )
        return adsorp_amt_equilibrium

    def __optimize_temp_reached(self, temp_reached,
                                gas_cp, Mgas, temp_now,
                                Habs, Hwin, Hwout, Hbb, Hroof,
                                stream):
        """セクション到達温度算出におけるソルバー用関数

        Args:
            temp_reached (float): セクション到達温度
            args (dict): 充填層が受ける熱を計算するためのパラメータ

        Returns:
            float: 充填層が受ける熱の熱収支基準と時間基準の差分
        """
        # 流入ガスが受け取る熱 [J]
        Hgas = gas_cp * Mgas * (temp_reached - temp_now)
        # 充填層が受け取る熱(ΔT基準) [J]
        Hbed_time = (
            self.common_conds["PACKED_BED_COND"]["Cbed"] * self.stream_conds[stream]["streamratio"]
            / self.num_sec * (temp_reached - temp_now)
        )
        # 充填層が受け取る熱(熱収支基準) [J]
        Hbed_heat_blc = Habs - Hgas + Hwin - Hwout - Hbb - Hroof
        return Hbed_heat_blc - Hbed_time

    def __optimize_temp_reached_wall(self, temp_reached,
                                     temp_now, Hwin, Hwout, Hbb, Hroof,
                                     ):
        """壁面の到達温度算出におけるソルバー用関数

        Args:
            temp_reached (float): セクション到達温度
            args (dict): 充填層が受ける熱を計算するためのパラメータ

        Returns:
            float: 充填層が受ける熱の熱収支基準と時間基準の差分
        """
        # 壁が受け取る熱(熱収支基準) [J]
        Hwall_heat_blc = Hwin - Hroof - Hwout - Hbb
        # 壁が受け取る熱(ΔT基準) [J]
        Hwall_time = (
            self.common_conds["DRUM_WALL_COND"]["sh_drumw"] * self.stream_conds[1+self.num_str]["Mwall"]
            * (temp_reached - temp_now)
        )

        return Hwall_heat_blc - Hwall_time

    def __optimize_temp_reached_lid(self, temp_reached,
                                    temp_now, Hlidall_heat, position
                                    ):
        """上下蓋の到達温度算出におけるソルバー用関数

        Args:
            temp_reached (float): セクション到達温度
            args (dict): 充填層が受ける熱を計算するためのパラメータ

        Returns:
            float: 充填層が受ける熱の熱収支基準と時間基準の差分
        """
        # 壁が受け取る熱(ΔT基準) [J]
        Hlid_time = (
            self.common_conds["DRUM_WALL_COND"]["sh_drumw"] * (temp_reached - temp_now)
        )
        if position == "up":
            Hlid_time *= self.common_conds["LID_COND"]["UP"]["Mflange"]
        else:
            Hlid_time *= self.common_conds["LID_COND"]["DOWN"]["Mflange"]

        return Hlidall_heat - Hlid_time