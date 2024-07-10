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

from utils import const, init_functions, plot_csv

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
        self.df_obs = pd.read_excel(filepath, sheet_name="python実装用_吸着のみ", index_col="time")

        # その他初期化

    def init_variables(self):
        """温度等の状態変数を初期化する

        Returns:
            dict: 初期化した状態変数
        """
        variables = {}
        # 各セルの温度
        variables["temp"] = {}
        for stream in range(1, self.num_str+2):
            variables["temp"][stream] = {}
            for section in range(1, self.num_sec+1):
                variables["temp"][stream][section] = \
                    self.common_conds["DRUM_WALL_COND"]["temp_outside"]

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
            "all_output": [],
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
            # 1step計算実行
            variables, all_output = self.calc_all_cell_balance(variables=variables)
            # timestamp更新
            timestamp += self.common_conds["dt"]
            timestamp = round(timestamp, 2)
            # 記録用配列の平坦化
            output_flatten = {}
            for stream in range(1, 2+self.num_str): # 熱バラ
                for section in range(1, 1+self.num_sec):
                        for key, value in all_output["heat"][stream][section].items():
                            output_flatten[key+"_"+str(stream)+str(section)] = value
            for stream in range(1, 1+self.num_str): # マテバラ
                for section in range(1, 1+self.num_sec):
                        for key, value in all_output["material"][stream][section].items():
                            output_flatten[key+"_"+str(stream)+str(section)] = value
            # 記録
            record_dict["timestamp"].append(timestamp)
            record_dict["all_output"].append(output_flatten)

        ### ◆(3/4) csv出力 -------------------------------------------------
        print("(2/3) csv output...")

        # DataFrame化
        values = []
        for i in range(len(record_dict["all_output"])):
            values.append(record_dict["all_output"][i].values())
        df = pd.DataFrame(values,
                          index=record_dict["timestamp"],
                          columns=record_dict["all_output"][0].keys())
        df.index.name = "timestamp"

        # DataFrameを細分化
        df_dict = {}
        tgt_items = [x[:-3] for x in record_dict["all_output"][0].keys()] # 記録対象
        tgt_items = set(tgt_items) # 重複削除
        for item in tgt_items: # 細分化
            tgt_col = [col for col in df.columns if item in col]
            df_dict[item] = df[tgt_col]
        del df

        # csv出力
        for item in tgt_items:
            foldapath = output_foldapath + f"{mode}/csv/"
            os.makedirs(foldapath, exist_ok=True)
            df_dict[item].to_csv(foldapath + const.TRANSLATION[item] + ".csv")

        ### ◆(4/4) 可視化 -------------------------------------------------
        print("(3/3) png output...")
        plot_csv.plot_csv_files(tgt_foldapath = output_foldapath + mode + "/",
                                unit_dict=const.UNIT)

    def calc_all_cell_balance(self, variables):
        """全体のマテバラ・熱バラを順次計算する

        Args:
            variables (dict): 状態変数

        Returns:
            dict: 更新後の状態変数
            dict: マテバラ・熱バラの全出力（参考・記録用）
        """
        ### マテバラ・熱バラ計算
        mb_dict = {}
        hb_dict = {}
        for stream in range(1, 1+self.num_str):
            mb_dict[stream] = {}
            hb_dict[stream] = {}
            # sec_1は手動で実施
            mb_dict[stream][1] = self.calc_cell_material(stream=stream,
                                                         section=1,
                                                         variables=variables)
            hb_dict[stream][1] = self.calc_cell_heat(stream=stream,
                                                     section=1,
                                                     variables=variables,
                                                     material_output=mb_dict[stream][1])
            # sec_2以降は自動で実施
            for section in range(2, 1+self.num_sec):
                mb_dict[stream][section] = self.calc_cell_material(stream=stream,
                                                                   section=section,
                                                                   variables=variables,
                                                                   inflow_gas=mb_dict[stream][section-1])
                hb_dict[stream][section] = self.calc_cell_heat(stream=stream,
                                                               section=section,
                                                               variables=variables,
                                                               material_output=mb_dict[stream][section])
        # 壁面熱バラ（stream = self.num_sec）
        hb_dict[self.num_sec] = {}
        hb_dict[self.num_sec][1] = self.calc_cell_heat_wall(section=1,
                                                            variables=variables,
                                                            heat_output=hb_dict[self.num_str-1][1])
        for section in range(2, 1+self.num_sec):
            hb_dict[self.num_sec][section] = self.calc_cell_heat_wall(section=section,
                                                                      variables=variables,
                                                                      heat_output=hb_dict[self.num_str-1][section],
                                                                      heat_wall_output=hb_dict[self.num_sec][section-1])

        ### 出力
        # 更新後の状態変数
        new_variables = {}
        # 温度
        new_variables["temp"] = {}
        for stream in range(1, 2+self.num_str): # 壁面考慮
            new_variables["temp"][stream] = {}
            for section in range(1, 1+self.num_sec):
                new_variables["temp"][stream][section] = hb_dict[stream][section]["temp_reached"]
        # 全出力結果(参考・記録用)
        output = {
            "material": mb_dict,
            "heat": hb_dict,
        }

        return new_variables, output

    def calc_cell_material(self, stream, section, variables, inflow_gas=None):
        """任意セルのマテリアルバランスを計算する

        Args:
            stream (int): 対象セルのstream番号
            section (int): 対象セルのsection番号
            variables (dict): 温度等の状態変数
            inflow_gas (dict): 上部セルの出力値

        Returns:
            dict: 対象セルの計算結果
        """
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
        press = self.common_conds["INFLOW_GAS_COND"]["press"]
        # CO2分圧 [MPaA]
        p_co2 = press * inflow_mf_co2
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
        adsorp_amt_equilibrium = (
            (0.0000021*(press*1000)**2-0.0003385*(press*1000)+0.0145345)*(temp+273.15)**2
            +(-0.0012701*(press*1000)**2+0.2091781*(press*1000)-9.9261428)*(temp+273.15)
            +0.1828984*(press*1000)**2-30.8594655*(press*1000)+1700.5767712
        )
        # 現在の既存吸着量 [cm3/g-abs]
        adsorp_amt_current = (
            (0.0000021*self.common_conds["PACKED_BED_COND"]["vp"]**2-0.0003385
             *self.common_conds["PACKED_BED_COND"]["vp"]+0.0145345)*(temp+273.15)**2
            +(-0.0012701*self.common_conds["PACKED_BED_COND"]["vp"]**2+0.2091781
              *self.common_conds["PACKED_BED_COND"]["vp"]-9.9261428)*(temp+273.15)
            +0.1828984*self.common_conds["PACKED_BED_COND"]["vp"]**2
            -30.8594655*self.common_conds["PACKED_BED_COND"]["vp"]+1700.5767712
        )
        # 理論新規吸着量 [cm3/g-abs]
        adsorp_amt_estimate_abs = (
            self.common_conds["PACKED_BED_COND"]["ks"] / self.common_conds["PACKED_BED_COND"]["rho_abs"]
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
        }

        return output

    def calc_cell_heat(self, stream, section, variables, material_output):
        """ 対象セルの熱バランスを計算する

        Args:
            stream (int): 対象セルのstream番号
            section (int): 対象セルのsection番号
            variables (dict): 状態変数
            material_output (dict): 対象セルのマテバラ出力

        Returns:
            dict: 対象セルの熱バラ出力
        """
        # セクション現在温度 [℃]
        temp_now = variables["temp"][stream][section]
        # 内側セクション温度 [℃]
        if stream == 1:
            temp_inside_cell = 18
        else:
            temp_inside_cell = variables["temp"][stream-1][section]
        # 外側セクション温度 [℃]
        temp_outside_cell = variables["temp"][stream+1][section]
        # 下流セクション温度 [℃]
        if section != self.num_sec:
            temp_below_cell = variables["temp"][stream][section+1]
        # 発生する吸着熱 [J]
        Habs = (
            material_output["adsorp_amt_estimate"] / 1000 / 22.4
            * self.common_conds["INFLOW_GAS_COND"]["mw_co2"]
            * self.common_conds["INFLOW_GAS_COND"]["adsorp_heat_co2"]
        )
        # 流入ガス質量 [g]
        Mgas = (
            material_output["inflow_fr_co2"] / 1000 / 22.4
            * self.common_conds["INFLOW_GAS_COND"]["mw_co2"]
            + material_output["inflow_fr_n2"] / 1000 / 22.4
            * self.common_conds["INFLOW_GAS_COND"]["mw_n2"]
        )
        # 流入ガス比熱 [J/g/K]
        gas_cp = material_output["gas_cp"]
        # 上流壁への放熱 [J]
        Hroof = 0
        # 内側境界面積 [m2]
        Ain = self.stream_conds[stream]["Ain"] / self.num_sec
        # 外側境界面積 [m2]
        Aout = self.stream_conds[stream]["Aout"] / self.num_sec
        # 下流セル境界面積 [m2]
        Abb = self.stream_conds[stream]["Sstream"]

        ### 層伝熱係数

        # 導入気体の熱伝導率 [W/m/K]
        kf = (
            self.common_conds["INFLOW_GAS_COND"]["c_co2"] * material_output["inflow_mf_co2"]
            + self.common_conds["INFLOW_GAS_COND"]["c_n2"] * material_output["inflow_mf_n2"]
        )
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
            / (2 - self.common_conds["PACKED_BED_COND"]["lambda_col"])
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
            self.common_conds["INFLOW_GAS_COND"]["vi_co2"] * material_output["inflow_mf_co2"]
            + self.common_conds["INFLOW_GAS_COND"]["vi_n2"] * material_output["inflow_mf_n2"]
        )
        # プラントル数
        Pr = mu * 1000 * gas_cp / kf
        # 流入ガス体積流量 [m3/s]
        f0 = (
            (material_output["inflow_fr_co2"] + material_output["inflow_fr_n2"])
            / 1e6 / (self.common_conds["dt"] * 60)
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
        # 層伝熱係数 [W/m2/K]
        u1 = 1 / (dlat / ke + 1 / habs)

        # 内側境界からの熱流束 [J]
        if stream == 1:
            Hwin = 0
        else:
            Hwin = u1 * Ain * (temp_inside_cell - temp_now) * self.common_conds["dt"] * 60
        # 外側境界からの熱流束 [J]
        Hwout = u1 * Aout * (temp_now - temp_outside_cell) * self.common_conds["dt"] * 60
        # 下流セルへの熱流束 [J]
        if section != self.num_sec:
            Hbb = (
                u1 * Abb * (temp_now - variables["temp"][stream][section])
                * self.common_conds["dt"] * 60
            )
        else:
            Hbb = 0
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
            "hw1": hw1,
        }

        return output

    def calc_cell_heat_wall(self, section, variables, heat_output, heat_wall_output=None):
        # セクション現在温度 [℃]
        temp_now = variables["temp"][self.num_str+1][section]
        # 内側セクション温度 [℃]
        temp_inside_cell = variables["temp"][self.num_str][section]
        # 外側セクション温度 [℃]
        temp_outside_cell = self.common_conds["DRUM_WALL_COND"]["temp_outside"]
        # 下流セクション温度 [℃]
        if section != self.num_sec:
            temp_below_cell = variables["temp"][self.num_str+1][section+1]
        # 上流壁への熱流束 [J]
        if section != 1:
            Hroof = heat_wall_output["Hbb"]
        else:
            Hroof = 0
        # 内側境界からの熱流束 [J]
        Hwin = (
            heat_output["hw1"] * self.stream_conds[3]["Ain"] / self.num_sec
            * (temp_inside_cell - temp_now) * self.common_conds["dt"] * 60
        )
        # 外側境界への熱流束 [J]
        Hwout = (
            self.common_conds["DRUM_WALL_COND"]["coef_outair_heat"] * self.stream_conds[3]["Aout"]
            / self.num_sec * (temp_now - self.common_conds["DRUM_WALL_COND"]["temp_outside"])
            * self.common_conds["dt"] * 60
        )
        # 下流壁への熱流束 [J]
        if section != self.num_sec:
            Hbb = (
                self.common_conds["DRUM_WALL_COND"]["c_drumw"] * self.stream_conds[3]["Sstream"]
                * (temp_now - variables["temp"][3][section+1])
            )
        else:
            Hbb = 0
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
        }

        return output

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
            self.common_conds["DRUM_WALL_COND"]["sh_drumw"] * self.stream_conds[self.num_str+1]["Mwall"]
            * (temp_reached - temp_now)
        )

        return Hwall_heat_blc - Hwall_time