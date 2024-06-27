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

from utils import const

import warnings
warnings.simplefilter('error')


class GasAdosorption_Batch_simulator():
    """ ガス吸着モデル(バッチプロセス)を実行するクラス
    """

    def __init__(self, obs_name, cond_id):
        """ 初期化関数

        Args:
            obs_name (str): 観測値の名前(ex. obs_case1)
            cond_id (str): 実験条件の名前(ex. test1)
        """

        # クラス変数初期化
        self.cond_id = cond_id
        self.obs_name = obs_name

        # 実験条件(conditions)の読み込み
        filepath = const.CONDITIONS_DIR + self.cond_id + "/sim_conds.yml"
        with open(filepath, encoding="utf-8") as f:
            init_conditions = yaml.safe_load(f)

        self.sim_conds = {}
        self.sim_conds["inflow_gas"] = init_conditions["INFLOW_GAS_COND"]
        self.sim_conds["adsorp_cond"] = init_conditions["ADSORPTION_COND"]
        self.sim_conds["press_loss"] = init_conditions["PRESSURE_LOSS_EVAL"]
        self.initialize_init_values()

        # 観測値(data)の読み込み
        filepath = const.DATA_DIR + self.obs_name + ".csv"
        self.df_obs = pd.read_csv(filepath, index_col="timestamp")

        # その他初期化

    def execute_simulation(self, filtered_states=None, output_foldapath=None):
        """ 物理計算を通しで実行
        """

        ### ◆(1/4) 前準備 ------------------------------------------------

        # 初期値用意
        filepath = const.CONDITIONS_DIR + self.cond_id + "/assim_conds.yml"
        with open(filepath, encoding="utf-8") as f:
            assim_conds = yaml.safe_load(f)
        state_vars = assim_conds["STATE_VARS"]

        # 記録用dictの用意
        record_dict = {
            "timestamp": [],
            "吸着量傾き": [],
            "推定吸着量": [],
        }

        # 出力用フォルダの用意
        if filtered_states is None:
            mode = "simulation"
            output_foldapath\
                = const.OUTPUT_DIR + f"{self.cond_id}/{self.obs_name}/"
            if not os.path.exists(output_foldapath):
                os.makedirs(output_foldapath)
        else:
            mode = "assimilation"

        ### ◆(2/4) シミュレーション実行 --------------------------------------

        if mode == "simulation":
            logging.info("(1/3) simulating...")

        # 観測値の分だけ計算を繰り返す
        timestamp = 1
        while timestamp < len(self.df_obs):

                # 状態変数の更新 (simulationならスルー)
            if filtered_states is not None:
                for key in state_vars.keys():
                    state_vars[key] = filtered_states.loc[timestamp-1, key]

            # 1step計算実行
            absorption_amt, absorp_inclination = self.calc_adsorption_bystep(state_vars, timestamp)

            # timestamp更新
            timestamp += 1

            # 記録
            record_dict["timestamp"].append(timestamp)
            record_dict["推定吸着量"].append(absorption_amt)
            record_dict["吸着量傾き"].append(absorp_inclination)

        ### ◆(3/4) csv出力 -------------------------------------------------

        if mode == "simulation":
            logging.info("(2/3) csv output...")

        # DataFrame化
        df = pd.DataFrame()
        for key, value in record_dict.items():
            df[key] = value

        # csv出力
        filename = output_foldapath + mode + ".csv"
        df.set_index("timestamp", inplace=True)
        df.to_csv(filename)

        ### ◆(4/4) 可視化 -------------------------------------------------

        if mode == "simulation":
            logging.info("(3/3) png output...")

        plt.rcParams["font.size"] = 12

        num_plt = 1
        fig, axes = plt.subplots(1, num_plt,
                                figsize=(8 * num_plt,5),
                                tight_layout=True)
        fig.patch.set_facecolor('white')

        # 推定吸着量のみ可視化
        for i, col in enumerate(df.columns):
            if col == "推定吸着量":
                axes.plot(self.df_obs["総流入量"], label="総流入量(観測値)")
                axes.plot(df[col], label=col)
                axes.set_title(col)
                axes.grid()
                axes.legend()
                axes.set_xlabel("timestamp")

        filename = output_foldapath + mode + ".png"
        plt.savefig(filename, dpi=100)
        plt.close()

    def calc_adsorption_bystep(self, state_vars, timestamp):
        """ 1step分の物理モデル計算を実行

        Args:
            state_vars (dict): 状態変数
            timestamp (int): 現在時刻 (s)

        Return:
            absorption_amt (float): 推定吸着量 (L)
            absorp_inclination (float): 吸着量傾き (L)
        """
        # 吸着能係数（粉化度）
        adsorp_cap = state_vars["adsorp_cap"]
        # adsorp_cap = self.sim_conds["adsorp_cond"]["init_adsorp_capa"]

        # 現在圧力
        press_measure = self.df_obs.loc[timestamp, "容器圧力"]

        # 補正圧力
        porosity = self.sim_conds["adsorp_cond"]["porosity"] # 空隙率
        ptcl_size_ave = self.sim_conds["adsorp_cond"]["ptcl_size_ave"] # 平均粒子径
        ptcl_sfarea_ave = 4 * math.pi * (ptcl_size_ave / 2) ** 2 # 平均粒子表面積
        ptcl_volume_ave = 4 / 3 * math.pi * (ptcl_size_ave / 2) ** 3 # 平均粒子体積
        num_ptcl = self.sim_conds["adsorp_cond"]["volume"] * (1 - porosity) / 1000000 / ptcl_volume_ave # 充填粒子個数
        ptcl_sfarea_total = num_ptcl * ptcl_sfarea_ave # 充填粒子総表面積
        ptcl_sfarea_per_vol = ptcl_sfarea_total / self.sim_conds["adsorp_cond"]["volume"] / 1000000 # 単位体積の比表面積
        inflow_rate = ( # 流入流速
            self.sim_conds["inflow_gas"]["inflow_amt"] / 1000 / 60
            / self.sim_conds["adsorp_cond"]["fill_height"] / porosity
        )
        press_specific = self.sim_conds["press_loss"]["press_specific"] # 容器固有差圧
        press_loss = ( # 圧力損失
            (150 / 36 * (1 - porosity) ** 2 * ptcl_sfarea_per_vol ** 2 * self.sim_conds["inflow_gas"]["viscosity"]
             * inflow_rate / porosity ** 3 + 1.75 / 6 * (1 - porosity) * ptcl_sfarea_per_vol
             * self.sim_conds["inflow_gas"]["density"] * inflow_rate ** 2 / porosity ** 3)
            * self.sim_conds["adsorp_cond"]["fill_height"] / 1000 + press_specific
        ) / 1000
        press_correct = press_measure - press_loss

        # 吸着量傾き
        temp = self.df_obs.loc[timestamp, "容器平均温度"] # 現在温度
        absorp_inclination = (
            0.00121528 * (temp + 273.15) ** 2
            - 0.772896664 * (temp + 273.15) +123.4683338758
        )
        # absorp_inclination = state_vars["absorp_inclination"]

        # 推定吸着量
        absorption_amt = (
            absorp_inclination * adsorp_cap * press_correct * 1000
            * self.sim_conds["adsorp_cond"]["weight"]
            - self.sim_conds["adsorp_cond"]["init_adsorp_amt"]
        )

        return absorption_amt, absorp_inclination

    def initialize_init_values(self):
        """ 初期値から計算される固定値を計算するメソッド
        """
        # 充填吸着材条件
        adsorp_cond = self.sim_conds["adsorp_cond"]
        adsorp_cond["cross_sect_area"] = ( # 容器断面積
            math.pi * (adsorp_cond["length"] / 1000 / 2) ** 2
        )
        adsorp_cond["fill_height"] = ( # 充填高さ
            adsorp_cond["volume"] / 1000000 / adsorp_cond["cross_sect_area"]
        )
        adsorp_cond["weight"] = ( # 充填重量
            adsorp_cond["volume"] * adsorp_cond["bulk_density"] / 1000
        )
        adsorp_cond["heat_capacity"] = ( # 熱容量
            adsorp_cond["weight"] * adsorp_cond["spec_heat"]
        )
        adsorp_cond["init_adsorp_amt"] = ( # 初期吸着量
            (
                0.00121528 * (adsorp_cond["init_temp"] + 273.15) ** 2 - 0.772896664
                * (adsorp_cond["init_temp"] + 273.15) + 123.4683338758
            ) * adsorp_cond["init_pressure"] * 1000 * adsorp_cond["weight"] * 0.95
        )