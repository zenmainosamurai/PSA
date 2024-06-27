import os
import sys
import yaml
import datetime
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from filterpy.kalman import UnscentedKalmanFilter, JulierSigmaPoints
from filterpy.common import Q_discrete_white_noise

from utils import const
from utils.custom_filter import CustomUKF
from models import GasAdosorption_Batch_simulator


class Assimilator():


    def __init__(self, obs_name, cond_id, mode):
        """ 初期定義

            obs_name (str): 実験・観測データ名
            assim_conds (str): データ同化の実験条件名
            mode (str): "simulation" or "assimilation"

        """
        # クラス変数初期化
        self.obs_name = obs_name
        self.cond_id = cond_id
        self.mode = mode

        # 出力先フォルダ
        self.output_foldapath = const.OUTPUT_DIR + f"{self.cond_id}/{self.obs_name}/"
        if not os.path.exists(self.output_foldapath):
            os.makedirs(self.output_foldapath)

        # logging設定
        if self.mode == "simulation":
            filename = self.output_foldapath + "sim_logging.log"
        elif self.mode == "assimilation":
            filename = self.output_foldapath + "assim_logging.log"

        logging.basicConfig(
            filename=filename,
            filemode="w",
            level=logging.DEBUG,
            format="[%(levelname)s] %(message)s",
            force=True
        )
        logging.getLogger('matplotlib.font_manager').disabled = True # matplotlibの冗長なlogを除去
        logging.getLogger("PIL.PngImagePlugin").disabled = True

        # 物理モデルの初期化
        self.simulator = GasAdosorption_Batch_simulator(self.obs_name, self.cond_id)

        # ukfの初期化
        filepath = const.CONDITIONS_DIR + self.cond_id + "/assim_conds.yml"
        with open(filepath, encoding="utf-8") as f:
            self.assim_conds = yaml.safe_load(f)
        self.ukf = self._init_ukf()

        # 対数尤度記録用リスト
        self.likelihood_list = []

    def _init_ukf(self):
        """ ukfを初期化する関数

            ukfの初期化に必要となる各種パラメータを読み込み、
            ukfをインスタンス化して返す。

            Args:
                assim_conds_name (str): データ同化の実験条件名

            Returns:
                ukf (instance): ukfのインスタンス

        """
        ### ◆初期値の読み込み-----------------------------------

        state_vars = self.assim_conds["STATE_VARS"]             # 状態変数
        obs_vars = self.assim_conds["OBS_VARS"]                 # 観測変数
        kf_params = self.assim_conds["KF_PARAMS"]               # パラメータ初期値

        dim_x = len(state_vars)
        dim_z = len(obs_vars)
        dt = kf_params["dt"]

        sigma_points = JulierSigmaPoints(dim_x) # シグマ点列生成アルゴリズム

        ### ◆ukfの初期化---------------------------------------

        ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt,
                                    fx=self._fx, hx=self._hx, points=sigma_points)

        # 状態変数のパラメータを設定
        ukf.x = np.array(list(state_vars.values()))
        ukf.P = np.diag(kf_params["p0"]) # 状態の予測誤差の初期値
        ukf.Q = np.diag(kf_params["Q"]) # 状態の予測誤差
        ukf.R = np.diag(kf_params["R"]) # 観測値の予測誤差

        return ukf

    def execute_assimilation(self):
        """ データ同化実行関数

            execute_assimilation() ← _filtering()
        """

        ### ◆ simulation----------------------------------

        if self.mode == "simulation":

            logging.info("smulation start!")
            self.simulator.execute_simulation(output_foldapath=self.output_foldapath)
            logging.info("complete!!!")

        ### ◆(1/5) filtering------------------------------------

        elif self.mode == "assimilation":

            logging.info("assmilation start!")
            logging.info("(1/5) filtering...")

            # 記録用配列
            filtered_x = self.ukf.x.reshape((1,-1))
            filtered_p = self.ukf.P.reshape((1,len(self.ukf.x),-1))

            # データ同化実施
            timestamp = 1
            while timestamp < len(self.simulator.df_obs): # 観測値の数だけ繰り返す

                # filtering
                self._filtering(timestamp)
                # 記録
                filtered_x = np.concatenate([filtered_x, self.ukf.x.reshape((1,-1))])
                filtered_p = np.concatenate([filtered_p,
                                            self.ukf.P.reshape((1,len(self.ukf.x),-1))])
                # timestamp更新
                timestamp += 1

            # 重複削除
            filtered_x = filtered_x[1:]
            filtered_p = filtered_p[1:]

            ### ◆(2/5) smoothing-------------------------------------

            logging.info("(2/5) smoothing...")
            sm_x, _, _ = self.ukf.rts_smoother(filtered_x, filtered_p)

            ### ◆(3/5) csv出力---------------------------------------

            logging.info("(3/5) csv output...")
            # DataFrame化
            columns = self.assim_conds["STATE_VARS"].keys()
            df_filtered = pd.DataFrame(filtered_x, columns=columns)
            df_smoothed = pd.DataFrame(sm_x, columns=columns)
            df_likelihood = pd.DataFrame(self.likelihood_list, columns=["log-likelihoods"])
            # csv出力
            df_filtered.to_csv(self.output_foldapath + "filtered_states.csv", index=False)
            df_smoothed.to_csv(self.output_foldapath + "smoothed_states.csv", index=False)
            df_likelihood.to_csv(self.output_foldapath + "log_likelihoods.csv", index=False)

            ### ◆(4/5) 可視化 ---------------------------------------

            logging.info("(4/5) png output...")
            plt.rcParams["font.size"] = 12

            # filtered_x
            fig, axis = plt.subplots(1, 1,
                                    figsize=(8,5),
                                    tight_layout=True)
            fig.patch.set_facecolor('white')
            for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
                axis.plot(df_filtered[key])
                axis.set_title(key)
                axis.grid()
            plt.savefig(self.output_foldapath + "filtered_states.png", dpi=100)
            plt.close()

            # smoothed_x
            fig, axis = plt.subplots(1, 1,
                                    figsize=(8,5),
                                    tight_layout=True)
            fig.patch.set_facecolor('white')
            for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
                axis.plot(df_smoothed[key])
                axis.set_title(key)
                axis.set_xlabel("timestamp")
                axis.grid()
            plt.savefig(self.output_foldapath + "smoothed_states.png", dpi=100)
            plt.close()

            # log-likelihood出力
            fig = plt.figure(figsize=(15,8), tight_layout=True)
            fig.patch.set_facecolor('white')
            plt.plot(self.likelihood_list)
            plt.title(f"likelihoods ({sum(self.likelihood_list)})")
            plt.xlabel("timestamp")
            plt.grid()
            plt.savefig(self.output_foldapath + "log-likelihoods.png", dpi=100)
            plt.close()

            ### ◆(5/5) resimulation----------------------------------

            logging.info("(5/5) re-simulation...")
            self.simulator.execute_simulation(df_smoothed, self.output_foldapath)
            # self.simulator.execute_simulation(df_filtered, self.output_foldapath)
            logging.info("complete!!!")

    def _filtering(self, timestamp):
        """ フィルタリング実行関数

            _filtering() ← _fx(),_hx()
        """

        # 状態変数の予測
        self.ukf.predict()

        # 観測値の抽出
        obs_values = self.simulator.df_obs.loc[timestamp, "総流入量"]

        # filtering
        self.ukf.update(np.array(obs_values), hx_args={"timestamp": timestamp})

        # ※対数尤度の記録
        self.likelihood_list.append(self.ukf.log_likelihood)

    def _fx(self, z, dt):
        """ 状態方程式

            物理計算を回し、前回時点の状態変数から次の状態変数を予測する

            Args:
                z (ndarray): 前回時点の状態変数
                dt (float): time rate

            Return:
                z (ndarray): 状態変数の予測値
        """

        # 状態変数の予測（恒等関数）

        return z

    def _hx(self, z, hx_args):
        """ 観測方程式

        Args:
            z (ndarray): 状態変数の予測値（粉化度、総推定吸着量）
            hx_args (dict): 物理計算の引数等

        Return:
            x (ndarray): 観測変数
        """

        ### 物理モデル1step分 ----------------------------------------

        # 状態変数抽出,辞書化
        state_vars = {}
        for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
            state_vars[key] = z[i]

        # 物理計算実行
        absorption_amt, _ = \
            self.simulator.calc_adsorption_bystep(state_vars, hx_args["timestamp"])

        ### 観測変数抽出 ----------------------------------------

        x = []

        # 1. 推定吸着量
        x.append(absorption_amt)
        
        x = np.array(x)

        return x