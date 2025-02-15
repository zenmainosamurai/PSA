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
from simulator import GasAdosorption_Breakthrough_simulator


class Assimilator():

    def __init__(self, cond_id):
        """ 初期定義

            assim_conds (str): データ同化の実験条件名
            mode (str): "simulation" or "assimilation"
        """
        # クラス変数初期化
        self.cond_id = cond_id
        # 出力先フォルダ
        self.output_foldapath_csv = const.OUTPUT_DIR + f"{self.cond_id}/assimilation/csv/"
        self.output_foldapath_png = const.OUTPUT_DIR + f"{self.cond_id}/assimilation/png/"
        os.makedirs(self.output_foldapath_csv, exist_ok=True)
        os.makedirs(self.output_foldapath_png, exist_ok=True)
        # 物理モデルの初期化
        self.simulator = GasAdosorption_Breakthrough_simulator(self.cond_id)
        # 観測値の読み込み
        self.df_obs = self._init_obs()
        # ukfの初期化
        filepath = const.CONDITIONS_DIR + self.cond_id + "/assim_conds.yml"
        with open(filepath, encoding="utf-8") as f:
            self.assim_conds = yaml.safe_load(f)
        self.dt = self.simulator.sim_conds["dt"]
        self.ukf = self._init_ukf()
        self.obs_index = 0
        # 対数尤度記録用リスト
        self.likelihood_list = []
        # 追加の初期化

    def execute_assimilation(self):
        """ データ同化実行関数

            execute_assimilation() ← filtering()
        """

        ### ◆(1/5) filtering------------------------------------

        print("(1/5) filtering ...")
        # 記録用配列
        filtered_x = self.ukf.x.reshape((1,-1))
        filtered_p = self.ukf.P.reshape((1,len(self.ukf.x),-1))
        # データ同化実施
        timestamp = 0
        _time_limit = 86400 # 1日(s)
        while timestamp < _time_limit:
            # filtering
            self.filtering(timestamp)
            # 記録
            filtered_x = np.concatenate([filtered_x, self.ukf.x.reshape((1,-1))])
            filtered_p = np.concatenate([filtered_p,
                                            self.ukf.P.reshape((1,len(self.ukf.x),-1))])
            # 時刻更新
            timestamp += self.dt
        # 重複削除
        filtered_x = filtered_x[1:]
        filtered_p = filtered_p[1:]

        ### ◆(2/5) smoothing-------------------------------------

        print("(2/5) smoothing ...")
        sm_x, sm_P, _ = self.ukf.rts_smoother(filtered_x, filtered_p)

        ### ◆(3/5) csv出力---------------------------------------

        print("(3/5) csv output ...")
        # DataFrame化
        columns = self.assim_conds["STATE_VARS"].keys()
        index_ = self.df_obs.index
        df_filtered_x = pd.DataFrame(filtered_x, columns=columns, index=index_)
        df_filtered_p = pd.DataFrame([list(val.diagonal()) for val in filtered_p], columns=columns, index=index_)
        df_smoothed_x = pd.DataFrame(sm_x, columns=columns, index=index_)
        df_smoothed_p = pd.DataFrame([list(val.diagonal()) for val in sm_P], columns=columns, index=index_)
        # df_likelihood = pd.DataFrame(self.likelihood_list, columns=["log-likelihoods"], index=index_)
        # csv出力
        df_filtered_x.to_csv(self.output_foldapath_csv + "filtered_states.csv")
        df_filtered_p.to_csv(self.output_foldapath_csv + "filtered_covariance.csv")
        df_smoothed_x.to_csv(self.output_foldapath_csv + "smoothed_states.csv")
        df_smoothed_p.to_csv(self.output_foldapath_csv + "smoothed_covariance.csv")
        # df_likelihood.to_csv(self.output_foldapath_csv + "log_likelihoods.csv")

        ### ◆(4/5) 可視化 ---------------------------------------

        print("(4/5) png output ...")
        plt.rcParams["font.size"] = 12

        # filtered_x
        fig = plt.figure(figsize=(20,4), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
            plt.plot(df_filtered_x[key], label=key)
        plt.xlabel("timestamp")
        plt.legend()
        plt.grid()
        plt.title("filtered_state_vars")
        plt.savefig(self.output_foldapath_png + "filtered_states.png", dpi=100)
        plt.close()

        # filtered_p
        fig = plt.figure(figsize=(20,4), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
            plt.plot(df_filtered_p.loc[datetime.datetime(2019,3,17,0,30):, key], label=key)
        plt.xlabel("timestamp")
        plt.legend()
        plt.grid()
        plt.title("filtered_state_vars")
        plt.savefig(self.output_foldapath_png + "filtered_covariance.png", dpi=100)
        plt.close()

        # smoothed_x
        fig = plt.figure(figsize=(20,4), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
            plt.plot(df_smoothed_x[key], label=key)
        plt.xlabel("timestamp")
        plt.legend()
        plt.grid()
        plt.title("smoothed_state_vars")
        plt.savefig(self.output_foldapath_png + "smoothed_states.png", dpi=100)
        plt.close()

        # smoothed_p
        fig = plt.figure(figsize=(20,4), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
            plt.plot(df_smoothed_p.loc[datetime.datetime(2019,3,17,0,30):, key], label=key)
        plt.xlabel("timestamp")
        plt.legend()
        plt.grid()
        plt.title("smoothed_state_vars")
        plt.savefig(self.output_foldapath_png + "smoothed_covariance.png", dpi=100)
        plt.close()

        # log-likelihood出力
        # fig = plt.figure(figsize=(15,8), tight_layout=True)
        # fig.patch.set_facecolor('white')
        # plt.plot(self.likelihood_list)
        # plt.title(f"likelihoods ({sum(self.likelihood_list)})")
        # plt.xlabel("timestamp")
        # plt.grid()
        # plt.savefig(self.output_foldapath_png + "log-likelihoods.png", dpi=100)
        # plt.close()

        ### ◆(5/5) resimulation----------------------------------

        print("(5/5) re-simulation ...")
        output_foldapath = const.OUTPUT_DIR + f"{self.cond_id}/simulation_assim/"
        self.simulator.execute_simulation(filtered_x=df_smoothed_x,
                                          assim_conds=self.assim_conds,
                                          output_foldapath=output_foldapath)

    def filtering(self, timestamp):
        """ フィルタリング実行関数

            filtering() ← _fx(),_hx()
        """
        # 状態変数の予測
        self.ukf.predict()
        # 観測値がある場合はフィルタリング
        diff_time = (self.df_obs.index[self.obs_index] - self.df_obs.index[0]).total_seconds()
        if timestamp >= diff_time:
            # 観測値の抽出
            obs_values = self.df_obs.iloc[self.obs_index,:]
            # filtering
            _timestamp = self.df_obs.index[self.obs_index]
            self.ukf.update(np.array(obs_values), gx_args={"timestamp": _timestamp}, hx_args={})
            self.obs_index += 1
            # マイナスチェック
            # NOTE: 必要な場合のみ
            self.ukf.x = np.array([1e-8 if val < 0 else val for val in self.ukf.x])
        # ※対数尤度の記録
        # self.likelihood_list.append(self.ukf.log_likelihood)

    def _fx(self, z, dt):
        """ 状態方程式

            Args:
                z (ndarray): 前回時点の状態変数
                dt (float): タイムステップ

            Return:
                z (ndarray): 現在の状態変数の予測値
        """
        # 状態変数の予測（マイナスチェック）
        z = np.array([1e-8 if val < 0 else val for val in z])
        return z

    def _gx(self, z, s_bfr, timestamp):
        """ 中間方程式
            現在の状態変数zと1時点前の中間変数から、現在の中間変数を求める
            NOTE: 状態変数だが、補正対象にしたくない変数を中間変数とする

            Args:
                z (ndarray): 現在の状態変数
                s_bfr (ndarray): 1時点前の中間変数

            Return:
                s (dict): 現在の中間変数と観測変数
        """
        ### ◆前準備----------------------------------

        ### ◆前回時点の状態変数、中間変数の復元----------------------------------

        # 次時点の状態変数(z)
        # NOTE: models.pyの入力形式（辞書）に戻す
        state_vars = {}
        for i, key in enumerate(self.assim_conds["STATE_VARS"].keys()):
            state_vars[key] = z[i]
        # 現時点の中間変数(s_bfr)
        # NOTE: models.pyの入力形式（辞書）に戻す
        num_grate = self.simulator.sim_conds["NUM_GRATE"]
        middle_vars = {}
        for i, var in enumerate(self.assim_conds["MID_VARS"]):
            middle_vars[var] = s_bfr[i*num_grate: (i+1)*num_grate]

        ### ◆中間変数の更新--------------------------------------

        # 物理計算実行
        middle_vars, all_outputs = self.simulator.calc_model_one_step(timestamp, middle_vars, state_vars)
        # 出力
        s = np.empty(0)
        for var in self.assim_conds["MID_VARS"]: # 中間変数
            s = np.concatenate([s, all_outputs[var]])
        for var in self.assim_conds["OBS_VARS"]: # 観測変数
            s = np.concatenate([s, [all_outputs[var]]])

        return s

    def _hx(self, s):
        """ 観測方程式

        Args:
            s (ndarray): 中間変数

        Return:
            x (ndarray): 観測変数
        """
        # 観測変数の復元
        mid_var_index = self.assim_conds["MID_VARS_INDEX"]
        x = s[mid_var_index:]        

        return x

    def _init_obs(self):
        """ 観測値のDataFrameを用意する

        Returns:
            pd.DataFrame: 観測値
        """
        # ガス発生量
        gass_entropy = pd.read_csv(const.DATA_DIR + "gass_entropy/20190317_fujimi_gass_entropy.csv",
                                   index_col=0, parse_dates=True)[["排ガス顕熱[MJ/10s]"]]
        # 揮発分放出量最大位置
        # idx_max_vol = self.simulator.df_temperature_1min_idxmax.values
        # 燃え切り点
        idx_burn_out = self.simulator.flame_positions.values
        # index
        index_ = self.simulator.df_weight_1min.index
        # 結合
        # df_obs = pd.DataFrame(np.stack([gass_entropy], 1),
        #                       columns=["gass_entropy"],
        #                       index=index_)

        return gass_entropy

    def _init_ukf(self):
        """ ukfを初期化する関数

            ukfの初期化に必要となる各種パラメータを読み込み、
            ukfをインスタンス化して返す。

            Returns:
                ukf (instance): ukfのインスタンス

        """
        ### ◆初期化-----------------------------------

        # 状態変数
        _state_vars = self.assim_conds["STATE_VARS"]
        # 観測変数
        _obs_vars = self.assim_conds["OBS_VARS"]
        # kfパラメータ
        _kf_params = self.assim_conds["KF_PARAMS"]
        dim_x = len(_state_vars)
        dim_z = len(_obs_vars)
        sigma_points = JulierSigmaPoints(dim_x) # シグマ点列生成アルゴリズム

        ### ◆ukfの初期化---------------------------------------

        ukf = CustomUKF(dim_x=dim_x, dim_z=dim_z, dt=self.dt,
                        fx=self._fx, hx=self._hx, gx=self._gx, points=sigma_points)
        ukf.x = np.array(list(_state_vars.values())) # 状態変数
        ukf.P = np.diag(_kf_params["p0"]) # 分散共分散行列
        ukf.Q = np.diag(_kf_params["Q"]) # 状態誤差
        ukf.R = np.diag(_kf_params["R"]) # 観測誤差

        ### ◆中間変数の初期化----------------------------------

        # 状態変数を初期化
        _, all_outputs = self.simulator._init_variables()
        # 中間変数として初期化
        # NOTE: 中間方程式に渡すには要flatten
        # NOTE: 順番はassim_condsに従う
        s = np.empty(0)
        for var in self.assim_conds["MID_VARS"]: # 中間変数
            s = np.concatenate([s, all_outputs[var]])
        for var in self.assim_conds["OBS_VARS"]: # 観測変数も追加
            s = np.concatenate([s, [all_outputs[var]]])
        ukf.initialize_states(s)

        return ukf