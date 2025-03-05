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
        self.num_tower = self.simulator.sim_conds[1]["NUM_TOWER"]
        self.num_str = self.simulator.sim_conds[1]["CELL_SPLIT"]["num_str"]
        self.num_sec = self.simulator.sim_conds[1]["CELL_SPLIT"]["num_sec"]
        # 観測値の読み込み
        self.df_obs = self._init_obs_data()
        # ukfの初期化
        filepath = const.CONDITIONS_DIR + self.cond_id + "/assim_conds.yml"
        with open(filepath, encoding="utf-8") as f:
            self.assim_conds = yaml.safe_load(f)
        self.dt = self.simulator.sim_conds[1]["dt"]
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
        timestamp_list = []
        _time_limit = self.simulator.df_operation["手動終了時刻"].iloc[-1] # 手動終了時刻の最後
        while timestamp < _time_limit:
            # filtering
            self.filtering(timestamp)
            # 記録
            filtered_x = np.concatenate([filtered_x, self.ukf.x.reshape((1,-1))])
            filtered_p = np.concatenate([filtered_p,
                                            self.ukf.P.reshape((1,len(self.ukf.x),-1))])
            # 時刻更新
            timestamp_list.append(timestamp)
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
        columns = []
        for tower_num, tgt_tower_params in self.assim_conds["STATE_VARS"].items():
            for cond_category, tgt_conds in tgt_tower_params.items():
                for cond_name in tgt_conds.keys():
                    columns.append(f"T{tower_num}_{cond_name}")
        index_ = timestamp_list
        df_filtered_x = pd.DataFrame(filtered_x, columns=columns, index=index_)
        df_filtered_p = pd.DataFrame([list(val.diagonal()) for val in filtered_p], columns=columns, index=index_)
        df_smoothed_x = pd.DataFrame(sm_x, columns=columns, index=index_)
        df_smoothed_p = pd.DataFrame([list(val.diagonal()) for val in sm_P], columns=columns, index=index_)
        # csv出力
        df_filtered_x.to_csv(self.output_foldapath_csv + "filtered_states.csv")
        df_filtered_p.to_csv(self.output_foldapath_csv + "filtered_covariance.csv")
        df_smoothed_x.to_csv(self.output_foldapath_csv + "smoothed_states.csv")
        df_smoothed_p.to_csv(self.output_foldapath_csv + "smoothed_covariance.csv")

        ### ◆(4/5) 可視化 ---------------------------------------

        print("(4/5) png output ...")
        self._plot_outputs(df_filtered_x=df_filtered_x,
                           df_filtered_p=df_filtered_p,
                           df_smoothed_x=df_smoothed_x,
                           df_smoothed_p=df_smoothed_p)

        ### ◆(5/5) resimulation----------------------------------

        df_filtered_x = pd.read_csv(self.output_foldapath_csv + "filtered_states.csv", index_col=0)
        print("(5/5) re-simulation ...")
        output_foldapath = const.OUTPUT_DIR + f"{self.cond_id}/simulation_assim/"
        self.simulator.execute_simulation(filtered_states=df_filtered_x,
                                          output_foldapath=output_foldapath)

    def filtering(self, timestamp):
        """ フィルタリング実行関数

            filtering() ← _fx(),_hx()
        """
        # 状態変数の予測
        self.ukf.predict()
        # 観測値がある場合はフィルタリング
        diff_time = self.df_obs.index[self.obs_index] - self.df_obs.index[0]
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

    def _fx(self, x, dt):
        """ 状態方程式

            Args:
                x (ndarray): 前回時点の状態変数
                dt (float): タイムステップ

            Return:
                x (ndarray): 現在の状態変数の予測値
        """
        # 状態変数の予測（マイナスチェック）
        x = np.array([1e-8 if val < 0 else val for val in x])
        return x

    def _gx(self, x, s_bfr, timestamp):
        """ 中間方程式
            現在の状態変数xと1時点前の中間変数から、現在の中間変数を求める
            NOTE: 状態変数だが、補正対象にしたくない変数を中間変数とする

            Args:
                x (ndarray): 現在の状態変数
                s_bfr (ndarray): 1時点前の中間変数

            Return:
                s (dict): 現在の中間変数と観測変数
        """
        ### ◆前準備 ----------------------------------

        # 状態変数(x)をシミュレータのパラメータに上書き
        i = 0
        for _tower_num, tgt_tower_params in self.assim_conds["STATE_VARS"].items():
            for cond_category, tgt_conds in tgt_tower_params.items():
                for cond_name in tgt_conds.keys():
                    self.simulator.sim_conds[_tower_num][cond_category][cond_name] = x[i]
                    i += 1
        # 中間変数(s_bfr)をシミュレータのvariablesに上書き
        middle_vars = self.__array_to_dict_middle_variables(s_bfr=s_bfr)

        ### ◆中間変数の更新 --------------------------------------
        # NOTE: 手動終了時刻採用が前提

        # 中間変数(圧力)の上書き
        _tgt_index = self.df_obs.index[np.abs(self.df_obs.index - (timestamp)).argmin()]
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["total_press"] = self.simulator.df_obs.loc[_tgt_index, f"T{_tower_num}_press"]
        # プロセス番号抽出
        p = 1 + sum([1 if timestamp > x else 0 for x in self.simulator.df_operation["手動終了時刻"]])
        # 各塔の稼働モード抽出
        mode_list = list(self.simulator.df_operation.loc[p, ["塔1", "塔2", "塔3"]])
        # 各塔の吸着計算実施
        middle_vars, all_outputs = self.simulator.calc_adsorption_mode_list(mode_list,
                                                                            middle_vars,
                                                                            timestamp)
        ### ◆出力 --------------------------------

        # 計算結果から中間変数と観測変数を抽出
        s = self.__dict_to_array_middle_obs_variables(middle_vars=middle_vars)

        return s

    def _hx(self, s):
        """ 観測方程式

        Args:
            s (ndarray): 中間変数

        Return:
            z (ndarray): 観測変数
        """
        # array形式から辞書形式に変換
        middle_vars = self.__array_to_dict_middle_variables(s)
        # 観測変数（温度）の抽出
        z = []
        for _tower_num in range(1, 1+self.num_tower):
            for section in self.simulator.sim_conds[1]["LOC_CENCER_SECTION"].values():
                z.append(middle_vars[_tower_num]["temp"][1][section])
        z = np.array(z)

        return z

    def _init_ukf(self):
        """ ukfを初期化する関数

            ukfの初期化に必要となる各種パラメータを読み込み、
            ukfをインスタンス化して返す。

            Returns:
                ukf (instance): ukfのインスタンス

        """
        ### ◆初期化-----------------------------------

        # 状態変数
        _state_vars = []
        for tower_num, tgt_tower_params in self.assim_conds["STATE_VARS"].items():
            for cond_category, tgt_conds in tgt_tower_params.items():
                for cond_name, value in tgt_conds.items():
                    _state_vars.append(value)
        # kfパラメータ
        _kf_params = self.assim_conds["KF_PARAMS"]
        dim_x = len(_state_vars)
        dim_z = sum(self.assim_conds["OBS_VARS"].values())
        sigma_points = JulierSigmaPoints(dim_x) # シグマ点列生成アルゴリズム

        ### ◆ukfの初期化---------------------------------------

        ukf = CustomUKF(dim_x=dim_x, dim_z=dim_z, dt=self.dt,
                        fx=self._fx, hx=self._hx, gx=self._gx, points=sigma_points)
        ukf.x = np.array(_state_vars) # 状態変数
        ukf.P = np.diag(_kf_params["p0"]) # 分散共分散行列
        ukf.Q = np.diag(_kf_params["Q"]) # 状態誤差
        ukf.R = np.diag(_kf_params["R"]) # 観測誤差

        ### ◆中間変数の初期化----------------------------------

        # 状態変数(モデル)を初期化
        middle_vars = self.simulator._init_variables()
        # 中間変数として初期化
        s = self.__dict_to_array_middle_obs_variables(middle_vars)
        ukf.initialize_states(s)

        return ukf

    def _init_obs_data(self):
        """ データ同化用の観測値作成

        Returns:
            pd.DataFrame: 観測値（温度）
        """
        # 温度のみ抽出
        tgt_cols = [f"T{_tower_num}_temp_{_section}" for _tower_num in range(1, 1+self.num_tower) for _section in [1,2,3]]
        df_obs = self.simulator.df_obs[tgt_cols]

        return df_obs

    def _plot_outputs(self, df_filtered_x, df_filtered_p, df_smoothed_x, df_smoothed_p):
        """ データ同化で得られた状態変数と分散共分散行列を可視化

        Args:
            df_filtered_x (pd.DataFrame): 状態変数の推移
            df_filtered_p (pd.DataFrame): 分散共分散行列(対角成分)
            df_smoothed_x (pd.DataFrame): 平滑化した状態変数の推移
            df_smoothed_p (pd.DataFrame): 平滑化した分散共分散行列(対角成分)の推移
        """
        num_vars = len(df_filtered_x.columns)
        num_row = int(np.ceil(num_vars/4))
        plt.rcParams["font.size"] = 20

        # filtered_x
        fig = plt.figure(figsize=(26, 5.5*num_row), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, col in enumerate(df_filtered_x.columns):
            plt.subplot(num_row, 4, i+1)
            plt.plot(df_filtered_x[col])
            plt.xlabel("timestamp")
            plt.grid()
            plt.title(col)
        plt.savefig(self.output_foldapath_png + "filtered_states.png", dpi=100)
        plt.close()

        # filtered_p
        fig = plt.figure(figsize=(24, 5.5*num_row), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, col in enumerate(df_filtered_x.columns):
            plt.subplot(num_row, 4, i+1)
            plt.plot(df_filtered_p[col])
            plt.xlabel("timestamp")
            plt.grid()
            plt.title(col)
        plt.savefig(self.output_foldapath_png + "filtered_covariance_diag.png", dpi=100)
        plt.close()

        # smoothed_x
        fig = plt.figure(figsize=(24, 5.5*num_row), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, col in enumerate(df_filtered_x.columns):
            plt.subplot(num_row, 4, i+1)
            plt.plot(df_smoothed_x[col])
            plt.xlabel("timestamp")
            plt.grid()
            plt.title(col)
        plt.savefig(self.output_foldapath_png + "smootheed_states.png", dpi=100)
        plt.close()

        # smoothed_p
        fig = plt.figure(figsize=(24, 5.5*num_row), tight_layout=True)
        fig.patch.set_facecolor('white')
        for i, col in enumerate(df_filtered_x.columns):
            plt.subplot(num_row, 4, i+1)
            plt.plot(df_smoothed_p[col])
            plt.xlabel("timestamp")
            plt.grid()
            plt.title(col)
        plt.savefig(self.output_foldapath_png + "smootheed_covariance_diag.png", dpi=100)
        plt.close()

    def __array_to_dict_middle_variables(self, s_bfr):
        """ 前時刻の中間変数(array)を辞書形式(dict)に変換

        Args:
            num_tower (int): 吸着塔の数
            num_str (int): ストリームの数
            num_sec (int): セクションの数
            s_bfr (np.array): 前時刻の中間変数
        
        Return:
            middle_vars (dict): 前時刻の中間変数を辞書形式に変換したもの
        """
        middle_vars = {}
        i = 0
        # 0. 塔ごとに設定
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num] = {}
        # 1. 温度
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["temp"] = {}
            for stream in range(1, 1+self.num_str):
                middle_vars[_tower_num]["temp"][stream] = {}
                for section in range(1, 1+self.num_sec):
                    middle_vars[_tower_num]["temp"][stream][section] = s_bfr[i]
                    i+=1
        # 2. 壁面温度
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["temp_wall"] = {}
            for section in range(1, 1+self.num_sec):
                middle_vars[_tower_num]["temp_wall"][section] = s_bfr[i]
                i+=1
        # 3. 上下蓋の温度
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["temp_lid"] = {}
            for position in ["up", "down"]:
                middle_vars[_tower_num]["temp_lid"][position] = s_bfr[i]
                i+=1
        # 4. 既存吸着量
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["adsorp_amt"] = {}
            for stream in range(1, 1+self.num_str):
                middle_vars[_tower_num]["adsorp_amt"][stream] = {}
                for section in range(1, 1+self.num_sec):
                    middle_vars[_tower_num]["adsorp_amt"][stream][section] = s_bfr[i]
                    i+=1
        # 5. モル分率(co2)
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["mf_co2"] = {}
            for stream in range(1, 1+self.num_str):
                middle_vars[_tower_num]["mf_co2"][stream] = {}
                for section in range(1, 1+self.num_sec):
                    middle_vars[_tower_num]["mf_co2"][stream][section] = s_bfr[i]
                    i+=1
        # 6. モル分率(n2)
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["mf_n2"] = {}
            for stream in range(1, 1+self.num_str):
                middle_vars[_tower_num]["mf_n2"][stream] = {}
                for section in range(1, 1+self.num_sec):
                    middle_vars[_tower_num]["mf_n2"][stream][section] = s_bfr[i]
                    i+=1
        # 7. 層伝熱係数
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["heat_t_coef"] = {}
            for stream in range(1, 1+self.num_str):
                middle_vars[_tower_num]["heat_t_coef"][stream] = {}
                for section in range(1, 1+self.num_sec):
                    middle_vars[_tower_num]["heat_t_coef"][stream][section] = s_bfr[i]
                    i+=1
        # 8. 壁―層伝熱係数
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["heat_t_coef_wall"] = {}
            for stream in range(1, 1+self.num_str):
                middle_vars[_tower_num]["heat_t_coef_wall"][stream] = {}
                for section in range(1, 1+self.num_sec):
                    middle_vars[_tower_num]["heat_t_coef_wall"][stream][section] = s_bfr[i]
                    i+=1
        # 9. 全圧
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["total_press"] = s_bfr[i]
            i+=1
        # 10. CO2,N2回収量
        for _tower_num in range(1, 1+self.num_tower):
            middle_vars[_tower_num]["vacuum_amt_co2"] = s_bfr[i]
            middle_vars[_tower_num]["vacuum_amt_n2"] = s_bfr[i]
            i+=1

        return middle_vars

    def __dict_to_array_middle_obs_variables(self, middle_vars):
        """ 辞書形式の中間変数と観測変数をarray形式に変換
            NOTE: _array_to_dict関数と順番を合わせる

        Args:
            num_tower (int): 吸着塔の数
            num_str (int): ストリームの数
            num_sec (int): セクションの数
            s_bfr (np.array): 前時刻の中間変数
        
        Return:
            middle_vars (dict): 前時刻の中間変数を辞書形式に変換したもの
        """
        ### 中間変数 -----------------------

        s = []
        # 1. 温度
        for _tower_num in range(1, 1+self.num_tower):
            for stream in range(1, 1+self.num_str):
                for section in range(1, 1+self.num_sec):
                    s.append(middle_vars[_tower_num]["temp"][stream][section])
        # 2. 壁面温度
        for _tower_num in range(1, 1+self.num_tower):
            for section in range(1, 1+self.num_sec):
                s.append(middle_vars[_tower_num]["temp_wall"][section])
        # 3. 上下蓋の温度
        for _tower_num in range(1, 1+self.num_tower):
            for position in ["up", "down"]:
                s.append(middle_vars[_tower_num]["temp_lid"][position])
        # 4. 既存吸着量
        for _tower_num in range(1, 1+self.num_tower):
            for stream in range(1, 1+self.num_str):
                for section in range(1, 1+self.num_sec):
                    s.append(middle_vars[_tower_num]["adsorp_amt"][stream][section])
        # 5. モル分率(co2)
        for _tower_num in range(1, 1+self.num_tower):
            for stream in range(1, 1+self.num_str):
                for section in range(1, 1+self.num_sec):
                    s.append(middle_vars[_tower_num]["mf_co2"][stream][section])
        # 6. モル分率(n2)
        for _tower_num in range(1, 1+self.num_tower):
            for stream in range(1, 1+self.num_str):
                for section in range(1, 1+self.num_sec):
                    s.append(middle_vars[_tower_num]["mf_n2"][stream][section])
        # 7. 層伝熱係数
        for _tower_num in range(1, 1+self.num_tower):
            for stream in range(1, 1+self.num_str):
                for section in range(1, 1+self.num_sec):
                    s.append(middle_vars[_tower_num]["heat_t_coef"][stream][section])
        # 8. 壁―層伝熱係数
        for _tower_num in range(1, 1+self.num_tower):
            for stream in range(1, 1+self.num_str):
                for section in range(1, 1+self.num_sec):
                    s.append(middle_vars[_tower_num]["heat_t_coef_wall"][stream][section])
        # 9. 全圧
        for _tower_num in range(1, 1+self.num_tower):
            s.append(middle_vars[_tower_num]["total_press"])
        # 10. CO2,N2回収量
        for _tower_num in range(1, 1+self.num_tower):
            s.append(middle_vars[_tower_num]["vacuum_amt_co2"])
            s.append(middle_vars[_tower_num]["vacuum_amt_n2"])

        s = np.array(s)

        return s