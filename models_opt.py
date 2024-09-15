import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import optimize
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
import sqlite3
import multiprocessing
from contextlib import redirect_stdout
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from utils import const, init_functions, plot_csv
from models import GasAdosorption_Breakthrough_simulator

import warnings
warnings.simplefilter('ignore')


class GasAdosorption_for_Optimize():
    """ ガス吸着モデル(バッチプロセス)を実行するクラス
    """

    def __init__(self, cond_id, opt_params):
        """ 初期化関数

        Args:
            cond_id (str): 実験条件の名前
            opt_params (dict): 最適化パラメータ
        """
        self.cond_id = cond_id
        self.n_processes = opt_params["num_processes"]
        self.max_trials = opt_params["num_trials"] / self.n_processes
        self.max_trials = round(self.max_trials)
        self.trials_num = 0

    def optimize_params(self):
        # 最適化の実施
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.storage_path = const.OUTPUT_DIR + self.cond_id + "/simulation/"
        os.makedirs(self.storage_path, exist_ok=True)
        processes = []
        for i in range(self.n_processes): # 並列化
            p = multiprocessing.Process(target=self.run_optimization)
            p.start()
            processes.append(p)
        for i, p in enumerate(processes): # 全プロセスが完了するまで待機
            p.join()
        print("")

        # 最適化結果
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{self.storage_path}/optimize.db')
        study = optuna.create_study(study_name="GasAdsorption", direction="minimize",
                                    storage=storage, sampler=TPESampler(), load_if_exists=True)
        params_dict = {
            "INFLOW_GAS_COND": {
                "adsorp_heat_co2": 1363.6 * study.best_params["adsorp_heat_co2"],
            },
            "PACKED_BED_COND": {
                "ks": 4 * study.best_params["ks"],
            },
            "DRUM_WALL_COND": {
                "coef_hw1": 1 * study.best_params["coef_hw1"],
            }
        }
        txt_filepath = const.OUTPUT_DIR + self.cond_id + "/simulation/best_params.txt"
        with open(txt_filepath, mode="w") as f:
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nbest_score = {study.best_value}")
        print("最適化結果 ---------------")
        print("params: ", params_dict)
        print("best_score: ", study.best_value)
        # 再シミュレーション
        print("再シミュレーション ---------------")
        instance = GasAdosorption_Breakthrough_simulator(self.cond_id)
        for cond_category, cond_dict in params_dict.items():
            for cond_name, value in cond_dict.items():
                instance.common_conds[cond_category][cond_name] = value
        instance.execute_simulation()

    def run_optimization(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{self.storage_path}/optimize.db')
        study = optuna.create_study(study_name="GasAdsorption", direction="minimize",
                                    storage=storage, sampler=TPESampler(), load_if_exists=True)
        study.optimize(self.objective, n_trials=self.max_trials)

    def objective(self, trial):
        """ 最適化条件の設定
        """
        self.trials_num += 1
        # 最適化条件
        params_dict = {
            "INFLOW_GAS_COND": {
                # "fr_co2": 22 * trial.suggest_float("fr", 0.1, 10, log=True),
                # "fr_n2": 28 * trial.suggest_float("fr", 0.1, 10, log=True),
                "adsorp_heat_co2": 1363.6 * trial.suggest_float("adsorp_heat_co2", 0.1, 10, log=True),
            },
            "PACKED_BED_COND": {
                # "Mabs": 10800 * trial.suggest_float("Mabs", 0.1, 10, log=True),
                "ks": 4 * trial.suggest_float("ks", 0.1, 10, log=True),
            },
            "DRUM_WALL_COND": {
                "coef_hw1": 1 * trial.suggest_float("coef_hw1", 0.1, 10, log=True),
            }
        }
        # score計算
        score = self.calc_score(params_dict)
        print("\r" + f"trial: {self.trials_num}/{self.max_trials}", end="")

        return score

    def calc_score(self, params_dict):
        """ 物理計算を通しで実行
        """
        # インスタンス化
        instance = GasAdosorption_Breakthrough_simulator(self.cond_id)
        # パラメータ置換
        for cond_category, cond_dict in params_dict.items():
            for cond_name, value in cond_dict.items():
                instance.common_conds[cond_category][cond_name] = value

        ### ◆(1/4) 前準備 ------------------------------------------------

        # 記録用dictの用意
        record_dict = {
            "timestamp": [],
            "all_output": [],
        }

        ### ◆(1/2) シミュレーション実行 --------------------------------------

        # 初期化
        variables = instance.init_variables()

        # 全体計算
        timestamp = 0
        while timestamp < instance.df_obs.index[-1]:
            # 最も時刻が近い観測値のindex
            idx_obs = instance.df_obs.index[np.argmin(np.abs(timestamp - instance.df_obs.index))]
            # 弁停止していない場合
            if instance.df_obs.loc[idx_obs, "valve_stop_flag"] == 0:
                # 通常のマテバラ・熱バラ計算を実行
                variables, all_output = instance.calc_all_cell_balance(variables, timestamp)
            # 弁停止している場合
            elif instance.df_obs.loc[idx_obs, "valve_stop_flag"] == 1:
                # 弁停止時の関数を実行
                variables, all_output = instance.calc_all_cell_balance_when_valve_stop(variables, timestamp)
            # timestamp更新
            timestamp += instance.common_conds["dt"]
            timestamp = round(timestamp, 2)
            # 記録用配列の平坦化
            output_flatten = {}
            for stream in range(1, 2+instance.num_str): # 熱バラ
                for section in range(1, 1+instance.num_sec):
                        for key, value in all_output["heat"][stream][section].items():
                            output_flatten[key+"_"+str(stream).zfill(3)+"_"+str(section).zfill(3)] = value
            # 記録
            record_dict["timestamp"].append(timestamp)
            record_dict["all_output"].append(output_flatten)

        # DataFrame化
        values = []
        for i in range(len(record_dict["all_output"])):
            values.append(record_dict["all_output"][i].values())
        df = pd.DataFrame(values,
                          index=record_dict["timestamp"],
                          columns=record_dict["all_output"][0].keys())
        df.index.name = "timestamp"

        ### ◆(2/2) スコア計算 -------------------------------------------------
        # センサーに最も近いセクションの算出
        nearest_sec = []
        loc_cells = np.arange(0, instance.common_conds["PACKED_BED_COND"]["Lbed"],
                              instance.common_conds["PACKED_BED_COND"]["Lbed"] / instance.num_sec) # 各セルの位置
        loc_cells += instance.common_conds["PACKED_BED_COND"]["Lbed"] / instance.num_sec / 2 # セルの半径を加算
        for value in instance.common_conds["LOC_CENCER"].values(): # 温度計に最も近いセクションを算出
            nearest_sec.append(1 + np.argmin(np.abs(loc_cells - value)))

        # データ準備
        tgt_cols = [f"temp_reached_{str(stream).zfill(3)}_{str(section).zfill(3)}" for stream in [1,2] for section in nearest_sec]
        rename_cols = [f"temp_{str(stream).zfill(3)}_{str(section).zfill(3)}" for stream in [1,2] for section in [1,2,3]]
        df_sim = df[tgt_cols]
        df_sim.columns = rename_cols
        common_index = [np.argmin(np.abs(instance.df_obs.index[i] - df_sim.index)) for i in range(len(instance.df_obs.index))]
        df_sim = df_sim.iloc[common_index]

        # スコア計算
        score_list = []
        for col in rename_cols:
            score = mean_squared_error(df_sim[col], instance.df_obs[col], squared=False) # RMSE
            # score, _ = fastdtw(df_sim[col], instance.df_obs[col], dist=euclidean) # DTW
            score_list.append(score)

        return np.mean(score_list)