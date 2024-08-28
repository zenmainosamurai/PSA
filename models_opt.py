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

from utils import const, init_functions, plot_csv
from models import GasAdosorption_Breakthrough_simulator

import warnings
warnings.simplefilter('error')


class GasAdosorption_for_Optimize(GasAdosorption_Breakthrough_simulator):
    """ ガス吸着モデル(バッチプロセス)を実行するクラス
    """

    def __init__(self, cond_id, opt_params):
        """ 初期化関数

        Args:
            cond_id (str): 実験条件の名前
            opt_params (dict): 最適化パラメータ
        """
        self.cond_id = cond_id
        super().__init__(self.cond_id)
        self.n_processes = opt_params["num_processes"]
        self.max_trials = opt_params["num_trials"] / self.n_processes
        self.max_trials = round(self.max_trials)
        self.trials_num = 0

    def optimize_params(self):
        # 最適化の実施
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage_path = const.OUTPUT_DIR + self.cond_id + "/simulation/"
        os.makedirs(storage_path, exist_ok=True)
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{storage_path}/optimize.db')
        self.study = optuna.create_study(study_name="GasAdsorption", direction="minimize",
                                    storage=storage, sampler=TPESampler(), load_if_exists=True)
        processes = []
        for i in range(self.n_processes): # 並列化
            p = multiprocessing.Process(target=self.run_optimization)
            p.start()
            processes.append(p)
        for i, p in enumerate(processes): # 全プロセスが完了するまで待機
            p.join()
        print("")

        # テキスト出力
        params_dict = {
            "INFLOW_GAS_COND": {
                # "fr_co2": 22 * self.study.best_params["fr"],
                # "fr_n2": 28 * self.study.best_params["fr"],
                "adsorp_heat_co2": 1363.6 * self.study.best_params["adsorp_heat_co2"],
            },
            "PACKED_BED_COND": {
                "Mabs": 10800 * self.study.best_params["Mabs"],
                "ks": 4 * self.study.best_params["ks"],
            },
            "DRUM_WALL_COND": {
                "coef_hw1": 1 * self.study.best_params["coef_hw1"],
            }
        }
        try:
            txt_filepath = const.OUTPUT_DIR + self.cond_id + "/simulation/best_params.txt"
            with open(txt_filepath, mode="w") as f:
                for key, value in self.study.best_params.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"\nbest_score = {self.study.best_value}")
        except:
            print("失敗")
        print("最適化結果 ---------------")
        print("params: ", params_dict)
        print("best_score: ", self.study.best_value)
        # 再シミュレーション
        for cond_category, cond_dict in params_dict.items():
            for cond_name, value in cond_dict.items():
                self.common_conds[cond_category][cond_name] = value
        self.execute_simulation()

    def run_optimization(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study.optimize(self.objective, n_trials=self.max_trials)

    def objective(self, trial):
        """ 最適化条件の設定
        """
        self.trials_num += 1
        # 最適化条件
        params_dict = {
            "INFLOW_GAS_COND": {
                # "fr_co2": 22 * trial.suggest_loguniform("fr", 0.1, 10),
                # "fr_n2": 28 * trial.suggest_loguniform("fr", 0.1, 10),
                "adsorp_heat_co2": 1363.6 * trial.suggest_loguniform("adsorp_heat_co2", 0.1, 10),
            },
            "PACKED_BED_COND": {
                "Mabs": 10800 * trial.suggest_loguniform("Mabs", 0.1, 10),
                "ks": 4 * trial.suggest_loguniform("ks", 0.1, 10),
            },
            "DRUM_WALL_COND": {
                "coef_hw1": 1 * trial.suggest_loguniform("coef_hw1", 0.1, 1),
            }
        }
        # パラメータ置換
        for cond_category, cond_dict in params_dict.items():
            for cond_name, value in cond_dict.items():
                self.common_conds[cond_category][cond_name] = value
        # rmse計算
        rmse = self.calc_rmse()
        print("\r" + f"trial: {self.trials_num}/{self.max_trials}", end="")
        
        return rmse

    def calc_rmse(self, filtered_states=None):
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

        ### ◆(1/2) シミュレーション実行 --------------------------------------

        # 初期化
        variables = self.init_variables()

        # 全体計算
        timestamp = 0
        while timestamp < self.df_obs.index[-1]:
            # 1step計算実行
            variables, all_output = self.calc_all_cell_balance(variables, timestamp)
            # timestamp更新
            timestamp += self.common_conds["dt"]
            timestamp = round(timestamp, 2)
            # 記録用配列の平坦化
            output_flatten = {}
            for stream in range(1, 2+self.num_str): # 熱バラ
                for section in range(1, 1+self.num_sec):
                        for key, value in all_output["heat"][stream][section].items():
                            output_flatten[key+"_"+str(stream).zfill(3)+"_"+str(section).zfill(3)] = value
            output_flatten["temp_reached_up"] = all_output["heat_lid"]["up"]["temp_reached"] # 熱バラ（上下蓋）
            output_flatten["temp_reached_dw"] = all_output["heat_lid"]["down"]["temp_reached"]
            for stream in range(1, 1+self.num_str): # マテバラ
                for section in range(1, 1+self.num_sec):
                        for key, value in all_output["material"][stream][section].items():
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

        ### ◆(2/2) RMSE計算 -------------------------------------------------
        # センサーに最も近いセクションの算出
        nearest_sec = []
        loc_cells = np.arange(0, self.common_conds["PACKED_BED_COND"]["Lbed"],
                              self.common_conds["PACKED_BED_COND"]["Lbed"] / self.num_sec) # 各セルの位置
        loc_cells += self.common_conds["PACKED_BED_COND"]["Lbed"] / self.num_sec / 2 # セルの半径を加算
        for value in self.common_conds["LOC_CENCER"].values(): # 温度計に最も近いセクションを算出
            nearest_sec.append(1 + np.argmin(np.abs(loc_cells - value)))

        # データ準備
        tgt_cols = [f"temp_reached_{str(stream).zfill(3)}_{str(section).zfill(3)}" for stream in [1,2] for section in nearest_sec[1:]]
        rename_cols = [f"temp_{str(stream).zfill(3)}_{str(section).zfill(3)}" for stream in [1,2] for section in [2,3]]
        df_sim = df[tgt_cols]
        df_sim.columns = rename_cols
        df_obs = pd.read_excel(const.DATA_DIR + self.common_conds["data_path"], # 観測値
                               sheet_name=self.common_conds["sheet_name"], index_col="time")
        common_index = [np.argmin(np.abs(df_obs.index[i] - df_sim.index)) for i in range(len(df_obs.index))]
        df_sim = df_sim.iloc[common_index]

        # RMSE計算
        rmse_list = []
        for col in rename_cols:
            rmse = mean_squared_error(df_sim[col], df_obs[col], squared=False)
            rmse_list.append(rmse)

        return np.mean(rmse_list)