import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
import sqlite3
import multiprocessing

from utils import const, init_functions, plot_csv
from simulator import GasAdosorption_Breakthrough_simulator

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

        # 出力先フォルダ
        self.opt_path = const.OUTPUT_DIR + self.cond_id + "/simulation/optimize/"
        os.makedirs(self.opt_path, exist_ok=True)

    def optimize_params(self):
        # 最適化の実施
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        processes = []
        for _ in range(self.n_processes): # 並列化
            p = multiprocessing.Process(target=self.run_optimization)
            p.start()
            processes.append(p)
        for p in processes: # 全プロセスが完了するまで待機
            p.join()
        print("")

        # 最適化結果の読み込み
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{self.opt_path}/optimize.db')
        study = optuna.create_study(study_name="GasAdsorption", direction="minimize",
                                    storage=storage, sampler=TPESampler(), load_if_exists=True)
        params_dict = {
            "INFLOW_GAS_COND": {
                "adsorp_heat_co2": study.best_params["adsorp_heat_co2"],
            },
            "PACKED_BED_COND": {
                "ks_adsorp": study.best_params["ks_adsorp"],
                "ks_desorp": study.best_params["ks_desorp"],
            },
            "DRUM_WALL_COND": {
                "coef_hw1": study.best_params["coef_hw1"],
            }
        }
        print("最適化結果 ---------------")
        print("params: ", params_dict)
        print("best_score: ", study.best_value)

        ### 記録 ----------------------------------------

        # txt化
        with open(self.opt_path + "best_params.txt", mode="w") as f:
            for key, value in study.best_params.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\nbest_score = {study.best_value}")
        # csv化
        df_opt = study.trials_dataframe()
        df_opt.to_csv(self.opt_path + "/study.csv", index=False)
        # 可視化: 目的関数の時系列プロット
        plt.rcParams["font.size"] = 14
        plt.figure(figsize=(16, 5), tight_layout=True)
        plt.subplot(1,2,1)
        plt.plot(df_opt["value"])
        plt.title('目的関数の時系列プロット')
        plt.xlabel("Trial Number")
        plt.ylabel("Object")
        plt.grid()
        # 可視化: 目的関数のヒストグラム
        plt.subplot(1,2,2)
        sns.histplot(data=df_opt, x='value')
        plt.title('目的関数のヒストグラム')
        plt.xlabel("Object")
        plt.grid()
        plt.savefig(self.opt_path + "line_histogram.png", dpi=100)
        plt.close()
        # 可視化: 散布図
        plt.figure(figsize=(16, 10), tight_layout=True)
        for i, key in enumerate(study.best_params.keys()):
            plt.subplot(2,2,i+1)
            plt.scatter(df_opt[f"params_{key}"], df_opt["value"])
            plt.title(key)
            plt.xlabel(key)
            plt.ylabel("Object")
            plt.grid()
        plt.savefig(self.opt_path + "scatter.png", dpi=100)
        plt.close()
        # ヒートマップ（相関係数）
        correlation = df_opt.filter(like='params_')
        correlation.columns = [col[7:] for col in correlation.columns]
        correlation = correlation.corr()
        plt.figure(figsize=(16, 10), tight_layout=True)
        sns.heatmap(correlation, annot=True,
                        cmap=sns.color_palette('coolwarm', 5),
                        fmt='.2f',
                        vmin = -1,
                        vmax = 1,
                        annot_kws={'fontsize': 16, 'color':'black'})
        plt.title('Parameter Correlations')
        plt.savefig(self.opt_path + "heatmap.png", dpi=100)
        plt.close()

        # 再シミュレーション
        print("再シミュレーション ---------------")
        instance = GasAdosorption_Breakthrough_simulator(self.cond_id)
        for cond_category, cond_dict in params_dict.items():
            for cond_name, value in cond_dict.items():
                instance.sim_conds[cond_category][cond_name] = value
        instance.execute_simulation()

    def run_optimization(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = optuna.storages.RDBStorage(url=f'sqlite:///{self.opt_path}/optimize.db')
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
                "adsorp_heat_co2": trial.suggest_float("adsorp_heat_co2", 100, 1500, log=True),
            },
            "PACKED_BED_COND": {
                "ks_adsorp": trial.suggest_float("ks_adsorp", 1e-3, 1, log=True),
                "ks_desorp": trial.suggest_float("ks_desorp", 1e-3, 1, log=True),
            },
            "DRUM_WALL_COND": {
                "coef_hw1": trial.suggest_float("coef_hw1", 1e-2, 1e1, log=True),
            }
        }
        # score計算
        try:
            score = self.calc_score(params_dict)
            print("\r" + f"trial: {self.trials_num}/{self.max_trials}", end="")
            return score
        # 例外処理
        except Exception as e:
            # エラーをログに記録
            print(f"Error occurred: {e}")
            # 試行を失敗として扱う
            return float('inf')  # または raise

    def calc_score(self, params_dict):
        """ 物理計算を通しで実行
        """
        # インスタンス化
        instance = GasAdosorption_Breakthrough_simulator(self.cond_id)
        # パラメータ置換
        for cond_category, cond_dict in params_dict.items():
            for cond_name, value in cond_dict.items():
                instance.sim_conds[cond_category][cond_name] = value

        ### ◆前準備 ------------------------------------------------

        # 記録用配列の用意
        _record_item_list = ["material", "heat", "heat_wall", "heat_lid", "others"]
        record_dict = {}
        for _tower_num in range(1, 1+instance.sim_conds["NUM_TOWER"]):
            record_dict[_tower_num] = {}
            record_dict[_tower_num]["timestamp"] = []
            for _item in _record_item_list:
                record_dict[_tower_num][_item] = []

        ### ◆(1/2) シミュレーション実行 --------------------------------------

        # a. 状態変数の初期化
        variables_tower = instance._init_variables()
        # b. 吸着計算
        timestamp = 0
        for p in instance.df_operation.index:
            # 各塔の稼働モード抽出
            mode_list = list(instance.df_operation.loc[p, ["塔1", "塔2", "塔3"]])
            # 終了条件(文字列)の抽出
            termination_cond_str = instance.df_operation.loc[p, "終了条件"]
            # プロセスpにおける各塔の吸着計算実施
            timestamp, variables_tower, record_dict = instance.calc_adsorption_process(instance.sim_conds,
                                                                                   mode_list,
                                                                                   termination_cond_str,
                                                                                   variables_tower,
                                                                                   record_dict,
                                                                                   timestamp)
        # 温度データの抽出
        values = {}
        num_data = len(record_dict[1]["timestamp"])
        # for _tower_num in range(1,1+instance.sim_conds["NUM_TOWER"]):
            # for stream in range(1,1+instance.sim_conds["CELL_SPLIT"]["num_str"]):
        _tower_num = 1
        for i, section in enumerate(instance.sim_conds["LOC_CENCER_SECTION"].values()):
            values[f"T{_tower_num}_temp_{i+1}"] = []
            for j in range(num_data):
                values[f"T{_tower_num}_temp_{i+1}"].append(record_dict[_tower_num]["heat"][j][1][section]["temp_reached"])
        df_sim = pd.DataFrame(values, index=record_dict[1]["timestamp"])
        df_sim.index.name = "timestamp"

        ### ◆(2/2) スコア計算 -------------------------------------------------

        # indexを合わせる
        df_obs = instance.df_obs.loc[:df_sim.index[-1], :]
        common_index = [np.argmin(np.abs(instance.df_obs.index[i] - df_sim.index)) for i in range(len(df_obs))]
        df_sim = df_sim.iloc[common_index]
        # スコア計算
        score_list = []
        for col in df_sim.columns:
            score = mean_squared_error(df_sim[col], df_obs[col], squared=False) # RMSE
            # score, _ = fastdtw(df_sim[col], instance.df_obs[col], dist=euclidean) # DTW
            score_list.append(score)

        return np.mean(score_list)