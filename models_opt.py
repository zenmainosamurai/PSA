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
            # "INFLOW_GAS_COND": {
            #     "adsorp_heat_co2": study.best_params["adsorp_heat_co2"],
            # },
            "PACKED_BED_COND": {
                # "ks_adsorp": study.best_params["ks_adsorp"],
                # "ks_desorp_1": study.best_params["ks_desorp_1"],
                # "ks_desorp_2": study.best_params["ks_desorp_2"],
                "ks_desorp_3": study.best_params["ks_desorp_3"],
            },
            # "DRUM_WALL_COND": {
            #     "coef_hw1": study.best_params["coef_hw1"],
            # }
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
        # sns.histplot(data=df_opt, x='value')
        plt.hist(df_opt["value"], bins=50)
        plt.title('目的関数のヒストグラム')
        plt.xlabel("Object")
        plt.grid()
        plt.savefig(self.opt_path + "line_histogram.png", dpi=100)
        plt.close()
        # 可視化: 散布図
        plt.figure(figsize=(16, 10), tight_layout=True)
        num_params = len(study.best_params.keys())
        num_row = int(np.ceil(num_params/2))
        for i, key in enumerate(study.best_params.keys()):
            plt.subplot(2, num_row, i+1)
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
        for cond_category, cond_dict in params_dict.items(): # パラメータ上書き
            for cond_name, value in cond_dict.items():
                if "ks_desorp" in cond_name:
                    continue
                else:
                    for _tower_num in [1,2,3]:
                        instance.sim_conds[_tower_num][cond_category][cond_name] = value
        # instance.sim_conds[1]["PACKED_BED_COND"]["ks_desorp"] = params_dict["PACKED_BED_COND"]["ks_desorp_1"]
        # instance.sim_conds[2]["PACKED_BED_COND"]["ks_desorp"] = params_dict["PACKED_BED_COND"]["ks_desorp_2"]
        instance.sim_conds[3]["PACKED_BED_COND"]["ks_desorp"] = params_dict["PACKED_BED_COND"]["ks_desorp_3"]
        instance.execute_simulation() # 再シミュレーション

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
            # "INFLOW_GAS_COND": {
            #     "adsorp_heat_co2": trial.suggest_float("adsorp_heat_co2", 100, 3000, log=True),
            # },
            "PACKED_BED_COND": {
                # "ks_adsorp": trial.suggest_float("ks_adsorp", 1e-8, 1, log=True),
                # "ks_desorp_1": trial.suggest_float("ks_desorp_1", 1e-8, 10, log=True),
                # "ks_desorp_2": trial.suggest_float("ks_desorp_2", 1e-8, 10, log=True),
                "ks_desorp_3": trial.suggest_float("ks_desorp_3", 5e-1, 1),
            },
            # "DRUM_WALL_COND": {
            #     "coef_hw1": trial.suggest_float("coef_hw1", 1e-3, 1e1, log=True),
            # }
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
            # print(f"Error occurred: {params_dict}")
            # 試行を失敗として扱う
            return 100  # または raise

    def calc_score(self, params_dict):
        """ 物理計算を通しで実行
        """
        # インスタンス化
        instance = GasAdosorption_Breakthrough_simulator(self.cond_id)
        # パラメータ置換
        for cond_category, cond_dict in params_dict.items():
            for cond_name, value in cond_dict.items():
                if "ks_desorp" in cond_name:
                    continue
                else:
                    for _tower_num in [1,2,3]:
                        instance.sim_conds[_tower_num][cond_category][cond_name] = value
        # instance.sim_conds[1]["PACKED_BED_COND"]["ks_desorp"] = params_dict["PACKED_BED_COND"]["ks_desorp_1"]
        # instance.sim_conds[2]["PACKED_BED_COND"]["ks_desorp"] = params_dict["PACKED_BED_COND"]["ks_desorp_2"]
        instance.sim_conds[3]["PACKED_BED_COND"]["ks_desorp"] = params_dict["PACKED_BED_COND"]["ks_desorp_3"]

        ### ◆前準備 ------------------------------------------------

        # 記録用配列の用意
        _record_item_list = ["material", "heat", "heat_wall", "heat_lid", "others"]
        record_dict = {}
        for _tower_num in range(1, 1+instance.sim_conds[1]["NUM_TOWER"]):
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
            # 手動終了条件の抽出
            termination_time = instance.df_operation.loc[p, "手動終了時刻"]
            # プロセスpにおける各塔の吸着計算実施
            timestamp, variables_tower, record_dict = instance.calc_adsorption_process(mode_list=mode_list,
                                                                                       termination_cond_str=termination_time,
                                                                                       variables_tower=variables_tower,
                                                                                       record_dict=record_dict,
                                                                                       timestamp=timestamp,
                                                                                       manual=True)

        ### ◆(2/2) score計算 -------------------------------------------------

        # 計算値（温度）の抽出
        values = {}
        num_data = len(record_dict[1]["timestamp"])
        for _tower_num in range(1,1+instance.sim_conds[1]["NUM_TOWER"]):
            stream = 1 # 現状、観測値はstream=1のみ
            for i, section in enumerate(instance.sim_conds[1]["LOC_CENCER_SECTION"].values()): # センサー付近のsectionが対象
                values[f"T{_tower_num}_temp_{i+1}"] = []
                for j in range(num_data):
                    values[f"T{_tower_num}_temp_{i+1}"].append(record_dict[_tower_num]["heat"][j][stream][section]["temp_reached"])
        df_sim = pd.DataFrame(values, index=record_dict[1]["timestamp"])
        df_sim.index.name = "timestamp"
        # 観測値の用意
        df_obs = instance.df_obs.loc[:df_sim.index[-1], :] # シミュレーション区間のみ抽出
        # 計算値と観測値のindexを合わせる
        common_index = [np.argmin(np.abs(instance.df_obs.index[i] - df_sim.index)) for i in range(len(df_obs))]
        df_sim = df_sim.iloc[common_index]

        # スコア計算
        scores = {}
        # score_0: rmse_真空脱着_塔3
        _time_start = 4.4
        _time_end = 14.12
        _score_list = []
        for section in [1,2,3]:
            _score_list.append(mean_squared_error(df_sim.loc[_time_start:_time_end, f"T3_temp_{section}"],
                                                  df_obs.loc[_time_start:_time_end, f"T3_temp_{section}"],
                                                  squared=False))
        scores[0] = np.mean(_score_list)

        return scores[0]