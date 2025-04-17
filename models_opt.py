import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
import sqlite3
import multiprocessing

from utils import const, init_functions, plot_csv
from simulator import GasAdosorption_Breakthrough_simulator

import warnings

warnings.simplefilter("ignore")


class GasAdosorption_for_Optimize:
    """ガス吸着モデル(バッチプロセス)を実行するクラス"""

    def __init__(self, cond_id, opt_params):
        """初期化関数

        Args:
            cond_id (str): 実験条件の名前
            opt_params (dict): 最適化パラメータ
        """
        self.cond_id = cond_id
        self.n_processes = opt_params["num_processes"]
        self.max_trials = opt_params["num_trials"] / self.n_processes
        self.max_trials = round(self.max_trials)
        self.trials_num = 0
        self.num_objective = 3
        self.objective_name = ["T1_rmse", "T2_rmse", "T3_rmse"]

        # 出力先フォルダ
        self.opt_path = const.OUTPUT_DIR + self.cond_id + "/simulation/optimize/"
        os.makedirs(self.opt_path, exist_ok=True)

    def optimize_params(self):
        # 最適化の実施
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        processes = []
        for _ in range(self.n_processes):  # 並列化
            p = multiprocessing.Process(target=self.run_optimization)
            p.start()
            processes.append(p)
        for p in processes:  # 全プロセスが完了するまで待機
            p.join()
        print("")

        # 最適化結果の読み込み
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.opt_path}/optimize.db"
        )
        study = optuna.create_study(
            study_name="GasAdsorption",
            directions=["minimize"] * self.num_objective,
            storage=storage,
            sampler=TPESampler(),
            load_if_exists=True,
        )

        ### 記録 ------------------------------

        # csv化
        df_opt = study.trials_dataframe()
        df_opt["best_trial"] = [
            1 if num in [trial.number for trial in study.best_trials] else 0
            for num in range(len(study.trials))
        ]
        df_opt.to_csv(self.opt_path + "/study.csv", index=False)
        # 可視化
        self._plot_outputs(study=study, df_opt=df_opt)

        ### 再シミュレーション --------------------------------------

        if self.num_objective != 1:
            print("再シミュレーション ---------------")
            # 総合最適パラメータの抽出
            df_opt["values_sum"] = 0
            for i in range(self.num_objective):
                df_opt["values_sum"] += df_opt[f"values_{i}"]
            df_tgt = df_opt.loc[df_opt["values_sum"].idxmin()]
            params_dict = {
                1: {
                    "PACKED_BED_COND": {
                        "ks_adsorp": df_tgt["params_ks_adsorp_1"],
                        "ks_desorp": df_tgt["params_ks_desorp_1"],
                        "vacuume_pressure": df_tgt["vacuume_pressure_1"],
                    },
                    # "DRUM_WALL_COND": {
                    #     "coef_hw1": df_tgt["params_coef_hw1"],},
                    "INFLOW_GAS_COND": {
                        "adsorp_heat_co2": df_tgt["params_adsorp_heat_co2"],
                    },
                },
                2: {
                    "PACKED_BED_COND": {
                        "ks_adsorp": df_tgt["params_ks_adsorp_2"],
                        "ks_desorp": df_tgt["params_ks_desorp_2"],
                        "vacuume_pressure": df_tgt["vacuume_pressure_2"],
                    },
                    # "DRUM_WALL_COND": {
                    #     "coef_hw1": df_tgt["params_coef_hw1"],},
                    "INFLOW_GAS_COND": {
                        "adsorp_heat_co2": df_tgt["params_adsorp_heat_co2"],
                    },
                },
                3: {
                    "PACKED_BED_COND": {
                        "ks_adsorp": df_tgt["params_ks_adsorp_3"],
                        "ks_desorp": df_tgt["params_ks_desorp_3"],
                        "vacuume_pressure": df_tgt["vacuume_pressure_3"],
                    },
                    # "DRUM_WALL_COND": {
                    #     "coef_hw1": df_tgt["params_coef_hw1"],},
                    "INFLOW_GAS_COND": {
                        "adsorp_heat_co2": df_tgt["params_adsorp_heat_co2"],
                    },
                },
            }
            # txt化
            with open(self.opt_path + "/best_params.txt", mode="w") as f:
                f.write(f"values_name = {self.objective_name}\n")
                _tgt_col = [col for col in df_tgt.index if "values" in col]
                f.write(f"best_values = {df_tgt[_tgt_col].values[:-1]}\n")
                _sum_values = sum(df_tgt[_tgt_col].values[:-1])
                f.write(f"sum_values = {_sum_values}\n")
                for _tower_num in [1, 2, 3]:
                    f.write(f"params_T{_tower_num} = {params_dict[_tower_num]}\n")
            # パラメータの置換
            instance = GasAdosorption_Breakthrough_simulator(self.cond_id)
            for _tower_num, tgt_tower_params in params_dict.items():
                for cond_category, tgt_conds in tgt_tower_params.items():
                    for cond_name, value in tgt_conds.items():
                        instance.sim_conds[_tower_num][cond_category][cond_name] = value
            # 追加の初期化
            for _tower_num in [1, 2, 3]:
                instance.sim_conds[_tower_num] = init_functions.add_sim_conds(
                    instance.sim_conds[_tower_num]
                )
            # stream条件の初期化
            instance.stream_conds = {}
            for _tower_num in [1, 2, 3]:
                instance.stream_conds[_tower_num] = {}
                for stream in range(1, 1 + instance.num_str):
                    instance.stream_conds[_tower_num][stream] = (
                        init_functions.init_stream_conds(  # 各ストリーム
                            instance.sim_conds[_tower_num],
                            stream,
                            instance.stream_conds[_tower_num],
                        )
                    )
                instance.stream_conds[_tower_num][stream + 1] = (
                    init_functions.init_drum_wall_conds(  # 壁面
                        instance.sim_conds[_tower_num],
                        instance.stream_conds[_tower_num],
                    )
                )
            # 再シミュレーション
            instance.execute_simulation()

    def run_optimization(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.opt_path}/optimize.db",
            engine_kwargs={
                "connect_args": {"timeout": 30},
                "isolation_level": "SERIALIZABLE",
            },
        )
        study = optuna.create_study(
            study_name="GasAdsorption",
            directions=["minimize"] * self.num_objective,
            storage=storage,
            sampler=TPESampler(),
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=self.max_trials)

    def objective(self, trial):
        """最適化条件の設定"""
        self.trials_num += 1
        # 最適化条件
        params_dict = {
            1: {
                "PACKED_BED_COND": {
                    "ks_adsorp": trial.suggest_float(
                        "ks_adsorp_1", 1e-8, 1e3, log=True
                    ),
                    "ks_desorp": trial.suggest_float(
                        "ks_desorp_1", 1e-8, 1e3, log=True
                    ),
                    "vacuume_pressure": trial.suggest_uniform(
                        "vacuume_pressure_1", 1, 10
                    ),
                },
                # "DRUM_WALL_COND": {
                #     "coef_hw1": trial.suggest_float("coef_hw1", 1e-5, 1e1, log=True),},
                "INFLOW_GAS_COND": {
                    "adsorp_heat_co2": trial.suggest_float(
                        "adsorp_heat_co2", 100, 3000
                    ),
                },
            },
            2: {
                "PACKED_BED_COND": {
                    "ks_adsorp": trial.suggest_float(
                        "ks_adsorp_2", 1e-8, 1e3, log=True
                    ),
                    "ks_desorp": trial.suggest_float(
                        "ks_desorp_2", 1e-8, 1e3, log=True
                    ),
                    "vacuume_pressure": trial.suggest_uniform(
                        "vacuume_pressure_2", 1, 10
                    ),
                },
                # "DRUM_WALL_COND": {
                #     "coef_hw1": trial.suggest_float("coef_hw1", 1e-5, 1e1, log=True),},
                "INFLOW_GAS_COND": {
                    "adsorp_heat_co2": trial.suggest_float(
                        "adsorp_heat_co2", 100, 3000
                    ),
                },
            },
            3: {
                "PACKED_BED_COND": {
                    "ks_adsorp": trial.suggest_float(
                        "ks_adsorp_3", 1e-8, 1e3, log=True
                    ),
                    "ks_desorp": trial.suggest_float(
                        "ks_desorp_3", 5e-2, 1e3, log=True
                    ),
                    "vacuume_pressure": trial.suggest_uniform(
                        "vacuume_pressure_3", 1, 10
                    ),
                },
                # "DRUM_WALL_COND": {
                #     "coef_hw1": trial.suggest_float("coef_hw1", 1e-5, 1e1, log=True),},
                "INFLOW_GAS_COND": {
                    "adsorp_heat_co2": trial.suggest_float(
                        "adsorp_heat_co2", 100, 3000
                    ),
                },
            },
        }
        # score計算
        try:
            score1, score2, score3 = self.calc_score(params_dict)
            print("\r" + f"trial: {self.trials_num}/{self.max_trials}", end="")
            return score1, score2, score3
        # 例外処理
        except Exception as e:
            # エラーをログに記録
            print(f"Error occurred: {e}")
            # print(f"Error occurred: {params_dict}")
            # 試行を失敗として扱う
            return [30] * self.num_objective  # または raise

    def calc_score(self, params_dict):
        """物理計算を通しで実行"""
        # インスタンス化
        instance = GasAdosorption_Breakthrough_simulator(self.cond_id)
        # パラメータ置換
        for _tower_num, tgt_tower_params in params_dict.items():
            for cond_category, tgt_conds in tgt_tower_params.items():
                for cond_name, value in tgt_conds.items():
                    instance.sim_conds[_tower_num][cond_category][cond_name] = value
        # 追加の初期化
        for _tower_num in [1, 2, 3]:
            instance.sim_conds[_tower_num] = init_functions.add_sim_conds(
                instance.sim_conds[_tower_num]
            )
        # stream条件の初期化
        instance.stream_conds = {}
        for _tower_num in [1, 2, 3]:
            instance.stream_conds[_tower_num] = {}
            for stream in range(1, 1 + instance.num_str):
                instance.stream_conds[_tower_num][stream] = (
                    init_functions.init_stream_conds(  # 各ストリーム
                        instance.sim_conds[_tower_num],
                        stream,
                        instance.stream_conds[_tower_num],
                    )
                )
            instance.stream_conds[_tower_num][stream + 1] = (
                init_functions.init_drum_wall_conds(  # 壁面
                    instance.sim_conds[_tower_num], instance.stream_conds[_tower_num]
                )
            )

        ### ◆前準備 ------------------------------------------------

        # 記録用配列の用意
        _record_item_list = ["material", "heat", "heat_wall", "heat_lid", "others"]
        record_dict = {}
        for _tower_num in range(1, 1 + instance.sim_conds[1]["NUM_TOWER"]):
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
            # termination_cond_str = instance.df_operation.loc[p, "終了条件"]
            # 手動終了条件の抽出
            termination_time = instance.df_operation.loc[p, "手動終了時刻"]
            # プロセスpにおける各塔の吸着計算実施
            timestamp, variables_tower, record_dict = instance.calc_adsorption_process(
                mode_list=mode_list,
                termination_cond_str=termination_time,
                variables_tower=variables_tower,
                record_dict=record_dict,
                timestamp=timestamp,
                manual=True,
            )

        ### ◆(2/2) score計算 -------------------------------------------------

        # 計算値（温度）の抽出
        values = {}
        num_data = len(record_dict[1]["timestamp"])
        for _tower_num in range(1, 1 + instance.sim_conds[1]["NUM_TOWER"]):
            stream = 1  # 現状、観測値はstream=1のみ
            for i, section in enumerate(
                instance.sim_conds[1]["LOC_CENCER_SECTION"].values()
            ):  # センサー付近のsectionが対象
                values[f"T{_tower_num}_temp_{i+1}"] = []
                for j in range(num_data):
                    values[f"T{_tower_num}_temp_{i+1}"].append(
                        record_dict[_tower_num]["heat"][j][stream][section][
                            "temp_reached"
                        ]
                    )
        df_sim = pd.DataFrame(values, index=record_dict[1]["timestamp"])
        df_sim.index.name = "timestamp"
        # 観測値の用意
        df_obs = instance.df_obs.loc[
            : df_sim.index[-1], :
        ]  # シミュレーション区間のみ抽出
        # 計算値と観測値のindexを合わせる
        common_index = [
            np.argmin(np.abs(instance.df_obs.index[i] - df_sim.index))
            for i in range(len(df_obs))
        ]
        df_sim = df_sim.iloc[common_index]

        # スコア計算
        try:
            scores = {}
            # score_0: rmse_真空脱着_塔3
            # _time_start = 0
            # _time_end = 14.12
            for _tower_num in [1, 2, 3]:
                _score_list = []
                for section in [1, 2, 3]:
                    if (_tower_num == 2) & (section == 2):
                        continue
                    _score_list.append(
                        mean_squared_error(
                            # df_sim.loc[_time_start:_time_end, f"T{_tower_num}_temp_{section}"],
                            # df_obs.loc[_time_start:_time_end, f"T{_tower_num}_temp_{section}"],
                            df_sim[f"T{_tower_num}_temp_{section}"],
                            df_obs[f"T{_tower_num}_temp_{section}"],
                            squared=False,
                        )
                    )
                scores[_tower_num] = np.mean(_score_list)

            return scores[1], scores[2], scores[3]
        except Exception as e:
            print(e.__class__.__name__)  # ZeroDivisionError
            print(e.args)  # ('division by zero',)
            print(e)  # division by zero
            print(f"{e.__class__.__name__}: {e}")  # ZeroDivisionError: division by zero
            import sys

            sys.exit()

    def _plot_outputs(self, study, df_opt):
        """最適化結果をpng出力する

        Args:
            study (db?): 最適化結果
            df_opt (pd.DataFrame): 最適化結果のDataFrame
        """
        # 1. 目的関数の時系列プロット
        if self.num_objective == 1:  # 目的関数が1つ
            plt.figure(figsize=(8, 5), tight_layout=True)
            plt.plot(df_opt["value"])
            plt.title("目的関数の時系列プロット")
            plt.xlabel("Trial Number")
            plt.ylabel("Object")
            plt.grid()
        else:  # 目的関数が複数
            num_row = int(np.ceil(self.num_objective / 2))
            plt.figure(figsize=(16, 5.5 * num_row), tight_layout=True)
            for num in range(self.num_objective):
                # df_opt.loc[df_opt[f"values_{num}"] > df_opt[f"values_{num}"].quantile(0.99),
                #         f"values_{num}"] = np.inf # 外れ値
                plt.subplot(num_row, 2, num + 1)
                plt.plot(df_opt[f"values_{num}"])
                plt.title(self.objective_name[num])
                plt.xlabel("Trial Number")
                plt.ylabel("Object")
                plt.grid()
        plt.savefig(self.opt_path + "lineplot_objectives.png", dpi=100)
        plt.close()
        # 2. 目的関数のヒストグラム
        if self.num_objective == 1:  # 目的関数が1つ
            plt.figure(figsize=(8, 5), tight_layout=True)
            num_bin = int(1 + np.log2(len(df_opt)))
            plt.hist(df_opt["value"], bins=num_bin)
            plt.title("目的関数のヒストグラム")
            plt.xlabel("Object")
            plt.grid()
        else:  # 目的関数が複数
            num_row = int(np.ceil(self.num_objective / 2))
            plt.figure(figsize=(16, 5.5 * num_row), tight_layout=True)
            for num in range(self.num_objective):
                plt.subplot(num_row, 2, num + 1)
                sns.histplot(data=df_opt, x=f"values_{num}")
                plt.title(self.objective_name[num])
                plt.xlabel("Object")
                plt.grid()
        plt.savefig(self.opt_path + "histogram.png", dpi=100)
        plt.close()
        # 3. 散布図
        if self.num_objective == 1:  # 目的関数が1つ
            num_params = len(study.best_params.keys())
            plt.figure(figsize=(8 * num_params, 5), tight_layout=True)
            for num, key in enumerate(study.best_params.keys()):
                plt.subplot(1, num_params, num + 1)
                plt.scatter(df_opt[f"params_{key}"], df_opt["value"])
                plt.title("散布図 (objectives vs states)")
                plt.xlabel(key)
                plt.ylabel(self.objective_name[0])
                plt.grid()
        else:  # 目的関数が複数
            num_row = self.num_objective
            num_col = len(study.best_trials[0].params.keys())
            best_trial_cond = df_opt["best_trial"] == 1
            plt.figure(figsize=(8 * num_col, 5.5 * num_row), tight_layout=True)
            plt.suptitle(f"散布図 (objectives vs states)", y=0.99)
            _plt_axis = 1
            for i in range(num_row):
                for key in study.best_trials[0].params.keys():
                    plt.subplot(num_row, num_col, _plt_axis)
                    plt.scatter(
                        df_opt.loc[~best_trial_cond, f"params_{key}"],
                        df_opt.loc[~best_trial_cond, f"values_{i}"],
                        c="tab:blue",
                        alpha=0.6,
                    )
                    plt.scatter(
                        df_opt.loc[best_trial_cond, f"params_{key}"],
                        df_opt.loc[best_trial_cond, f"values_{i}"],
                        c="tab:red",
                        label="best_trial",
                        alpha=0.6,
                    )
                    plt.title(self.objective_name[i])
                    plt.xlabel(key)
                    plt.grid()
                    plt.legend()
                    plt.xscale("log")
                    _plt_axis += 1
        plt.savefig(self.opt_path + f"scatters_obj_state.png", dpi=100)
        plt.close()
        # 散布図行列_objectives
        if self.num_objective != 1:
            df_tgt = df_opt[
                [f"values_{num}" for num in range(self.num_objective)] + ["best_trial"]
            ]
            df_tgt.columns = self.objective_name + ["best_trial"]
            plt.figure(figsize=(16, 16), tight_layout=True)
            sns.pairplot(
                df_tgt,
                hue="best_trial",
                diag_kws={"alpha": 0.5},
                plot_kws={"alpha": 0.3},
            )  # 散布図行列
            plt.savefig(self.opt_path + f"pairplot_objectives.png", dpi=100)
            plt.title("散布図 obj vs obj")
            plt.close()
        # 散布図行列_states
        params_list = list(study.best_trials[0].params.keys())  # パラメータ一覧
        num_state = int(len(params_list))  # パラメータの数
        if num_state != 1:  # 状態変数が複数のときのみ
            # best_trialによる色分け
            df_tgt = df_opt[
                [f"params_{p_name}" for p_name in params_list] + ["best_trial"]
            ]
            df_tgt.columns = list(params_list) + [
                "best_trial"
            ]  # 色分け用にbest_tiralを追加
            plt.figure(figsize=(16, 16), tight_layout=True)
            pp = sns.pairplot(
                df_tgt,
                hue=f"best_trial",
                diag_kws={"alpha": 0.5},
                plot_kws={"alpha": 0.3},
            )  # 散布図行列
            for ax in pp.axes.flat:
                ax.set(xscale="log")
                ax.set(yscale="log")
            plt.savefig(self.opt_path + f"pairplot_states_best_trial.png", dpi=100)
            plt.title("散布図 states vs states")
            plt.close()
            # 各目的関数による色分け
            for i, obj_name in enumerate(self.objective_name):
                df_tgt = df_opt[
                    [f"params_{p_name}" for p_name in params_list] + [f"values_{i}"]
                ]
                df_tgt.columns = list(params_list) + [
                    f"values_{i}"
                ]  # 色分け用にvaluesを追加
                df_tgt = df_tgt.replace([np.inf, -np.inf], np.nan)
                df_tgt.dropna()
                df_tgt[obj_name] = pd.qcut(df_tgt[f"values_{i}"], q=5)
                df_tgt = df_tgt.drop(columns=[f"values_{i}"])
                plt.figure(figsize=(4 * num_state, 4 * num_state), tight_layout=True)
                pp = sns.pairplot(
                    df_tgt,
                    hue=obj_name,
                    diag_kws={"alpha": 0.5},
                    plot_kws={"alpha": 0.8},
                    palette="RdBu",
                )  # 散布図行列
                for ax in pp.axes.flat:
                    ax.set(xscale="log")
                    ax.set(yscale="log")
                plt.suptitle(f"散布図 states vs states ({obj_name})", y=1.01)
                plt.savefig(
                    self.opt_path + f"pairplot_states_{obj_name}.png",
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.close()
