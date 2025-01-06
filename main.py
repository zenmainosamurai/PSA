import yaml
import time

from simulator import GasAdosorption_Breakthrough_simulator
from models_opt import GasAdosorption_for_Optimize

if __name__ == '__main__':

    # try:
    # 実行条件の読み込み
    with open("./main_cond.yml", encoding="utf-8") as f:
        main_cond = yaml.safe_load(f)

    start = time.time() # 時間計測

    # 計算実行
    print("計算開始----------------------------------")
    for mode in main_cond["mode_list"]:
        # シミュレーション
        if mode == "simulation":
            for cond_id in main_cond["cond_list"]:
                print("シミュレーション 実施中 ...")
                print("cond = ", cond_id)
                instance = GasAdosorption_Breakthrough_simulator(cond_id)
                instance.execute_simulation()
        # 最適化
        elif mode == "optimize":
            for cond_id in main_cond["cond_list"]:
                print("パラメータ探索 実施中 ...")
                print("cond = ", cond_id)
                instance = GasAdosorption_for_Optimize(cond_id, main_cond["opt_params"])
                instance.optimize_params()
    
    end = time.time() # 時間計測
    ptime = end - start
    ptime_hour = int(ptime//3600)
    ptime_min = int(ptime%3600//60)
    ptime_s = int(ptime%3600%60)
    print(f"実行時間: {ptime_hour} h {ptime_min} m {ptime_s}s")

    print("complete!")