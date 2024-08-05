import yaml
import time

from models import GasAdosorption_Breakthrough_simulator
from models_opt import GasAdosorption_for_Optimize

if __name__ == '__main__':

    # try:
    # 実行条件の読み込み
    with open("./main_cond.yml", encoding="utf-8") as f:
        main_cond = yaml.safe_load(f)

    start = time.time() # 時間計測

    # 計算実行    
    for cond_id in main_cond["cond_list"]:
        print("cond = ", cond_id)
        instance = GasAdosorption_for_Optimize(cond_id)
        instance.optimize_params()
    
    end = time.time() # 時間計測
    ptime = end - start
    ptime_hour = int(ptime//3600)
    ptime_min = int(ptime%3600//60)
    ptime_s = int(ptime%3600%60)
    print(f"実行時間: {ptime_hour} h {ptime_min} m {ptime_s}s")
    print("complete!")