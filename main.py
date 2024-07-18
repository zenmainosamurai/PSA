import yaml
import logging
import pprint

from models import GasAdosorption_Breakthrough_simulator

if __name__ == '__main__':

    # try:
    # 実行条件の読み込み
    with open("./main_cond.yml", encoding="utf-8") as f:
        main_cond = yaml.safe_load(f)

    # 計算実行
    print("計算開始----------------------------------")
    for cond_id in main_cond["cond_list"]:
        print("cond = ", cond_id)
        instance = GasAdosorption_Breakthrough_simulator(cond_id)
        instance.execute_simulation()
    print("complete!")