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
    instance = GasAdosorption_Breakthrough_simulator(main_cond["cond_list"][0])
    instance.execute_simulation()
    print("complete!")