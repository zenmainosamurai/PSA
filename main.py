import yaml
import logging

from assimilator import Assimilator

if __name__ == '__main__':

    # try:
    # 実行条件の読み込み
    with open("./main_cond.yml", encoding="utf-8") as f:
        main_cond = yaml.safe_load(f)
    mode_list = main_cond["mode_list"]
    plot_ = main_cond["plot_overcases"]
    cond_list = main_cond["cond_list"]
    obs_list = main_cond["obs_list"]

    print("実行条件----------------------------------")
    print("mode_list = ", mode_list, 
            "\ncondition_list = ", cond_list,
            "\nobs_list = ", obs_list)

    # 計算実行
    print("計算開始----------------------------------")

    num_loop = len(mode_list) * len(cond_list) * len(obs_list)
    if plot_:
        num_loop += len(cond_list)

    current_num = 1
    for mode in mode_list:
        for cond in cond_list:
            for obs_name in obs_list:
                # 進捗
                print(f"{current_num}/{num_loop} ... (mode, cond, obs) = ({mode}, {cond}, {obs_name})")
                # 計算開始
                instance = Assimilator(obs_name, cond_id=cond, mode=mode)
                instance.execute_assimilation()
                # 番号更新
                current_num += 1

    print("complete!")