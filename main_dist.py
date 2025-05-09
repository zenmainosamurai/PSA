import pandas as pd
import time

from utils import const
from utils.other_utils import set_logger, getLogger
from simulator import GasAdosorptionBreakthroughsimulator

if __name__ == "__main__":

    # 実行条件の読み込み
    main_cond = pd.read_excel("main_cond.xlsx", index_col=0)

    # 計算実行
    for cond_id in main_cond.index:
        print(f"cond_name = {cond_id}")

        # ロガーの作成
        set_logger(log_dir=const.OUTPUT_DIR + cond_id + "/")
        logger = getLogger(cond_id)
        logger.info("start -----------------------------------")

        # 時間計測(開始)
        start = time.time()

        # シミュレーション開始
        try:
            instance = GasAdosorptionBreakthroughsimulator(cond_id)
            instance.execute_simulation()
            logger.info("complete!")
        except Exception as e:
            logger.error(f"エラーが発生したため処理を中断します: \n{e}")
            logger.error("break")

        # 時間計測(終了)
        end = time.time()
        ptime = end - start
        ptime_hour = int(ptime // 3600)
        ptime_min = int(ptime % 3600 // 60)
        ptime_s = int(ptime % 3600 % 60)
        logger.info(f"実行時間: {ptime_hour} h {ptime_min} m {ptime_s}s")
