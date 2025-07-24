import yaml
import time

from core import GasAdosorptionBreakthroughsimulator
import utils.prop_table
import log

logger = log.logger


def read_settings():
    """
    実行条件の読み込み

    Returns:
        dict: 実行条件
    """
    try:
        with open("./main_cond.yml", encoding="utf-8") as f:
            main_cond = yaml.safe_load(f)
        return main_cond
    except FileNotFoundError as e:
        raise Exception(f"設定ファイル(main_cond.yml)の読み込み時にエラーが発生: {str(e)}")
    except yaml.YAMLError as e:
        raise Exception(f"設定ファイル(main_cond.yml)の解析時にエラーが発生: {str(e)}")


def execute_simulation_mode(cond_list):
    """
    シミュレーションモードの実行

    Args:
        cond_list (list): 実行条件リスト
    """
    for cond_id in cond_list:
        logger.info(f"シミュレーション実施中... cond = {cond_id}")
        try:
            instance = GasAdosorptionBreakthroughsimulator(cond_id)
            instance.execute_simulation()
            logger.info(f"シミュレーション完了: cond = {cond_id}")
        except Exception as e:
            raise Exception(f"シミュレーション実行時にエラーが発生 (cond_id: {cond_id}): {str(e)}")


def execute_assimilation_mode():
    """
    データ同化モードの実行
    """
    # TODO: データ同化処理の実装
    logger.info("データ同化モード: 未実装")
    pass


def execute_optimize_mode():
    """
    最適化モードの実行
    """
    # TODO: 最適化処理の実装
    logger.info("最適化モード: 未実装")
    pass


def execute_calculation(main_cond):
    """
    計算実行のメイン処理

    Args:
        main_cond (dict): 実行条件
    """
    logger.info("計算開始")

    for mode in main_cond["mode_list"]:
        if mode == "simulation":
            execute_simulation_mode(main_cond["cond_list"])
        elif mode == "assimilation":
            execute_assimilation_mode()
        elif mode == "optimize":
            execute_optimize_mode()
        else:
            logger.warning(f"未対応のモード: {mode}")


def main():
    """
    メインルーチン
    """
    try:
        logger.info("処理開始")
        main_cond = read_settings()
        start = time.time()
        execute_calculation(main_cond)
        end = time.time()
        ptime = end - start
        ptime_hour = int(ptime // 3600)
        ptime_min = int(ptime % 3600 // 60)
        ptime_s = int(ptime % 3600 % 60)
        logger.info(f"実行時間: {ptime_hour} h {ptime_min} m {ptime_s}s")
        logger.info("処理完了")

    except Exception as e:
        logger.exception(str(e))
        raise


if __name__ == "__main__":
    main()
