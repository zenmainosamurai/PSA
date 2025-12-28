import yaml
import time

from process import GasAdsorptionBreakthroughSimulator
import logger as log

logger = log.logger


def read_settings():
    """
    実行条件の読み込み

    Returns:
        dict: 実行条件
    """
    try:
        with open("./settings.yml", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
        return settings
    except FileNotFoundError as e:
        raise Exception(f"設定ファイル(settings.yml)の読み込み時にエラーが発生: {str(e)}")
    except yaml.YAMLError as e:
        raise Exception(f"設定ファイル(settings.yml)の解析時にエラーが発生: {str(e)}")


def execute_simulation_mode(cond_list):
    """
    シミュレーションモードの実行

    Args:
        cond_list (list): 実行条件リスト
    """
    for cond_id in cond_list:
        logger.info(f"シミュレーション実施中... cond = {cond_id}")
        try:
            instance = GasAdsorptionBreakthroughSimulator(cond_id)
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


def execute_optimize_mode():
    """
    最適化モードの実行
    """
    # TODO: 最適化処理の実装
    logger.info("最適化モード: 未実装")


def execute_calculation(settings):
    """
    計算実行のメイン処理

    Args:
        settings (dict): 実行条件
    """
    logger.info("計算開始")

    for mode in settings["mode_list"]:
        if mode == "simulation":
            execute_simulation_mode(settings["cond_list"])
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
        settings = read_settings()
        start = time.time()
        execute_calculation(settings)
        end = time.time()
        elapsed = end - start
        elapsed_hour = int(elapsed // 3600)
        elapsed_min = int(elapsed % 3600 // 60)
        elapsed_s = int(elapsed % 3600 % 60)
        logger.info(f"実行時間: {elapsed_hour} h {elapsed_min} m {elapsed_s}s")
        logger.info("処理完了")

    except Exception as e:
        logger.exception(str(e))
        raise


if __name__ == "__main__":
    main()
