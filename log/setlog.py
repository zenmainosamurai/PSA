import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler

# プロジェクト番号
PJ_NO = "PJ493"

# 表示したい最小のログレベル
DEFAULT_SH_LOG_LEVEL = logging.INFO
DEFAULT_FH_LOG_LEVEL = logging.INFO

# 出力書式設定
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

# ローカルログファイルの出力設定
# プロジェクトの出力ディレクトリ構造に合わせてlogsサブディレクトリに出力
LOG_FILENAME = time.strftime("%Y%m%d-%H%M%S") + ".log"
DIR_LOGFILE = "output/logs"


def setup_logger(
    streamhandler_log_level=DEFAULT_SH_LOG_LEVEL,
    filehandler_log_level=DEFAULT_FH_LOG_LEVEL,
):
    """
    標準出力用にloggerを作成

    Args:
        streamhandler_log_level (int, optional): 標準出力で表示される最小のログレベル。
            デフォルトはDEFAULT_SH_LOG_LEVEL
        filehandler_log_level (int, optional): ファイル出力で表示される最小のログレベル。
            デフォルトはDEFAULT_FH_LOG_LEVEL

    Returns:
        logger (logging.Logger)
    """
    # logging.getLogger()を使用し、ロガーに名前（プロジェクト番号）をつける。
    logger = logging.getLogger(PJ_NO)
    logger.setLevel(streamhandler_log_level)
    logger.propagate = False

    # 出力書式設定
    formatter = logging.Formatter(LOG_FORMAT)

    # ハンドラに標準出力を追加
    sh = logging.StreamHandler()
    sh.setLevel(streamhandler_log_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    os.makedirs(DIR_LOGFILE, exist_ok=True)

    # ハンドラにファイル出力を追加
    fh = TimedRotatingFileHandler(
        filename=os.path.join(DIR_LOGFILE, LOG_FILENAME),
        when="midnight",
        backupCount=31,
        encoding="utf-8",
    )
    fh.setLevel(filehandler_log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
