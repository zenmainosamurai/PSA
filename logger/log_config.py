"""ログ設定

シミュレーション実行時のログ出力を設定します。
ログは標準出力とファイルの両方に出力されます。

ログファイルの出力先: output/logs/YYYYMMDD-HHMMSS.log

使用例:
    from logger import logger
    
    logger.info("シミュレーション開始")
    logger.warning("警告メッセージ")
    logger.error("エラーメッセージ")
"""

import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler


# ============================================================
# 設定値
# ============================================================

# プロジェクト識別子（ログの名前空間）
PROJECT_ID = "PJ493"

# デフォルトのログレベル
DEFAULT_CONSOLE_LOG_LEVEL = logging.INFO
DEFAULT_FILE_LOG_LEVEL = logging.INFO

# ログ出力フォーマット
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

# ログファイル出力先ディレクトリ
LOG_DIR = "output/logs"


# ============================================================
# ログ設定関数
# ============================================================

def setup_logger(
    console_log_level: int = DEFAULT_CONSOLE_LOG_LEVEL,
    file_log_level: int = DEFAULT_FILE_LOG_LEVEL,
) -> logging.Logger:
    """
    ロガーを設定して返す
    
    Args:
        console_log_level: コンソール出力の最小ログレベル
        file_log_level: ファイル出力の最小ログレベル
        
    Returns:
        設定済みのロガー
    """
    # ロガーを取得（プロジェクトID名前空間）
    logger = logging.getLogger(PROJECT_ID)
    logger.setLevel(min(console_log_level, file_log_level))
    logger.propagate = False
    
    # 既存のハンドラをクリア（重複防止）
    if logger.handlers:
        logger.handlers.clear()
    
    # フォーマッター
    formatter = logging.Formatter(LOG_FORMAT)
    
    # コンソールハンドラ（標準出力）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラ（ログファイル）
    os.makedirs(LOG_DIR, exist_ok=True)
    log_filename = time.strftime("%Y%m%d-%H%M%S") + ".log"
    log_filepath = os.path.join(LOG_DIR, log_filename)
    
    file_handler = TimedRotatingFileHandler(
        filename=log_filepath,
        when="midnight",
        backupCount=31,  # 31日分保持
        encoding="utf-8",
    )
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger() -> logging.Logger:
    """
    設定済みロガーを取得
    
    Returns:
        設定済みのロガー（未設定の場合は自動で初期化）
    """
    logger = logging.getLogger(PROJECT_ID)
    if not logger.handlers:
        return setup_logger()
    return logger


def get_child_logger(parent_logger: logging.Logger, name: str) -> logging.Logger:
    """
    子ロガーを取得
    
    モジュール単位でロガーを分けたい場合に使用します。
    
    Args:
        parent_logger: 親ロガー
        name: 子ロガーの名前（通常は __name__ を使用）
        
    Returns:
        子ロガー
        
    使用例:
        from logger import logger, get_child_logger
        
        module_logger = get_child_logger(logger, __name__)
        module_logger.info("モジュール固有のログ")
    """
    return parent_logger.getChild(name)
