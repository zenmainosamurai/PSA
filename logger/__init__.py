"""ログ設定モジュール

PSA担当者向け説明:
シミュレーションのログ出力設定を提供します。

主要なエクスポート:
- setup_logger: ログの初期設定
- get_logger: ログインスタンスの取得
"""

from .log_config import setup_logger, get_logger

__all__ = [
    "setup_logger",
    "get_logger",
]
