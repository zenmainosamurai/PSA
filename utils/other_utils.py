import numpy as np
import pandas as pd

import sys
import os
from logging import getLogger, handlers, Formatter, StreamHandler, INFO


def resample_obs_data(df_obs, simulation_step):
    """観測値を計算粒度に合うようにリサンプリングする

    Args:
        df_obs (pd.DataFrame): 観測値のデータフレーム
        simulation_step (float): シミュレーション時の計算ステップ
    Returns:
        pd.DataFrame: リサンプリング後のデータフレーム
    """
    # 観測値の計算ステップ
    obs_step = df_obs.index[1] - df_obs.index[0]
    # ステップ分割数を計算
    num_split = obs_step / simulation_step
    num_split = int(np.ceil(num_split))  # 切り上げ
    # 新しいindexを用意
    new_index = []
    for idx in df_obs.index:
        new_index.append(idx)
        for i in range(1, num_split):
            new_index.append(idx + i * obs_step / num_split)
    # 新しいindexに基づくリサンプリング
    resampled_data = {
        col: np.interp(new_index, df_obs.index, df_obs[col]) for col in df_obs.columns
    }
    new_df = pd.DataFrame(resampled_data, index=new_index)

    return new_df


def set_logger(log_dir):
    """全体のログ設定
    ファイルに書き出す。ログが100KB溜まったらバックアップにして新しいファイルを作る。
    """
    # ロガーの設定
    root_logger = getLogger()
    root_logger.setLevel(INFO)
    # 保存先の有無チェック
    os.makedirs(log_dir, exist_ok=True)
    # 設定済みならリセット
    if root_logger.handlers:
        while root_logger.handlers:
            handler = root_logger.handlers[0]
            root_logger.removeHandler(handler)
    # ファイル出力用のハンドラ
    rotating_handler = handlers.RotatingFileHandler(
        log_dir + r"/result.log",
        mode="a",
        maxBytes=100 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    format = Formatter("%(asctime)s : %(levelname)s : %(filename)s - %(message)s")
    rotating_handler.setFormatter(format)
    root_logger.addHandler(rotating_handler)

    # 標準出力用のハンドラ
    stream_handler = StreamHandler(sys.stdout)
    stream_handler.setLevel(INFO)
    root_logger.addHandler(stream_handler)
