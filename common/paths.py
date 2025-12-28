"""パス設定

入出力ファイルのパスを定義しています。
環境に応じて変更が必要な場合はこのファイルを修正してください。

ディレクトリ構成:
    conditions/{cond_id}/     # 条件ファイル
        sim_conds.xlsx        # シミュレーション条件
        稼働工程表.xlsx        # 運転スケジュール
    
    data/                     # 観測データ
        3塔データ.csv          # 実験データ
    
    output/{cond_id}/         # 出力先
        csv/                  # CSV形式の結果
        png/                  # グラフ画像
        xlsx/                 # Excel形式の結果
"""

import os

# ============================================================
# 入力ディレクトリ
# ============================================================

# 条件ファイルのディレクトリ
# 各条件ID (cond_id) ごとにサブディレクトリが作成される
CONDITIONS_DIR = "conditions/"

# 観測データのディレクトリ
DATA_DIR = "data/"


# ============================================================
# 出力ディレクトリ
# ============================================================

# シミュレーション結果の出力先
OUTPUT_DIR = "output/"

# ログファイルの出力先
LOG_DIR = "output/logs/"


# ============================================================
# ファイル名パターン
# ============================================================

# シミュレーション条件ファイル名
SIM_CONDITIONS_FILENAME = "sim_conds.xlsx"

# 稼働工程表ファイル名
OPERATION_SCHEDULE_FILENAME = "稼働工程表.xlsx"

# 観測データファイル名
OBSERVATION_DATA_FILENAME = "3塔データ.csv"


# ============================================================
# パス生成ヘルパー関数
# ============================================================

def get_condition_dir(cond_id: str) -> str:
    """
    条件IDに対応するディレクトリパスを取得
    
    Args:
        cond_id: 条件ID（例: "5_08_mod_logging2"）
        
    Returns:
        条件ディレクトリのパス
    """
    return os.path.join(CONDITIONS_DIR, cond_id)


def get_sim_conditions_path(cond_id: str) -> str:
    """
    シミュレーション条件ファイルのパスを取得
    
    Args:
        cond_id: 条件ID
        
    Returns:
        sim_conds.xlsx のパス
    """
    return os.path.join(CONDITIONS_DIR, cond_id, SIM_CONDITIONS_FILENAME)


def get_operation_schedule_path(cond_id: str) -> str:
    """
    稼働工程表ファイルのパスを取得
    
    Args:
        cond_id: 条件ID
        
    Returns:
        稼働工程表.xlsx のパス
    """
    return os.path.join(CONDITIONS_DIR, cond_id, OPERATION_SCHEDULE_FILENAME)


def get_output_dir(cond_id: str) -> str:
    """
    出力ディレクトリのパスを取得
    
    Args:
        cond_id: 条件ID
        
    Returns:
        出力ディレクトリのパス
    """
    return os.path.join(OUTPUT_DIR, cond_id)


def get_observation_data_path() -> str:
    """
    観測データファイルのパスを取得
    
    Returns:
        観測データファイルのパス
    """
    return os.path.join(DATA_DIR, OBSERVATION_DATA_FILENAME)
