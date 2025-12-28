"""シミュレーション入出力モジュール

PSA担当者向け説明:
シミュレーションに必要なファイルの読み込み処理を提供します。

- 条件ファイル (sim_conds.xlsx) の読み込み
- 稼働工程表 (稼働工程表.xlsx) の読み込み
- 観測データ (3塔データ.csv) の読み込み

使用例:
    from process.simulation_io import SimulationIO
    
    io = SimulationIO()
    sim_conds = io.load_conditions("5_08_mod_logging2")
    df_operation = io.load_operation_schedule("5_08_mod_logging2")
    df_obs = io.load_observation_data(dt=0.01)
"""

from typing import Optional
import pandas as pd

from config.sim_conditions import SimulationConditions
from common.paths import CONDITIONS_DIR, DATA_DIR
from utils.other_utils import resample_obs_data
import logger as log


class SimulationIO:
    """シミュレーション入出力クラス
    
    PSA担当者向け説明:
    シミュレーションに必要なファイルの読み込みを担当します。
    条件ファイル、稼働工程表、観測データを読み込みます。
    """
    
    def __init__(self):
        """初期化"""
        self.logger = log.logger.getChild(__name__)
    
    def load_conditions(self, cond_id: str) -> SimulationConditions:
        """
        条件ファイルを読み込む
        
        Args:
            cond_id: 条件ID (例: "5_08_mod_logging2")
            
        Returns:
            SimulationConditions: シミュレーション条件
        """
        return SimulationConditions(cond_id)
    
    def load_operation_schedule(self, cond_id: str) -> pd.DataFrame:
        """
        稼働工程表を読み込む
        
        Args:
            cond_id: 条件ID
            
        Returns:
            pd.DataFrame: 稼働工程表（工程番号をインデックスに持つ）
        """
        filepath = f"{CONDITIONS_DIR}{cond_id}/稼働工程表.xlsx"
        return pd.read_excel(filepath, index_col="工程", sheet_name="工程")
    
    def load_observation_data(
        self,
        dt: float,
        filepath: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        観測データを読み込む
        
        PSA担当者向け説明:
        観測データファイル（3塔データ.csv）を読み込み、
        計算時間刻み(dt)に合わせてリサンプリングします。
        ファイルが存在しない場合はNoneを返します。
        
        Args:
            dt: 計算時間刻み（分）
            filepath: 観測データファイルパス（省略時はデフォルト）
            
        Returns:
            pd.DataFrame or None: 観測データ（存在しない場合はNone）
        """
        if filepath is None:
            filepath = f"{DATA_DIR}3塔データ.csv"
        
        try:
            if filepath.lower().endswith("csv"):
                df = pd.read_csv(filepath, index_col=0)
            else:
                df = pd.read_excel(filepath, index_col="time")
            
            # リサンプリング
            return resample_obs_data(df, dt)
            
        except FileNotFoundError:
            self.logger.warning(f"観測データファイルが存在しないため比較をスキップします: {filepath}")
            return None
        except Exception as exc:
            self.logger.error(f"観測データの読み込みに失敗しましたが処理を継続します: {exc}")
            return None
