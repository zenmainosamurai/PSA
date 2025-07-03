"""CSV出力後の単位変換処理モジュール（最適化版）"""

import os
import pandas as pd
import numpy as np
from logging import getLogger
import glob
from typing import Dict, Tuple, List, Optional


class CsvUnitConverter:
    """CSVファイルに対する単位変換を行うクラス（最適化版）"""

    def __init__(self, num_streams: int, num_sections: int):
        """
        初期化

        Args:
            num_streams (int): ストリーム数
            num_sections (int): セクション数
        """
        self.logger = getLogger(__name__)
        self.num_streams = num_streams
        self.num_sections = num_sections

        # 変換定数（事前計算）
        self.STANDARD_PRESSURE_PA = 101325  # Pa (1 atm)
        self.STANDARD_TEMPERATURE_K = 273.15  # K (0℃)
        self.PRESSURE_CONVERSION_FACTOR = 1e6  # MPa to Pa
        self.VOLUME_CONVERSION_FACTOR = 1e-6  # cm³ to m³

    def _convert_cm3_to_nm3_vectorized(
        self, volume_cm3: pd.Series, pressure_mpa: pd.Series, temperature_k: pd.Series
    ) -> pd.Series:
        """
        cm³からNm³（標準状態での体積）に変換する（ベクトル化版）

        Args:
            volume_cm3 (pd.Series): 体積 [cm³]
            pressure_mpa (pd.Series): 圧力 [MPa]
            temperature_k (pd.Series): 温度 [K]

        Returns:
            pd.Series: 標準状態での体積 [Nm³]
        """
        pressure_pa = pressure_mpa * self.PRESSURE_CONVERSION_FACTOR
        volume_nm3 = (
            volume_cm3
            * (pressure_pa / self.STANDARD_PRESSURE_PA)
            * (self.STANDARD_TEMPERATURE_K / temperature_k)
            * self.VOLUME_CONVERSION_FACTOR
        )
        return volume_nm3

    def _get_target_columns_for_conversion(self) -> Dict[str, list]:
        """
        単位変換対象のカラム名パターンを取得

        Returns:
            Dict[str, list]: ファイル名をキーとして、変換対象カラム名のパターンのリスト
        """
        # 変換対象: 流入CO2流量、流入N2流量、下流流出CO2流量、下流流出N2流量
        target_patterns = {
            "流入CO2流量.csv": ["inlet_co2_volume"],
            "流入N2流量.csv": ["inlet_n2_volume"],
            "下流流出CO2流量.csv": ["outlet_co2_volume"],
            "下流流出N2流量.csv": ["outlet_n2_volume"],
        }
        return target_patterns

    def convert_tower_csv_files(self, tower_csv_dir: str, tower_num: int) -> None:
        """
        1つの塔のCSVファイルに対して単位変換を適用（最適化版）

        Args:
            tower_csv_dir (str): 塔のCSVディレクトリパス
            tower_num (int): 塔番号
        """
        self.logger.info(f"Starting unit conversion for tower {tower_num}...")

        # 圧力・温度データの事前読み込み
        pressure_file = os.path.join(tower_csv_dir, "others", "total_pressure.csv")
        temp_file = os.path.join(tower_csv_dir, "heat", "セクション到達温度.csv")

        if not os.path.exists(pressure_file):
            self.logger.warning(f"Pressure file not found: {pressure_file}")
            return
        if not os.path.exists(temp_file):
            self.logger.warning(f"Temperature file not found: {temp_file}")
            return

        try:
            df_pressure = pd.read_csv(pressure_file, index_col="timestamp")
            df_temp = pd.read_csv(temp_file, index_col="timestamp")
        except Exception as e:
            self.logger.error(f"Error reading pressure/temperature files for tower {tower_num}: {e}")
            return

        # 変換対象ファイルのパターン
        target_patterns = self._get_target_columns_for_conversion()

        # materialディレクトリ内のファイルを並列処理
        material_dir = os.path.join(tower_csv_dir, "material")
        if not os.path.exists(material_dir):
            self.logger.warning(f"Material directory not found: {material_dir}")
            return

        # 存在するファイルのみを処理対象とする
        files_to_process = []
        for filename, column_patterns in target_patterns.items():
            file_path = os.path.join(material_dir, filename)
            if os.path.exists(file_path):
                files_to_process.append((file_path, column_patterns))

        if not files_to_process:
            self.logger.debug(f"No target files found for tower {tower_num}")
            return

        # ファイル処理（将来的にマルチプロセシングも可能）
        for file_path, column_patterns in files_to_process:
            self._convert_csv_file(file_path, df_pressure, df_temp, column_patterns)

        self.logger.info(f"Unit conversion completed for tower {tower_num}")

    def _convert_csv_file(
        self, csv_file_path: str, df_pressure: pd.DataFrame, df_temp: pd.DataFrame, column_patterns: list
    ) -> None:
        """
        個別のCSVファイルに対して単位変換を適用（ベクトル化最適化版）

        Args:
            csv_file_path (str): CSVファイルパス
            df_pressure (pd.DataFrame): 圧力データ
            df_temp (pd.DataFrame): 温度データ
            column_patterns (list): 変換対象カラムのパターン
        """
        try:
            df = pd.read_csv(csv_file_path, index_col="timestamp")

            # 共通のタイムスタンプを取得
            common_timestamps = df.index.intersection(df_pressure.index).intersection(df_temp.index)
            if len(common_timestamps) == 0:
                self.logger.warning(f"No common timestamps found for {csv_file_path}")
                return

            # データフレームを共通タイムスタンプでフィルタ
            df_filtered = df.loc[common_timestamps]
            pressure_filtered = df_pressure.loc[common_timestamps, "total_pressure"]

            # 変換対象カラムを事前に特定
            target_columns = []
            for pattern in column_patterns:
                for stream in range(1, self.num_streams + 1):
                    for section in range(1, self.num_sections + 1):
                        target_column = f"{pattern}-{stream:03d}-{section:03d}"
                        temp_column = f"temp_reached-{stream:03d}-{section:03d}"

                        if target_column in df.columns and temp_column in df_temp.columns:
                            target_columns.append((target_column, temp_column))

            # ベクトル化された変換処理
            for target_column, temp_column in target_columns:
                # 温度データの取得（°C → K変換）
                temp_data = df_temp.loc[common_timestamps, temp_column] + 273.15

                # 有効なデータのマスクを作成
                volume_data = df_filtered[target_column]
                valid_mask = (
                    pd.notna(volume_data) & (volume_data != 0) & pd.notna(temp_data) & pd.notna(pressure_filtered)
                )

                if not valid_mask.any():
                    continue

                # ベクトル化された単位変換
                valid_volumes = volume_data[valid_mask]
                valid_pressures = pressure_filtered[valid_mask]
                valid_temps = temp_data[valid_mask]

                # ベクトル化された単位変換（事前計算済み定数を使用）
                converted_volumes = self._convert_cm3_to_nm3_vectorized(valid_volumes, valid_pressures, valid_temps)

                # 結果を元のデータフレームに書き戻し
                df.loc[valid_mask.index[valid_mask], target_column] = converted_volumes

            # 変換後のデータを保存
            df.to_csv(csv_file_path)
            self.logger.debug(f"Unit conversion applied to: {csv_file_path}")

        except Exception as e:
            self.logger.error(f"Error converting file {csv_file_path}: {e}")

    def convert_all_towers(self, output_dir: str, num_towers: int, parallel: bool = False) -> None:
        """
        全ての塔のCSVファイルに対して単位変換を適用

        Args:
            output_dir (str): 出力ディレクトリ
            num_towers (int): 塔数
            parallel (bool): 並列処理を使用するかどうか（デフォルト: False）
        """
        self.logger.info("Starting unit conversion (cm³ → Nm³) for gas flow rates...")

        # 存在する塔ディレクトリを事前に特定
        tower_dirs = []
        for tower_num in range(1, num_towers + 1):
            tower_csv_dir = os.path.join(output_dir, "csv", f"tower_{tower_num}")
            if os.path.exists(tower_csv_dir):
                tower_dirs.append((tower_csv_dir, tower_num))
            else:
                self.logger.warning(f"Tower CSV directory not found: {tower_csv_dir}")

        if not tower_dirs:
            self.logger.warning("No tower directories found for unit conversion")
            return

        # 並列処理または逐次処理
        if parallel and len(tower_dirs) > 1:
            try:
                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=min(4, len(tower_dirs))) as executor:
                    futures = [
                        executor.submit(self._convert_tower_worker, tower_csv_dir, tower_num)
                        for tower_csv_dir, tower_num in tower_dirs
                    ]
                    for future in futures:
                        future.result()  # 例外があれば再発生
            except ImportError:
                self.logger.warning("concurrent.futures not available, falling back to sequential processing")
                for tower_csv_dir, tower_num in tower_dirs:
                    self.convert_tower_csv_files(tower_csv_dir, tower_num)
        else:
            # 逐次処理
            for tower_csv_dir, tower_num in tower_dirs:
                self.convert_tower_csv_files(tower_csv_dir, tower_num)

        self.logger.info("Unit conversion completed for all towers.")

    def _convert_tower_worker(self, tower_csv_dir: str, tower_num: int) -> None:
        """
        並列処理用のワーカー関数

        Args:
            tower_csv_dir (str): 塔のCSVディレクトリパス
            tower_num (int): 塔番号
        """
        # 新しいConverterインスタンスを作成（プロセス間共有の問題を回避）
        converter = CsvUnitConverter(self.num_streams, self.num_sections)
        converter.convert_tower_csv_files(tower_csv_dir, tower_num)


def apply_unit_conversion_to_csv_files(
    output_dir: str, num_towers: int, num_streams: int, num_sections: int, parallel: bool = False
) -> None:
    """
    CSV出力後の単位変換を実行する関数（最適化版）

    Args:
        output_dir (str): 出力ディレクトリ
        num_towers (int): 塔数
        num_streams (int): ストリーム数
        num_sections (int): セクション数
        parallel (bool): 並列処理を使用するかどうか（デフォルト: False）
    """
    converter = CsvUnitConverter(num_streams, num_sections)
    converter.convert_all_towers(output_dir, num_towers, parallel)
