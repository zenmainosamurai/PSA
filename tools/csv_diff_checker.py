import pandas as pd
import numpy as np
from pathlib import Path


def compare_csv_files(dir1, dir2):
    """
    2つのディレクトリ配下のCSVファイルの内容を比較する

    Args:
        dir1 (str): 比較対象ディレクトリ1
        dir2 (str): 比較対象ディレクトリ2
    """
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)

    # 各ディレクトリからCSVファイルを取得
    csv_files1 = list(dir1_path.rglob("*.csv"))
    csv_files2 = list(dir2_path.rglob("*.csv"))

    # 相対パスでCSVファイルのリストを作成
    relative_files1 = {f.relative_to(dir1_path) for f in csv_files1}
    relative_files2 = {f.relative_to(dir2_path) for f in csv_files2}

    print(f"ディレクトリ1のCSVファイル数: {len(relative_files1)}")
    print(f"ディレクトリ2のCSVファイル数: {len(relative_files2)}")

    # ファイル構成の確認
    if relative_files1 != relative_files2:
        print("⚠️  CSVファイルの構成が異なります")
        only_in_dir1 = relative_files1 - relative_files2
        only_in_dir2 = relative_files2 - relative_files1

        if only_in_dir1:
            print(f"ディレクトリ1のみに存在: {only_in_dir1}")
        if only_in_dir2:
            print(f"ディレクトリ2のみに存在: {only_in_dir2}")
        return

    print("✅ CSVファイルの構成は同じです")
    print()

    # 各CSVファイルの内容を比較
    identical_files = 0
    different_files = 0
    error_files = 0

    for relative_path in relative_files1:
        file1 = dir1_path / relative_path
        file2 = dir2_path / relative_path

        try:
            # CSVファイルを読み込み
            try:
                df1 = pd.read_csv(file1)
            except UnicodeDecodeError:
                df1 = pd.read_csv(file1, encoding="shift-jis")
            try:
                df2 = pd.read_csv(file2)
            except UnicodeDecodeError:
                df2 = pd.read_csv(file2, encoding="shift-jis")

            # データフレームの比較
            if df1.equals(df2):
                print(f"✅ {relative_path}: 同一")
                identical_files += 1
            else:
                print(f"❌ {relative_path}: 異なる")
                different_files += 1

                # 詳細な差分情報を表示
                print(f"   ディレクトリ1: {df1.shape} (行数x列数)")
                print(f"   ディレクトリ2: {df2.shape} (行数x列数)")

                # 形状が同じ場合は詳細な差分をチェック
                if df1.shape == df2.shape:
                    # 列名の比較
                    column_names_match = df1.columns.equals(df2.columns)
                    if not column_names_match:
                        print(f"   列名が異なりますが、位置で比較します")
                        # print(f"   ディレクトリ1の列: {list(df1.columns)}")
                        # print(f"   ディレクトリ2の列: {list(df2.columns)}")

                    # 数値データの比較（位置ベース）
                    print(f"   数値的な比較結果:")

                    # df1の数値列のインデックスを取得
                    numeric_cols1 = df1.select_dtypes(include=[np.number]).columns
                    numeric_cols2 = df2.select_dtypes(include=[np.number]).columns

                    # 数値列の位置を取得
                    numeric_positions1 = [df1.columns.get_loc(col) for col in numeric_cols1]
                    numeric_positions2 = [df2.columns.get_loc(col) for col in numeric_cols2]

                    # 共通する位置の数値列を比較
                    common_numeric_positions = set(numeric_positions1) & set(numeric_positions2)

                    if len(common_numeric_positions) > 0:
                        # 各位置ごとに数値比較
                        for pos in sorted(common_numeric_positions):
                            col1_name = df1.columns[pos]
                            col2_name = df2.columns[pos]

                            try:
                                # 位置ベースで列データを取得
                                col1_data = df1.iloc[:, pos]
                                col2_data = df2.iloc[:, pos]

                                # NaNを含む場合の処理
                                mask1 = pd.notna(col1_data)
                                mask2 = pd.notna(col2_data)

                                # 両方とも有効な値がある箇所
                                valid_mask = mask1 & mask2

                                if valid_mask.sum() > 0:
                                    # 完全一致の確認
                                    exact_match = (col1_data.loc[valid_mask] == col2_data.loc[valid_mask]).all()

                                    if exact_match:
                                        print(f"     位置{pos} ('{col1_name}' vs '{col2_name}'): 完全一致 ✅")
                                    else:
                                        # 数値的近似の確認（相対誤差1e-10、絶対誤差1e-15）
                                        numeric_close = np.allclose(
                                            col1_data.loc[valid_mask],
                                            col2_data.loc[valid_mask],
                                            rtol=1e-10,
                                            atol=1e-15,
                                        )

                                        if numeric_close:
                                            print(
                                                f"     位置{pos} ('{col1_name}' vs '{col2_name}'): 数値的に一致（丸め誤差レベル） ≈"
                                            )
                                        else:
                                            # 差分の統計情報
                                            diff = col1_data.loc[valid_mask] - col2_data.loc[valid_mask]
                                            max_abs_diff = np.abs(diff).max()
                                            mean_abs_diff = np.abs(diff).mean()
                                            diff_count = (diff != 0).sum()

                                            print(f"     位置{pos} ('{col1_name}' vs '{col2_name}'): 数値差分あり ❌")
                                            print(f"       - 差分がある行数: {diff_count}/{valid_mask.sum()}")
                                            print(f"       - 最大絶対差分: {max_abs_diff:.2e}")
                                            print(f"       - 平均絶対差分: {mean_abs_diff:.2e}")

                                # NaNの一致確認
                                nan_match = (mask1 == mask2).all()
                                if not nan_match:
                                    nan1_count = (~mask1).sum()
                                    nan2_count = (~mask2).sum()
                                    print(
                                        f"     位置{pos} ('{col1_name}' vs '{col2_name}'): NaN分布が異なる (Dir1: {nan1_count}, Dir2: {nan2_count})"
                                    )

                            except Exception as e:
                                print(f"     位置{pos} ('{col1_name}' vs '{col2_name}'): 比較エラー - {str(e)}")

                    # 非数値データの比較（位置ベース）
                    non_numeric_cols1 = df1.select_dtypes(exclude=[np.number]).columns
                    non_numeric_cols2 = df2.select_dtypes(exclude=[np.number]).columns

                    if len(non_numeric_cols1) > 0 or len(non_numeric_cols2) > 0:
                        print(f"   非数値列の比較:")

                        # 非数値列の位置を取得
                        non_numeric_positions1 = [df1.columns.get_loc(col) for col in non_numeric_cols1]
                        non_numeric_positions2 = [df2.columns.get_loc(col) for col in non_numeric_cols2]

                        # 共通する位置の非数値列を比較
                        common_non_numeric_positions = set(non_numeric_positions1) & set(non_numeric_positions2)

                        for pos in sorted(common_non_numeric_positions):
                            col1_name = df1.columns[pos]
                            col2_name = df2.columns[pos]

                            col1_data = df1.iloc[:, pos]
                            col2_data = df2.iloc[:, pos]

                            string_match = col1_data.equals(col2_data)
                            if string_match:
                                print(f"     位置{pos} ('{col1_name}' vs '{col2_name}'): 完全一致 ✅")
                            else:
                                diff_count = (col1_data != col2_data).sum()
                                print(f"     位置{pos} ('{col1_name}' vs '{col2_name}'): {diff_count}行で差分あり ❌")

                print()

        except Exception as e:
            print(f"⚠️  {relative_path}: 読み込みエラー - {str(e)}")
            error_files += 1

    # 結果サマリー
    print("=" * 50)
    print("比較結果サマリー:")
    print(f"同一ファイル数: {identical_files}")
    print(f"異なるファイル数: {different_files}")
    print(f"エラーファイル数: {error_files}")
    print(f"総ファイル数: {len(relative_files1)}")

    if different_files == 0 and error_files == 0:
        print("🎉 すべてのCSVファイルが同一です！")
    else:
        print("⚠️  一部のCSVファイルに差分またはエラーがあります")


# 実行
if __name__ == "__main__":
    dir1 = "output/5_08_mod_logging2"
    dir2 = "output/5_08_mod_logging2_original"

    compare_csv_files(dir1, dir2)
