import os
import pandas as pd
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
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)

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
                    if not df1.columns.equals(df2.columns):
                        print(f"   列名が異なります")
                        print(f"   ディレクトリ1の列: {list(df1.columns)}")
                        print(f"   ディレクトリ2の列: {list(df2.columns)}")
                    else:
                        # 値の違いを確認
                        diff_mask = df1 != df2
                        if diff_mask.any().any():
                            print(f"   値が異なる箇所があります")
                            # 最初の数行の差分を表示
                            for col in df1.columns:
                                if diff_mask[col].any():
                                    diff_rows = diff_mask[col].sum()
                                    print(f"   列'{col}': {diff_rows}行で差分あり")

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
