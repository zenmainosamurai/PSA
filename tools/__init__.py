"""開発ツール

開発・デバッグ用のツールを提供します。
シミュレーション本体では使用しません。

ツール一覧:
- build_prop_table.py: 蒸気表（prop_table.npz）の生成
- compare_props.py: プロパティ計算の比較
- csv_diff_checker.py: CSV出力の差分チェック

使用例:
    # 蒸気表の生成
    python tools/build_prop_table.py
    
    # CSV差分チェック
    python tools/csv_diff_checker.py output/old/ output/new/
"""
