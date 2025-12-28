"""ユーティリティモジュール

シミュレーションで使用する補助的な機能を提供します。

モジュール構成:
- const.py: 定数（common/からの再エクスポート、後方互換性用）
- plot_csv.py: CSV出力とグラフ描画
- plot_xlsx.py: Excel出力
- prop_table.py: 蒸気表（CoolProp高速化）
- other_utils.py: その他ユーティリティ（リサンプリング等）

使用例:
    # 定数（後方互換性用、新規コードではcommon/を推奨）
    from utils import const
    print(const.GAS_CONSTANT)
    
    # CSV出力
    from utils.plot_csv import outputs_to_csv
    outputs_to_csv(output_dir, tower_results, common_conds)
    
    # 蒸気表の高速化（main.pyで自動有効化）
    import utils.prop_table  # インポートするだけでCoolPropがパッチされる
"""
