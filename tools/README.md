# tools/ - 開発支援ツール

開発・デバッグ用のスクリプト。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `build_prop_table.py` | 物性テーブル生成（CoolProp高速化用） |
| `compare_props.py` | 物性値の比較ツール |
| `csv_diff_checker.py` | CSV差分チェッカー |

## build_prop_table.py - 物性テーブル生成

CoolProp高速化のための物性テーブルを生成。

**生成されるファイル:** `data/prop_table.npz`

**含まれる物性値:**
- L: 熱伝導率 [W/(m·K)]
- V: 粘度 [Pa·s]
- CPMASS: 質量比熱 [J/(kg·K)]
- D: 密度 [kg/m³]

**対象流体:** CO2, Nitrogen

**実行方法:**
```bash
python tools/build_prop_table.py
```

## csv_diff_checker.py - CSV差分チェック

2つのCSVファイルの差分を確認。
リファクタリング前後の結果比較に使用。
