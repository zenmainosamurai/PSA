# utils/ - ユーティリティ

出力処理、プロット、高速化などの補助機能。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `plot_csv.py` | CSV出力、PNGグラフ生成 |
| `plot_xlsx.py` | Excel出力（グラフ付き） |
| `plot_xlsx_new.py` | Excel出力（新形式） |
| `prop_table.py` | CoolProp高速化（物性テーブル補間） |
| `props_logger.py` | 物性値呼び出しのログ記録 |
| `init_functions.py` | 初期化ヘルパー関数 |
| `other_utils.py` | その他ユーティリティ |
| `const.py` | 後方互換性のための定数エイリアス |
| `custom_filter.py` | カスタムフィルター |

## prop_table.py - CoolProp高速化

CoolPropの`PropsSI`関数は呼び出しコストが高い。
事前計算した物性テーブルから線形補間することで高速化。

**仕組み:**
1. `tools/build_prop_table.py`で物性テーブル生成（data/prop_table.npz）
2. `import utils.prop_table`でモンキーパッチ適用
3. `PropsSI`呼び出し時にテーブルから補間（ヒット時は高速）

**効果:** 約87%のヒット率で大幅な高速化

## 出力ファイル構成

### CSV出力（plot_csv.py）
```
output/{cond_id}/csv/tower_{n}/
├── heat/           # 熱収支結果
├── material/       # 物質収支結果
└── others/         # その他（圧力、温度等）
```

### PNG出力（plot_csv.py）
```
output/{cond_id}/png/tower_{n}/
├── heat_tower{n}.png
├── material_tower{n}.png
└── others_tower{n}.png
```

### Excel出力（plot_xlsx.py）
```
output/{cond_id}/xlsx/
└── results_tower{n}.xlsx
```
