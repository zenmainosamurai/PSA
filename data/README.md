# data/ - データファイル

シミュレーションで使用するデータファイル。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `prop_table.npz` | CoolProp高速化用の物性テーブル |

## prop_table.npz - 物性テーブル

`tools/build_prop_table.py`で生成される物性値の事前計算テーブル。

**含まれるデータ:**
- `T`: 温度グリッド [K]
- `P_FIXED`: 固定圧力 [Pa]
- `co2_l`: CO2熱伝導率 [W/(m·K)]
- `co2_v`: CO2粘度 [Pa·s]
- `co2_cpmass`: CO2質量比熱 [J/(kg·K)]
- `co2_d`: CO2密度 [kg/m³]
- `nitrogen_l`, `nitrogen_v`, `nitrogen_cpmass`, `nitrogen_d`: N2の各物性

**生成方法:**
```bash
python tools/build_prop_table.py
```

**使用方法:**
`import utils.prop_table`でモンキーパッチが適用され、
CoolPropの`PropsSI`呼び出しが自動的に高速化される。
