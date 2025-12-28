# state/ - 状態変数・計算結果

シミュレーション中の状態と計算結果を保持するデータ構造。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `state_variables.py` | 状態変数管理クラス（温度、圧力、吸着量等） |
| `results.py` | 計算結果のデータクラス群 |

## 状態変数の構造（state_variables.py）

```
StateVariables
└── towers[1], towers[2], towers[3]    # 各塔（1オリジン）
    └── TowerStateArrays
        ├── temp[stream][section]              # セル温度 [℃]
        ├── loading[stream][section]           # 吸着量 [cm³/g-abs]
        ├── co2_mole_fraction[stream][section] # CO2モル分率 [-]
        ├── n2_mole_fraction[stream][section]  # N2モル分率 [-]
        ├── total_press                        # 全圧 [MPaA]
        ├── temp_wall[section]                 # 壁温度 [℃]
        ├── lid_temperature                    # 上蓋温度 [℃]
        └── bottom_temperature                 # 下蓋温度 [℃]
```

## 計算結果クラス（results.py）

| クラス | 説明 |
|-------|------|
| `MaterialBalanceResult` | 物質収支結果（流入/流出ガス量、吸着量等） |
| `HeatBalanceResult` | 熱収支結果（温度、熱流束、伝熱係数等） |
| `VacuumPumpingResult` | 真空排気結果（排気量、回収量等） |
| `GasFlow` | ガス流量（CO2/N2体積、モル分率） |
| `GasProperties` | ガス物性（密度、比熱） |
