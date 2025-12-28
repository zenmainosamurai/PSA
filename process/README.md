# process/ - シミュレーション実行制御

シミュレーションの実行フローを制御するモジュール。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `simulator.py` | メインシミュレータークラス（ファサード） |
| `simulation_io.py` | 入出力（条件・工程表・観測データの読み込み） |
| `simulation_runner.py` | シミュレーション実行ループ |
| `process_executor.py` | 工程実行ロジック（塔間依存の処理） |
| `result_exporter.py` | 結果出力（CSV/PNG/XLSX） |
| `termination_conditions.py` | 工程終了条件の判定 |
| `simulation_results.py` | シミュレーション結果データ構造 |

## 主なクラス

### GasAdsorptionBreakthroughSimulator（simulator.py）
シミュレーション全体を管理するファサードクラス。
条件読み込み → 実行 → 結果出力 を一括で行う。

### SimulationIO（simulation_io.py）
- `load_conditions()`: 条件ファイル読み込み
- `load_operation_schedule()`: 稼働工程表読み込み
- `load_observation_data()`: 観測データ読み込み

### SimulationRunner（simulation_runner.py）
- `run()`: 全工程を順番に実行
- 各工程の終了条件を判定しながらループ

### ResultExporter（result_exporter.py）
- `export_csv()`: CSV形式で出力
- `export_png()`: グラフをPNG画像で出力
- `export_xlsx()`: Excel形式で出力

## 終了条件の種類（termination_conditions.py）

| 条件 | 例 |
|-----|---|
| 時間経過 | `時間経過_0.4_min` |
| 圧力到達 | `圧力到達_塔1_0.1_MPaA` |
| 温度到達 | `温度到達_塔1_48` |

## 実行フロー

```
1. 条件ファイル読み込み（sim_conds.xlsx）
2. 稼働工程表読み込み（operation_schedule.xlsx）
3. 工程ループ
   ├── 工程1: [停止, 停止, 停止] → 終了条件まで実行
   ├── 工程2: [初回ガス導入, 停止, 停止]
   ├── ...
   └── 工程N: [流通吸着, 真空脱着, 均圧]
4. 結果出力（CSV/PNG/XLSX）
```
