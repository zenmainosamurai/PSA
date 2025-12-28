# config/ - シミュレーション条件

Excelファイルからシミュレーション条件を読み込むモジュール。

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `sim_conditions.py` | 条件ファイル読み込み、データクラス定義 |

## 主なクラス

| クラス | 説明 |
|-------|------|
| `SimulationConditions` | 全塔の条件を管理するメインクラス |
| `TowerConditions` | 1塔分の条件（充填層、容器、供給ガス等） |
| `CommonConditions` | 共通設定（セクション数、計算ステップ時間等） |
| `PackedBedConditions` | 充填層条件（吸着材密度、空隙率、粒子径等） |
| `VesselConditions` | 容器条件（直径、高さ、壁厚等） |
| `FeedGasConditions` | 供給ガス条件（流量、組成、温度等） |
| `StreamConditions` | ストリーム条件（断面積、吸着材量等） |

## 入力ファイル

`conditions/{cond_id}/sim_conds.xlsx` から読み込み。
