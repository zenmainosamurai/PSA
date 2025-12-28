# operation_modes/ - 運転モード

PSA工程の各運転モードを実装。

## ファイル一覧

| ファイル | 運転モード | 説明 |
|---------|-----------|------|
| `initial_gas_introduction.py` | 初回ガス導入 | 空の塔にガスを充填 |
| `flow_adsorption.py` | 流通吸着 | ガスを流しながらCO2を吸着 |
| `batch_adsorption.py` | バッチ吸着 | バルブを閉じて吸着を継続 |
| `equalization.py` | 均圧 | 塔間で圧力を移動（加圧/減圧） |
| `vacuum_desorption.py` | 真空脱着 | 真空引きでCO2を脱着・回収 |
| `stop.py` | 停止 | バルブ閉鎖状態（熱交換のみ） |
| `common.py` | 共通処理 | 全セル計算ループ、壁面・蓋の熱収支 |
| `mode_types.py` | モード定義 | OperationMode列挙型、モード分類 |

## 運転モード一覧（mode_types.py）

| モード名 | 説明 |
|---------|------|
| `INITIAL_GAS_INTRODUCTION` | 初回ガス導入 |
| `STOP` | 停止 |
| `FLOW_ADSORPTION_UPSTREAM` | 流通吸着（単独/上流） |
| `FLOW_ADSORPTION_DOWNSTREAM` | 流通吸着（下流） |
| `BATCH_ADSORPTION_UPSTREAM` | バッチ吸着（上流） |
| `BATCH_ADSORPTION_DOWNSTREAM` | バッチ吸着（下流） |
| `BATCH_ADSORPTION_UPSTREAM_WITH_VALVE` | バッチ吸着（上流・圧調弁あり） |
| `BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE` | バッチ吸着（下流・圧調弁あり） |
| `EQUALIZATION_PRESSURIZE` | 均圧加圧 |
| `EQUALIZATION_DEPRESSURIZE` | 均圧減圧 |
| `VACUUM_DESORPTION` | 真空脱着 |

## 各モードの処理内容

### 吸着系モード（flow_adsorption, batch_adsorption）
- 供給ガスからCO2を吸着材に吸着
- 物質収支・熱収支を計算

### 均圧モード（equalization）
- 高圧塔から低圧塔へガスを移動
- 加圧側と減圧側で異なる処理

### 真空脱着モード（vacuum_desorption）
- 真空ポンプで減圧
- 吸着材からCO2を脱着して回収
