# PSAシミュレーター

PSA（Pressure Swing Adsorption）プロセスのシミュレーションを行うためのPythonプログラムです。

## クイックスタート

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# シミュレーション実行
python main.py
```

## ディレクトリ構成

```
PSA/
├── main.py              # エントリーポイント
├── settings.yml        # 実行条件設定ファイル
│
├── common/              # 共通定義
│   ├── constants.py     # 物理定数
│   ├── paths.py         # パス設定
│   └── translations.py  # 日英翻訳辞書
│
├── config/              # 条件ファイル読み込み
│   └── sim_conditions.py
│
├── operation_modes/     # 運転モード別の計算ロジック
│   ├── mode_types.py    # モード定義
│   ├── common.py        # 共通処理
│   ├── flow_adsorption.py      # 流通吸着
│   ├── batch_adsorption.py     # バッチ吸着
│   ├── equalization.py         # 均圧
│   ├── vacuum_desorption.py    # 真空脱着
│   └── ...
│
├── physics/             # 物理計算
│   ├── mass_balance.py  # 物質収支
│   ├── heat_balance.py  # 熱収支
│   ├── pressure.py      # 圧力計算
│   └── ...
│
├── process/             # シミュレーション実行制御
│   ├── simulator.py     # シミュレーター本体
│   ├── simulation_io.py     # 入出力
│   ├── simulation_runner.py # 計算実行
│   ├── result_exporter.py   # 結果出力
│   └── process_executor.py  # 工程実行
│
├── state/               # 状態変数・計算結果
│   ├── state_variables.py   # 状態変数管理
│   └── results.py           # 計算結果データクラス
│
├── utils/               # ユーティリティ
│   ├── plot_csv.py      # CSV出力・グラフ描画
│   ├── plot_xlsx.py     # Excel出力
│   └── prop_table.py    # 蒸気表（CoolProp高速化）
│
├── conditions/          # 実験条件データ
│   └── {cond_id}/
│       ├── sim_conds.xlsx   # 条件ファイル
│       └── 稼働工程表.xlsx   # 稼働工程表
│
├── data/                # 入力データ
│   ├── 3塔データ.csv    # 観測データ
│   └── prop_table.npz   # 蒸気表キャッシュ
│
├── output/              # 出力先
│   └── {cond_id}/
│       ├── csv/         # CSV出力
│       ├── png/         # グラフ出力
│       └── xlsx/        # Excel出力
│
├── tests/               # テスト
├── tools/               # 開発ツール
└── analysis/            # 分析用Notebook
```

## 設定ファイル

### settings.yml

実行モードと対象条件を指定します。

```yaml
mode_list:
  - simulation    # シミュレーション実行

cond_list:
  - 5_08_mod_logging2   # 条件ID
```

### conditions/{cond_id}/sim_conds.xlsx

塔の物理条件（寸法、充填層、フィードガスなど）を定義します。

### conditions/{cond_id}/稼働工程表.xlsx

PSAサイクルの工程（各塔のモードと終了条件）を定義します。

## 主要な使い方

### 1. 基本的な実行

```python
from process import GasAdsorptionBreakthroughSimulator

simulator = GasAdsorptionBreakthroughSimulator("5_08_mod_logging2")
simulator.execute_simulation()
```

### 2. 責務分離した使い方（推奨）

```python
from process import SimulationIO, SimulationRunner, ResultExporter
from state import StateVariables

# 入力
io = SimulationIO()
sim_conds = io.load_conditions("5_08_mod_logging2")
df_operation = io.load_operation_schedule("5_08_mod_logging2")

# 状態変数初期化
state_manager = StateVariables(
    num_towers=3,
    num_streams=sim_conds.get_tower(1).common.num_streams,
    num_sections=sim_conds.get_tower(1).common.num_sections,
    sim_conds=sim_conds,
)

# シミュレーション実行
runner = SimulationRunner(sim_conds, state_manager, df_operation)
output = runner.run()

# 結果出力
exporter = ResultExporter(sim_conds)
exporter.export_all(
    output_dir="output/5_08_mod_logging2/",
    simulation_results=output.results,
    df_operation=df_operation,
    process_completion_log=output.process_completion_log,
)
```

## 運転モード

| モード | 説明 |
|--------|------|
| 停止 | 弁閉止状態 |
| 初回ガス導入 | 初期ガス導入 |
| 流通吸着_単独/上流 | フィードガスを直接導入 |
| 流通吸着_下流 | 上流塔からのガスを導入 |
| バッチ吸着_上流 | バッチ式吸着（上流） |
| バッチ吸着_下流 | バッチ式吸着（下流） |
| バッチ吸着_上流（圧調弁あり） | 圧調弁付きバッチ吸着 |
| バッチ吸着_下流（圧調弁あり） | 圧調弁付きバッチ吸着 |
| 均圧_減圧 | 均圧配管を通じて減圧 |
| 均圧_加圧 | 均圧配管を通じて加圧 |
| 真空脱着 | 真空ポンプで脱着 |

## 出力ファイル

- `csv/tower_{n}/heat/`: 熱収支データ
- `csv/tower_{n}/material/`: 物質収支データ
- `csv/tower_{n}/others/`: 全圧、モル分率、回収量
- `png/tower_{n}/`: グラフ画像
- `xlsx/tower_{n}/`: Excel形式データ

## 開発ツール

```bash
# 蒸気表の再生成（CoolProp呼び出しを高速化）
python tools/build_prop_table.py

# CSV出力の差分チェック
python tools/csv_diff_checker.py output/old/ output/new/
```

## テスト

```bash
# 基本テスト
python tests/test_simulation.py

# フルシミュレーションテスト（出力なし）
python tests/test_full_simulation.py
```
