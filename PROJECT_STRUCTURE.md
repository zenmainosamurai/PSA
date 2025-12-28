# PSAシミュレーター プロジェクト構造

このドキュメントでは、PSA（Pressure Swing Adsorption：圧力スイング吸着）シミュレーターの
プロジェクト構造と各モジュールの役割を説明します。

---

## ディレクトリ構造概要

```
PSA/
├── main.py                 # エントリーポイント
├── settings.yml            # 実行設定ファイル
├── settings.xlsx           # 実行設定（Excel版）
├── requirements.txt        # Python依存パッケージ
│
├── common/                 # 共通定義（定数・パス・翻訳）
├── config/                 # シミュレーション条件の読み込み
├── state/                  # 状態変数・計算結果のデータ構造
├── physics/                # 物理計算（物質収支・熱収支・圧力）
├── operation_modes/        # 運転モード別の処理
├── process/                # シミュレーション実行制御
├── logger/                 # ログ設定
├── utils/                  # ユーティリティ（出力・プロット）
├── tools/                  # 開発支援ツール
├── tests/                  # テストコード
│
├── conditions/             # シミュレーション条件ファイル（入力）
├── data/                   # 物性テーブル等のデータ
└── output/                 # シミュレーション結果（出力）
```

---

## 主要モジュール詳細

### `main.py` - エントリーポイント

シミュレーションの実行を開始するメインスクリプト。

```python
# 使用例
python main.py
```

`settings.yml`から実行モードと条件IDを読み込み、シミュレーションを実行します。

---

### `common/` - 共通定義

プロジェクト全体で使用する定数・パス・翻訳辞書を定義。

| ファイル | 説明 |
|---------|------|
| `constants.py` | 物理定数（気体定数、標準圧力など）、単位変換係数 |
| `paths.py` | ディレクトリパス定義（conditions/, output/, data/） |
| `translations.py` | 日本語変数名と単位の辞書（CSV出力用） |
| `unit_conversion.py` | 単位変換関数 |

---

### `config/` - シミュレーション条件

Excelファイルからシミュレーション条件を読み込む。

| ファイル | 説明 |
|---------|------|
| `sim_conditions.py` | `SimulationConditions`クラス：塔条件、ストリーム条件、供給ガス条件などを管理 |

**主なデータクラス:**
- `TowerConditions`: 塔全体の条件
- `CommonConditions`: 共通設定（セクション数、計算ステップ等）
- `PackedBedConditions`: 充填層条件（吸着材、空隙率等）
- `FeedGasConditions`: 供給ガス条件（流量、組成等）

---

### `state/` - 状態変数・計算結果

シミュレーション中の状態と計算結果を保持するデータ構造。

| ファイル | 説明 |
|---------|------|
| `state_variables.py` | `StateVariables`クラス：各塔・セル（ストリーム×セクション）の状態（温度、圧力、吸着量等）を管理 |
| `results.py` | 計算結果のデータクラス群（`MaterialBalanceResult`, `HeatBalanceResult`等） |

**状態変数の構造:**
```
StateVariables
└── towers[1], towers[2], towers[3]  # 各塔
    └── TowerStateArrays
        ├── temp[stream][section]           # 温度
        ├── loading[stream][section]        # 吸着量
        ├── co2_mole_fraction[stream][section]  # CO2モル分率
        ├── total_press                     # 全圧
        └── ...
```

---

### `physics/` - 物理計算

PSAプロセスの物理現象を計算する中核モジュール。

| ファイル | 説明 |
|---------|------|
| `mass_balance.py` | **物質収支計算**: 吸着/脱着時のCO2・N2の移動量を計算。LDFモデルによる吸着速度 |
| `heat_balance.py` | **熱収支計算**: 吸着熱、ガス流入熱、隣接セルとの熱交換から温度変化を計算 |
| `heat_transfer.py` | **熱伝達係数計算**: Yagi-Kuniiモデルによる充填層の有効熱伝導率 |
| `pressure.py` | **圧力計算**: 真空排気、均圧、バッチ吸着時の圧力変化 |
| `adsorption_isotherm.py` | **吸着平衡線**: 温度・圧力から平衡吸着量を計算（シンボリック回帰式） |

**計算の流れ:**
```
各タイムステップで:
1. 物質収支 (mass_balance) → CO2/N2の吸着・脱着量
2. 熱収支 (heat_balance) → 温度変化
3. 圧力計算 (pressure) → 全圧更新
```

---

### `operation_modes/` - 運転モード

PSA工程の各運転モードを実装。

| ファイル | 運転モード | 説明 |
|---------|-----------|------|
| `initial_gas_introduction.py` | 初回ガス導入 | 空の塔にガスを充填 |
| `flow_adsorption.py` | 流通吸着 | ガスを流しながらCO2を吸着 |
| `batch_adsorption.py` | バッチ吸着 | バルブを閉じて吸着を継続 |
| `equalization.py` | 均圧（加圧/減圧） | 塔間で圧力を移動 |
| `vacuum_desorption.py` | 真空脱着 | 真空引きでCO2を脱着・回収 |
| `stop.py` | 停止 | バルブ閉鎖状態（熱交換のみ） |
| `common.py` | 共通処理 | 全セル計算、壁面熱収支など |
| `mode_types.py` | モード定義 | `OperationMode`列挙型 |

**運転モードの一覧:**
```python
OperationMode:
  INITIAL_GAS_INTRODUCTION  # 初回ガス導入
  STOP                      # 停止
  FLOW_ADSORPTION_UPSTREAM  # 流通吸着（上流）
  FLOW_ADSORPTION_DOWNSTREAM # 流通吸着（下流）
  BATCH_ADSORPTION_UPSTREAM # バッチ吸着（上流）
  BATCH_ADSORPTION_DOWNSTREAM # バッチ吸着（下流）
  EQUALIZATION_PRESSURIZE   # 均圧加圧
  EQUALIZATION_DEPRESSURIZE # 均圧減圧
  VACUUM_DESORPTION         # 真空脱着
```

---

### `process/` - シミュレーション実行制御

シミュレーションの実行フローを制御。

| ファイル | 説明 |
|---------|------|
| `simulator.py` | `GasAdsorptionBreakthroughSimulator`: メインシミュレータークラス（ファサード） |
| `simulation_io.py` | `SimulationIO`: 条件・工程表・観測データの読み込み |
| `simulation_runner.py` | `SimulationRunner`: シミュレーション実行ループ |
| `process_executor.py` | 工程実行ロジック（塔間依存の処理） |
| `result_exporter.py` | `ResultExporter`: 結果のCSV/PNG/XLSX出力 |
| `termination_conditions.py` | 工程終了条件の判定（時間経過、圧力到達等） |
| `simulation_results.py` | シミュレーション結果データ構造 |

**実行フロー:**
```
1. SimulationIO.load_conditions()  # 条件読み込み
2. SimulationIO.load_operation_schedule()  # 工程表読み込み
3. SimulationRunner.run()  # シミュレーション実行
   ├── 工程1: [停止, 停止, 停止]
   ├── 工程2: [初回ガス導入, 停止, 停止]
   ├── ...
   └── 工程N: [流通吸着, 真空脱着, 均圧]
4. ResultExporter.export_all()  # 結果出力
```

---

### `logger/` - ログ設定

| ファイル | 説明 |
|---------|------|
| `log_config.py` | ログフォーマット、ハンドラー設定 |

---

### `utils/` - ユーティリティ

| ファイル | 説明 |
|---------|------|
| `plot_csv.py` | CSV出力、PNGグラフ生成 |
| `plot_xlsx.py` | Excel出力（グラフ付き） |
| `prop_table.py` | **CoolProp高速化**: 物性値テーブルによるモンキーパッチ |
| `init_functions.py` | 初期化ヘルパー関数 |
| `other_utils.py` | その他ユーティリティ |
| `const.py` | 後方互換性のための定数エイリアス |

**prop_table.py の役割:**
CoolPropの`PropsSI`呼び出しは遅いため、事前計算したテーブルから補間することで高速化。
`import utils.prop_table` でモンキーパッチが適用される。

---

### `tools/` - 開発支援ツール

| ファイル | 説明 |
|---------|------|
| `build_prop_table.py` | 物性テーブル生成スクリプト |
| `compare_props.py` | 物性値比較ツール |
| `csv_diff_checker.py` | CSV差分チェッカー |

---

### `tests/` - テストコード

| ファイル | 説明 |
|---------|------|
| `test_simulation.py` | シミュレーション動作確認テスト |
| `test_full_simulation.py` | 全工程シミュレーションテスト |
| `test_index_comparison.py` | インデックス修正前後の比較テスト |

---

## データディレクトリ

### `conditions/` - 入力データ

シミュレーション条件ファイルを格納。

```
conditions/
└── 5_08_mod_logging2/          # 条件ID
    ├── sim_conds.xlsx          # シミュレーション条件
    └── operation_schedule.xlsx # 稼働工程表
```

### `data/` - 物性テーブル

```
data/
└── prop_table.npz    # CoolProp高速化用の物性テーブル
```

### `output/` - 出力データ

```
output/
└── 5_08_mod_logging2/          # 条件IDごとの出力
    ├── csv/                    # CSV形式の結果
    │   └── tower_1/
    │       ├── heat/           # 熱収支結果
    │       ├── material/       # 物質収支結果
    │       └── others/         # その他（圧力等）
    ├── png/                    # グラフ画像
    └── xlsx/                   # Excel形式の結果
```

---

## 設定ファイル

### `settings.yml` - 実行設定

```yaml
mode_list: [simulation]  # 実行モード
cond_list: [5_08_mod_logging2]  # 実行する条件ID
opt_params:
  num_processes: 4
  num_trials: 100
```

### `requirements.txt` - 依存パッケージ

主な依存:
- `numpy`, `pandas`: データ処理
- `scipy`: 最適化計算
- `CoolProp`: 気体物性計算
- `matplotlib`: グラフ生成
- `openpyxl`, `xlsxwriter`: Excel入出力

---

## クイックスタート

```bash
# 1. 依存パッケージインストール
pip install -r requirements.txt

# 2. 物性テーブル生成（初回のみ）
python tools/build_prop_table.py

# 3. シミュレーション実行
python main.py
```

---

## 関連ドキュメント

| ファイル | 説明 |
|---------|------|
| `README.md` | プロジェクト概要 |
| `REFACTORING_PLAN.md` | リファクタリング計画 |
| `UNIT_CALCULATION_ISSUES.md` | 単位計算の問題点 |
| `PRIORITY_ISSUES.md` | 優先度付き課題一覧 |
| `REMAINING_ISSUES.md` | 残課題一覧 |
| `HOW_TO_BUILD_EXE.md` | EXE化手順 |
