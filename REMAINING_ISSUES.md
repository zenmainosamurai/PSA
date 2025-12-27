# 残存する構造的問題（優先度: 中以上）

このドキュメントは、リファクタリング後に残っている構造的な問題点をまとめたものです。
今後の保守性向上のために、順次対応することを推奨します。

---

## 1. 【優先度: 高】`state/` に同じ目的のファイルが2つある

### 現状

```
state/
├── results.py              # 旧コード（345行）
├── calculation_results.py  # 新コード（534行）
├── state_variables.py
└── __init__.py
```

### 問題点

- `results.py` と `calculation_results.py` に**ほぼ同じクラスが重複定義**されている
- 以下のクラスが両方に存在:
  - `GasFlow`
  - `GasProperties`
  - `AdsorptionState`
  - `PressureState`
  - `MaterialBalanceResult`
  - `HeatBalanceResult`
  - `WallHeatBalanceResult`
  - `LidHeatBalanceResult`
  - `VacuumPumpingResult`
  - `DepressurizationResult`
  - `DownstreamFlowResult`
  - `DesorptionMoleFractionResult`
- どちらを使うべきか混乱する
- `state/__init__.py` が両方からインポートしているため、名前衝突のリスク

### 現状の使用状況

```python
# results.py のクラス → 物質収支・熱収支のコレクションクラス
MassBalanceResults       # Dict[stream, Dict[section, MaterialBalanceResult]] のラッパー
HeatBalanceResults       # Dict[stream, Dict[section, HeatBalanceResult]] のラッパー
MoleFractionResults      # 脱着時専用

# calculation_results.py のクラス → 新コードで定義したが未使用
TowerResults             # 直感的なアクセサー付き（未使用）
OperationResult          # 運転モード計算結果（未使用）
MassBalanceCalculationResult  # physics/mass_balance.py で使用
```

### 推奨対応

1. **統合方針を決定**
   - `results.py` の `MassBalanceResults`, `HeatBalanceResults` は実際に使用されている
   - `calculation_results.py` の `TowerResults`, `OperationResult` は未使用

2. **対応手順**
   ```
   a. calculation_results.py の未使用クラス（TowerResults, OperationResult）を削除
   b. 基本データクラス（GasFlow等）を1ファイルに統合
   c. results.py の階層的アクセサー（SectionResults, StreamSectionResults等）を
      シンプルな辞書アクセスに置き換えるか、そのまま残すか判断
   d. 最終的に1ファイルに統合するか、役割で分離するか決定
   ```

3. **影響範囲**
   - `operation_modes/*.py` - `MassBalanceResults`, `HeatBalanceResults` を使用
   - `physics/*.py` - `MaterialBalanceResult`, `MassBalanceCalculationResult` を使用
   - `process/process_executor.py` - `MassBalanceResults` を使用

---

## 2. 【優先度: 高】`utils/const.py` と `common/` の重複

### 現状

```python
# utils/const.py（旧コード、130行）
CELSIUS_TO_KELVIN_OFFSET = 273.15
GAS_CONSTANT = 8.314
CONDITIONS_DIR = "conditions/"
OUTPUT_DIR = "output/"
TRANSLATION = {...}   # 英語→日本語変換辞書（45項目）
UNIT = {...}          # 単位辞書（42項目）
OPERATION_MODE = {...}  # 稼働モード辞書

# common/constants.py（新コード）
CELSIUS_TO_KELVIN_OFFSET = 273.15  # 同じ定義
GAS_CONSTANT = 8.314               # 同じ定義

# common/paths.py（新コード）
CONDITIONS_DIR = "conditions/"     # 同じ定義
OUTPUT_DIR = "output/"             # 同じ定義

# common/translations.py（新コード）
TRANSLATION = {...}  # 同じ定義
UNIT = {...}         # 同じ定義
```

### 問題点

- 同じ定数が2箇所に定義されている
- 以下のモジュールが `utils/const` を参照:
  - `process/simulator.py`
  - `utils/plot_csv.py`
  - `utils/plot_xlsx.py`
  - `utils/plot_xlsx_new.py`
- 以下のモジュールが `common/` を参照:
  - `physics/heat_transfer.py` → `common/constants`
  - `physics/*.py` → `common/constants`
- 将来的に値の乖離が起きる可能性

### 推奨対応

**方針A: `utils/const.py` を削除し、`common/` に統合**

```python
# utils/const.py を以下に置き換え（エイリアス化）
from common.constants import *
from common.paths import *
from common.translations import *

# 追加で必要な項目（common/ に未移行のもの）
OPERATION_MODE = {
    1: "初回ガス導入",
    2: "停止",
    ...
}
```

**方針B: 参照元を全て `common/` に変更**

```bash
# 変更が必要なファイル
process/simulator.py:    from utils import const → from common import paths, translations
utils/plot_csv.py:       from utils import const → from common import paths, translations
utils/plot_xlsx.py:      from utils import const → from common import paths, translations
utils/plot_xlsx_new.py:  from utils import const → from common import paths, translations
```

### 影響範囲

- `process/simulator.py` - `const.CONDITIONS_DIR`, `const.OUTPUT_DIR`, `const.DATA_DIR`
- `utils/plot_csv.py` - `const.TRANSLATION`, `const.UNIT`
- `utils/plot_xlsx.py` - `const.TRANSLATION`, `const.UNIT`

---

## 3. 【優先度: 中】シミュレーターの責務が大きすぎる

### 現状

```python
# process/simulator.py（約500行）
class GasAdosorptionBreakthroughsimulator:
    def __init__(self, cond_id: str):
        # 1. 条件ファイル読み込み（sim_conds.xlsx）
        # 2. 観測データ読み込み（3塔データ.csv）
        # 3. 稼働工程表読み込み（稼働工程表.xlsx）
        # 4. 状態変数初期化
    
    def execute_simulation(self, ...):
        # 5. 全工程ループ実行
        # 6. 結果のCSV出力
        # 7. 結果のグラフ出力（PNG）
        # 8. 結果のExcel出力
    
    def _execute_process(self, ...):
        # 工程単位の計算
    
    def calc_adsorption_process(self, ...):
        # 吸着プロセス計算
    
    def _output_results(self, ...):
        # 結果出力（CSV/PNG/XLSX）
```

### 問題点

- 単一クラスに入出力・計算・出力が混在
- 単体テストが困難（ファイルI/Oが必須）
- 出力形式の変更が本体クラスに影響
- 条件読み込みとシミュレーション実行が密結合

### 推奨対応

**責務分離案**

```python
# process/simulation_runner.py（新規）
class SimulationRunner:
    """シミュレーション実行のみを担当"""
    def __init__(self, sim_conds: SimulationConditions, state_manager: StateVariables):
        ...
    
    def run(self) -> SimulationResults:
        """全工程を実行して結果を返す"""
        ...

# process/simulation_io.py（新規）
class SimulationIO:
    """入出力を担当"""
    @staticmethod
    def load_conditions(cond_id: str) -> SimulationConditions:
        ...
    
    @staticmethod
    def load_operation_schedule(cond_id: str) -> pd.DataFrame:
        ...
    
    @staticmethod
    def load_observation_data(filepath: str) -> Optional[pd.DataFrame]:
        ...

# output/result_exporter.py（新規）
class ResultExporter:
    """結果出力を担当"""
    def export_csv(self, results: SimulationResults, output_dir: str):
        ...
    
    def export_png(self, results: SimulationResults, output_dir: str):
        ...
    
    def export_xlsx(self, results: SimulationResults, output_dir: str):
        ...

# process/simulator.py（既存、ファサードとして残す）
class GasAdsorptionBreakthroughSimulator:
    """後方互換性のためのファサード"""
    def __init__(self, cond_id: str):
        self.io = SimulationIO()
        self.sim_conds = self.io.load_conditions(cond_id)
        self.runner = SimulationRunner(self.sim_conds, ...)
        self.exporter = ResultExporter()
    
    def execute_simulation(self, ...):
        results = self.runner.run()
        self.exporter.export_csv(results, ...)
        self.exporter.export_png(results, ...)
```

### メリット

- 各クラスが単一責務になり、テストが容易
- 出力形式の追加・変更が `ResultExporter` のみで完結
- シミュレーション実行のみのテストが可能

### 影響範囲

- `main.py` - 後方互換のファサードを使えば変更不要
- `tests/test_simulation.py` - より詳細なテストが可能に

---

## 4. 【優先度: 中】定数の命名が不統一

### 現状

```python
# operation_modes/common.py - 熱収支計算のモード定数（整数）
MODE_ADSORPTION = 0
MODE_VALVE_CLOSED = 1
MODE_DESORPTION = 2

# operation_modes/mode_types.py - 運転モード（Enum）
class OperationMode(Enum):
    STOP = "停止"
    FLOW_ADSORPTION_UPSTREAM = "流通吸着_単独/上流"
    VACUUM_DESORPTION = "真空脱着"
    ...

# 変換ロジック（operation_modes/common.py）
def _get_heat_mode(mode: OperationMode) -> int:
    if mode == OperationMode.STOP:
        return MODE_VALVE_CLOSED  # 1
    elif mode == OperationMode.VACUUM_DESORPTION:
        return MODE_DESORPTION    # 2
    else:
        return MODE_ADSORPTION    # 0
```

### 問題点

- `MODE_ADSORPTION`（整数）と `OperationMode.FLOW_ADSORPTION_UPSTREAM`（Enum）が混在
- 熱収支計算では整数、工程実行ではEnumを使用
- 変換ロジックが `_get_heat_mode()` に隠れている
- 新しいモードを追加する際に2箇所の修正が必要

### 推奨対応

**方針A: OperationMode に熱計算モードを統合**

```python
# operation_modes/mode_types.py
class HeatCalculationMode(IntEnum):
    """熱収支計算用のモード"""
    ADSORPTION = 0
    VALVE_CLOSED = 1
    DESORPTION = 2

class OperationMode(Enum):
    STOP = "停止"
    FLOW_ADSORPTION_UPSTREAM = "流通吸着_単独/上流"
    VACUUM_DESORPTION = "真空脱着"
    ...
    
    @property
    def heat_mode(self) -> HeatCalculationMode:
        """熱収支計算用のモード番号を取得"""
        if self == OperationMode.STOP:
            return HeatCalculationMode.VALVE_CLOSED
        elif self == OperationMode.VACUUM_DESORPTION:
            return HeatCalculationMode.DESORPTION
        return HeatCalculationMode.ADSORPTION
```

**方針B: マッピング辞書を定義**

```python
# operation_modes/mode_types.py
HEAT_MODE_MAP: Dict[OperationMode, int] = {
    OperationMode.STOP: 1,  # VALVE_CLOSED
    OperationMode.VACUUM_DESORPTION: 2,  # DESORPTION
    # その他は全て 0 (ADSORPTION)
}

def get_heat_mode(mode: OperationMode) -> int:
    return HEAT_MODE_MAP.get(mode, 0)
```

### 影響範囲

- `operation_modes/common.py` - `_get_heat_mode()` を削除
- `physics/heat_balance.py` - モード引数の型を変更（int → HeatCalculationMode）

---

## 対応優先度まとめ

| 優先度 | 問題 | 対応コスト | 推奨タイミング |
|--------|------|-----------|---------------|
| **高** | `state/` の重複ファイル | 中（2-3時間） | 次回リファクタリング時 |
| **高** | `utils/const` と `common/` の重複 | 低（1時間） | 即時対応可能 |
| **中** | シミュレーターの責務過多 | 高（1日） | 機能追加時に段階的に |
| **中** | 定数の命名不統一 | 中（2時間） | 次回リファクタリング時 |

---

## 更新履歴

- 2024-12-27: 初版作成
