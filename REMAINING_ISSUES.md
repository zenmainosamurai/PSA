# 残存する構造的問題（優先度: 中以上）

このドキュメントは、リファクタリング後に残っている構造的な問題点をまとめたものです。
今後の保守性向上のために、順次対応することを推奨します。

---

## 解決済みの問題

### ~~1. 【優先度: 高】`state/` に同じ目的のファイルが2つある~~

**解決日: 2024-12-27**

- `state/calculation_results.py` を削除
- `MassBalanceCalculationResult` を `state/results.py` に移動
- 未使用クラス（`TowerResults`, `OperationResult`）を削除

現在の構成:
```
state/
├── results.py          # 全計算結果クラス
├── state_variables.py  # 状態変数管理
└── __init__.py         # エクスポート定義
```

---

### ~~2. 【優先度: 高】`utils/const.py` と `common/` の重複~~

**解決日: 2024-12-27**

- `utils/const.py` を `common/` からの再エクスポートに変更
- 定数の一元管理を実現（`common/` が正）
- 後方互換性を維持

---

## 未解決の問題

### 3. 【優先度: 中】シミュレーターの責務が大きすぎる

#### 現状

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

#### 問題点

- 単一クラスに入出力・計算・出力が混在
- 単体テストが困難（ファイルI/Oが必須）
- 出力形式の変更が本体クラスに影響
- 条件読み込みとシミュレーション実行が密結合

#### 推奨対応

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

#### メリット

- 各クラスが単一責務になり、テストが容易
- 出力形式の追加・変更が `ResultExporter` のみで完結
- シミュレーション実行のみのテストが可能

#### 影響範囲

- `main.py` - 後方互換のファサードを使えば変更不要
- `tests/test_simulation.py` - より詳細なテストが可能に

---

### 4. 【優先度: 中】定数の命名が不統一

#### 現状

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

#### 問題点

- `MODE_ADSORPTION`（整数）と `OperationMode.FLOW_ADSORPTION_UPSTREAM`（Enum）が混在
- 熱収支計算では整数、工程実行ではEnumを使用
- 変換ロジックが `_get_heat_mode()` に隠れている
- 新しいモードを追加する際に2箇所の修正が必要

#### 推奨対応

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

#### 影響範囲

- `operation_modes/common.py` - `_get_heat_mode()` を削除
- `physics/heat_balance.py` - モード引数の型を変更（int → HeatCalculationMode）

---

## 対応優先度まとめ

| 優先度 | 問題 | 対応コスト | 状態 |
|--------|------|-----------|------|
| ~~**高**~~ | ~~`state/` の重複ファイル~~ | ~~中~~ | **解決済み** |
| ~~**高**~~ | ~~`utils/const` と `common/` の重複~~ | ~~低~~ | **解決済み** |
| **中** | シミュレーターの責務過多 | 高（1日） | 未着手 |
| **中** | 定数の命名不統一 | 中（2時間） | 未着手 |

---

## 更新履歴

- 2024-12-27: 初版作成
- 2024-12-27: 優先度高の2件を解決済みに更新
