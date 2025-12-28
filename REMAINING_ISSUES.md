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

### ~~3. 【優先度: 中】シミュレーターの責務が大きすぎる~~

**解決日: 2024-12-28**

シミュレーターを3つのクラスに責務分離:

```python
# process/simulation_io.py（新規）
class SimulationIO:
    """入出力を担当"""
    def load_conditions(cond_id: str) -> SimulationConditions: ...
    def load_operation_schedule(cond_id: str) -> pd.DataFrame: ...
    def load_observation_data(dt: float) -> Optional[pd.DataFrame]: ...

# process/simulation_runner.py（新規）
class SimulationRunner:
    """シミュレーション実行のみを担当"""
    def __init__(self, sim_conds, state_manager, df_operation): ...
    def run(self) -> SimulationOutput: ...

# process/result_exporter.py（新規）
class ResultExporter:
    """結果出力を担当"""
    def export_all(output_dir, results, ...): ...
    def export_csv(output_dir, results): ...
    def export_png(output_dir, ...): ...
    def export_xlsx(output_dir, ...): ...

# process/simulator.py（既存、ファサードとして残す）
class GasAdsorptionBreakthroughSimulator:
    """後方互換性のためのファサード"""
    ...
```

**使用例（新方式）**:
```python
from process import SimulationIO, SimulationRunner, ResultExporter
from state import StateVariables

io = SimulationIO()
sim_conds = io.load_conditions("5_08_mod_logging2")
df_operation = io.load_operation_schedule("5_08_mod_logging2")
df_obs = io.load_observation_data(dt=0.01)

state_manager = StateVariables(...)

runner = SimulationRunner(sim_conds, state_manager, df_operation)
output = runner.run()

exporter = ResultExporter(sim_conds)
exporter.export_all(output_dir, output.results, df_operation, output.process_completion_log, df_obs, output.final_timestamp)
```

---

### ~~4. 【優先度: 中】定数の命名が不統一~~

**解決日: 2024-12-28**

熱収支計算用のモード定数を `HeatCalculationMode` IntEnum として定義し、
`OperationMode` に `heat_mode` プロパティを追加:

```python
# operation_modes/mode_types.py

class HeatCalculationMode(IntEnum):
    """熱収支計算用のモード"""
    ADSORPTION = 0      # 吸着（ガス流通あり）
    VALVE_CLOSED = 1    # 停止（弁閉止、ガス流通なし）
    DESORPTION = 2      # 脱着（真空排気）


class OperationMode(Enum):
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

**使用例**:
```python
mode = OperationMode.FLOW_ADSORPTION_UPSTREAM
heat_mode = mode.heat_mode  # HeatCalculationMode.ADSORPTION (0)
```

---

## 対応優先度まとめ

| 優先度 | 問題 | 対応コスト | 状態 |
|--------|------|-----------|------|
| ~~**高**~~ | ~~`state/` の重複ファイル~~ | ~~中~~ | **解決済み** |
| ~~**高**~~ | ~~`utils/const` と `common/` の重複~~ | ~~低~~ | **解決済み** |
| ~~**中**~~ | ~~シミュレーターの責務過多~~ | ~~高（1日）~~ | **解決済み** |
| ~~**中**~~ | ~~定数の命名不統一~~ | ~~中（2時間）~~ | **解決済み** |

---

## 追加の改善提案（優先度: 低）

### 5. クラス名のタイポ修正

現状:
- `GasAdosorptionBreakthroughsimulator` (タイポあり: Adosorption → Adsorption)

推奨:
- 新名称 `GasAdsorptionBreakthroughSimulator` を使用
- 旧名称はエイリアスとして後方互換性のために維持

### 6. クロスプラットフォーム対応

**解決済み（2024-12-28）**:
- `utils/plot_csv.py` のパス処理を `os.path` に統一
- 日本語フォントのフォールバック機能を追加（Windows/macOS/Linux対応）

---

## 更新履歴

- 2024-12-27: 初版作成
- 2024-12-27: 優先度高の2件を解決済みに更新
- 2024-12-28: 優先度中の2件を解決済みに更新、クロスプラットフォーム対応を追加
