# PSAシミュレーター 優先度「中」以上の問題点と改善案

## 概要

PSA担当部署の方がコードを理解しやすくするための改善項目です。
「〇〇の計算はどこにあるか」がすぐわかる構造を目指しています。

---

## 優先度「高」の問題

### 1. Strategyパターンの過剰適用

#### 現状の問題

**対象ファイル:** `core/physics/mass_balance_strategies.py`

```python
# 現状: 3種類の計算に抽象クラス + 3つの実装クラス
class MassBalanceStrategy(ABC):
    @abstractmethod
    def calculate(...) -> MassBalanceCalculationResult:
        pass

class AdsorptionStrategy(MassBalanceStrategy): ...
class DesorptionStrategy(MassBalanceStrategy): ...
class ValveClosedStrategy(MassBalanceStrategy): ...
```

**問題点:**
- 「吸着の物質収支計算はどこ？」→ Strategy → base_models と2段階たどる必要がある
- 「Strategy」「Abstract」はソフトウェア用語であり、PSA担当者には馴染みがない
- 3種類しかない分岐にデザインパターンを適用する必要性が低い

#### 改善案

**移行先:** `physics/mass_balance.py`

```python
"""物質収支計算モジュール

PSA担当者向け説明:
- 吸着モード: CO2が吸着材に吸着される際の物質収支
- 脱着モード: 真空引きでCO2が脱着される際の物質収支
- 停止モード: バルブ閉鎖時（物質移動なし）
"""

def calculate_mass_balance(
    mode: OperationMode,
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    ...
) -> MassBalanceCalculationResult:
    """物質収支を計算する"""
    if mode in ADSORPTION_MODES:
        return _calculate_adsorption_mass_balance(...)
    elif mode == OperationMode.VACUUM_DESORPTION:
        return _calculate_desorption_mass_balance(...)
    elif mode == OperationMode.STOP:
        return _calculate_valve_closed_mass_balance(...)
```

**改善ポイント:**
- 抽象クラス・継承を廃止し、シンプルな関数に
- 1ファイルで物質収支計算が完結
- 「〇〇の計算はどこ？」→ `physics/mass_balance.py` と即答できる

---

### 2. Results系クラスの多層構造

#### 現状の問題

**対象ファイル:** `core/state/results.py`

```python
# 現状: 4層のクラス階層
class SectionResults: ...
class MaterialBalanceSectionResults(SectionResults): ...
class MaterialBalanceStreamSectionResults(StreamSectionResults): ...
class MassBalanceResults:
    def get_result(self, stream_id, section_id):
        return self.material_balance_stream_section_results.get_material_balance_result(...)
```

**問題点:**
- 「セクション3の吸着量は？」という単純な質問に4クラスの関係理解が必要
- 長いクラス名: `MaterialBalanceStreamSectionResults`
- 同じパターンが `HeatBalance`, `MoleFraction` で3回繰り返される

#### 改善案

**移行先:** `state/calculation_results.py`（作成済み）

```python
@dataclass
class TowerResults:
    """
    1塔の計算結果
    
    使用例:
        temp = results.get_temperature(stream=1, section=3)
        mb = results.get_mass_balance(stream=1, section=3)
    """
    _mass_balance: Dict[Tuple[int, int], MaterialBalanceResult]
    _heat_balance: Dict[Tuple[int, int], HeatBalanceResult]
    
    def get_temperature(self, stream: int, section: int) -> float:
        """セクション温度を取得 [℃]"""
        return self._heat_balance[(stream, section)].bed_temperature
```

**改善ポイント:**
- 4層の継承階層 → 1クラス `TowerResults` に統合
- `get_temperature(stream, section)` のような直感的なメソッド

---

## 優先度「中」の問題

### 3. CellCalculatorクラスの不要なクラス化

#### 現状の問題

**対象ファイル:** `core/physics/operation_models.py`

```python
class CellCalculator:
    @staticmethod
    def calculate_mass_and_heat_balance(...): ...
    
    @staticmethod
    def calculate_wall_heat_balance(...): ...

# 使用側: インスタンス化に意味がない
calculator = CellCalculator()
balance_results = calculator.calculate_mass_and_heat_balance(...)
```

**問題点:**
- 全メソッドが `@staticmethod` → クラスにする意味がない
- 「セル計算」という名前だが、壁・蓋の計算も含む

#### 改善案

クラスを廃止し、各モジュールに関数として配置:

```python
# physics/heat_balance.py
def calculate_bed_heat_balance(...) -> HeatBalanceResult: ...
def calculate_wall_heat_balance(...) -> WallHeatResult: ...

# physics/mass_balance.py
def calculate_mass_balance(...) -> MassBalanceCalculationResult: ...
```

---

### 4. PhysicsCalculatorの過度な抽象化

#### 現状の問題

**対象ファイル:** `core/physics/adsorption_base_models.py` (末尾)

```python
class OperationModePhysicsCalculator(ABC):
    @abstractmethod
    def calculate_adsorption_heat(...): ...

class AdsorptionPhysicsCalculator(OperationModePhysicsCalculator): ...
class ValveClosedPhysicsCalculator(OperationModePhysicsCalculator): ...  # 全て0を返すだけ
class DesorptionPhysicsCalculator(OperationModePhysicsCalculator): ...  # 吸着とほぼ同じ
```

**問題点:**
- 3メソッドだけの抽象クラスに3つの実装クラス → 過剰
- 1500行ファイルの末尾に隠れており発見しづらい
- 停止モードは全て0を返すだけ、吸着と脱着はほぼ同じ

#### 改善案

**移行先:** `physics/heat_balance.py` 内のヘルパー関数

```python
def _get_adsorption_heat(
    mode: OperationMode, 
    material_output: MaterialBalanceResult, 
    tower_conds: TowerConditions
) -> float:
    """吸着熱を計算 [J]"""
    if mode == OperationMode.STOP:
        return 0.0
    # 吸着・脱着は同じ計算式
    return (material_output.actual_uptake_volume / 1000 
            / STANDARD_MOLAR_VOLUME
            * tower_conds.feed_gas.co2_molecular_weight
            * tower_conds.feed_gas.co2_adsorption_heat)
```

---

### 5. 運転モードのEnum化

#### 現状の問題

**対象ファイル:** `core/simulator.py`, `core/state/state_variables.py`

```python
# 複数ファイルで同じ文字列比較が重複
if mode == "初回ガス導入":
    ...
elif mode == "停止":
    ...
elif mode == "流通吸着_単独/上流":
    ...
```

**問題点:**
- 文字列比較のためタイポしても検出不可
- 同じ分岐が複数箇所に重複
- 新モード追加時に複数箇所を修正必要

#### 改善案

**移行先:** `operation_modes/mode_types.py`（作成済み）

```python
class OperationMode(Enum):
    INITIAL_GAS_INTRODUCTION = "初回ガス導入"
    STOP = "停止"
    FLOW_ADSORPTION_UPSTREAM = "流通吸着_単独/上流"
    ...
    
    @classmethod
    def from_japanese(cls, name: str) -> "OperationMode":
        """日本語名からEnumに変換"""
        for mode in cls:
            if mode.value == name:
                return mode
        raise ValueError(f"未対応の運転モード: {name}")

# モードのグループ分け
ADSORPTION_MODES = {OperationMode.INITIAL_GAS_INTRODUCTION, ...}
UPSTREAM_MODES = {OperationMode.FLOW_ADSORPTION_UPSTREAM, ...}
```

---

### 6. operation_results.pyのモード別細分化

#### 現状の問題

**対象ファイル:** `core/physics/operation_results.py`

```python
@dataclass
class StopModeResult: ...

@dataclass  
class BatchAdsorptionResult: ...

@dataclass
class FlowAdsorptionResult: ...
# 5種類の結果クラスが分散
```

**問題点:**
- 各モードで別のResultクラスを返すため、呼び出し側で型を意識する必要がある
- 共通フィールドが重複定義されている

#### 改善案

**移行先:** `state/calculation_results.py`（作成済み）

```python
@dataclass
class OperationResult:
    """
    運転モード計算結果（全モード共通）
    
    全モード共通: tower_results
    オプション: pressure_after, downstream_flow, vacuum_pumping など
    """
    tower_results: TowerResults
    pressure_after: Optional[float] = None
    downstream_flow: Optional[Dict] = None
    vacuum_pumping: Optional[Dict] = None
```

---

## 改善の優先順位まとめ

| 優先度 | 問題 | 理由 |
|--------|------|------|
| **高** | Strategyパターン廃止 | 「計算はどこ？」に答えやすくする |
| **高** | Results多層構造の簡素化 | データアクセスを直感的に |
| **中** | CellCalculator廃止 | 不要なクラスを削除 |
| **中** | PhysicsCalculator簡素化 | 抽象クラスをif分岐に |
| **中** | OperationMode Enum化 | タイポ防止・一元管理 |
| **中** | 結果クラス統合 | 戻り値の型を統一 |

---

## 移行ステータス

### Phase 1: 基盤整備 ✅ 完了

| 項目 | ステータス |
|------|-----------|
| `operation_modes/mode_types.py` (Enum定義) | ✅ 完了 |
| `state/calculation_results.py` (簡素化版Results) | ✅ 完了 |
| `common/` フォルダ作成と定数分離 | ✅ 完了 |
| 各フォルダの `__init__.py` | ✅ 完了 |

### Phase 2: 物理計算の移行 ⏳ 未着手

- [ ] `physics/mass_balance.py` 新規作成
- [ ] `physics/heat_balance.py` 新規作成
- [ ] `physics/pressure.py` 新規作成
- [ ] 旧 `adsorption_base_models.py` から移行

### Phase 3: 運転モードの移行 ⏳ 未着手

- [ ] `operation_modes/` 配下にモードファイル作成
- [ ] 旧 `operation_models.py` から移行
- [ ] Strategyパターン廃止

### Phase 4: プロセス制御の移行 ⏳ 未着手

- [ ] `process/simulator.py` 新規作成（軽量化版）
- [ ] `process/process_executor.py` 新規作成
- [ ] 旧 `simulator.py` から移行

### Phase 5: クリーンアップ ⏳ 未着手

- [ ] 旧ファイル構造を削除
- [ ] import文の整理
- [ ] テスト実行・動作確認
