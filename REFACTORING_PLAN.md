# PSAシミュレーター リファクタリング計画

## 目的

PSA担当部署の技術者がコードを読んで理解しやすい構造にリファクタリングする。

**設計方針:**
- 「〇〇の計算はどこ？」にすぐ答えられるファイル構成
- 余計なクラス階層をたどらないシンプルな設計
- 計算式が直接見える素直なコード

---

## 1. リファクタリング後のフォルダ構成

```
psa_simulator/
│
├── main.py                              # エントリーポイント
├── main_cond.yml                        # 実行条件設定
│
├── config/                              # 設定読込
│   ├── __init__.py
│   ├── simulation_conditions.py         # sim_conds.xlsx読込・データクラス定義
│   └── operation_schedule.py            # 稼働工程表読込（新規分離）
│
├── process/                             # プロセス制御
│   ├── __init__.py
│   ├── simulator.py                     # シミュレーター本体（責務を軽量化）
│   ├── process_executor.py              # 工程実行ロジック（新規分離）
│   └── termination_conditions.py        # 終了条件判定（新規分離）
│
├── operation_modes/                     # 運転モード別計算
│   ├── __init__.py
│   ├── mode_types.py                    # OperationMode Enum定義
│   ├── stop.py                          # 停止モード
│   ├── flow_adsorption.py               # 流通吸着（単独/上流/下流）
│   ├── batch_adsorption.py              # バッチ吸着（上流/下流/圧調弁）
│   ├── equalization.py                  # 均圧（加圧/減圧）
│   ├── vacuum_desorption.py             # 真空脱着
│   └── initial_gas_introduction.py      # 初回ガス導入
│
├── physics/                             # 物理計算
│   ├── __init__.py
│   ├── mass_balance.py                  # 物質収支計算
│   ├── heat_balance.py                  # 熱収支計算（層・壁・蓋）
│   ├── pressure.py                      # 圧力計算（真空排気・均圧・バッチ後）
│   ├── adsorption_isotherm.py           # 吸着等温線（平衡吸着量）
│   └── heat_transfer_coefficient.py     # 伝熱係数計算
│
├── state/                               # 塔の状態管理
│   ├── __init__.py
│   ├── tower_state.py                   # 状態変数（StateVariables, TowerStateArrays）
│   └── calculation_results.py           # 計算結果データクラス（簡素化版）
│
├── output/                              # 結果出力
│   ├── __init__.py
│   ├── csv_exporter.py                  # CSV出力
│   ├── png_exporter.py                  # グラフPNG出力
│   └── xlsx_exporter.py                 # Excel出力
│
├── common/                              # 共通ユーティリティ
│   ├── __init__.py
│   ├── constants.py                     # 物理定数
│   ├── paths.py                         # パス設定
│   ├── translations.py                  # 日本語表示用辞書
│   └── unit_conversion.py               # 単位変換
│
├── logger/                              # ログ
│   ├── __init__.py
│   └── log_config.py                    # ログ設定
│
├── conditions/                          # 条件ファイル（既存）
│   └── {cond_id}/
│       ├── sim_conds.xlsx
│       └── 稼働工程表.xlsx
│
├── data/                                # 観測データ（既存）
│   └── 3塔データ.csv
│
└── output_results/                      # 出力先（既存のoutput/から名称変更）
    └── {cond_id}/
        ├── csv/
        ├── png/
        └── xlsx/
```

### フォルダ構成の設計意図

| フォルダ | 役割 | PSA担当者への説明 |
|----------|------|-------------------|
| `config/` | 設定読込 | sim_conds.xlsxや稼働工程表の読み込み |
| `process/` | プロセス制御 | 工程の順番通りにシミュレーションを進める部分 |
| `operation_modes/` | 運転モード | 「流通吸着」「真空脱着」など各モードの計算 |
| `physics/` | 物理計算 | 物質収支・熱収支・圧力などの基礎計算 |
| `state/` | 状態管理 | 各塔の温度・圧力・吸着量などの現在値 |
| `output/` | 結果出力 | CSV・PNG・Excelへの出力処理 |
| `common/` | 共通部品 | 物理定数・単位変換など |

---

## 2. 優先度「高」の問題と改善

### 2.1 Strategyパターンの過剰適用

#### 現状の問題

**ファイル:** `core/physics/mass_balance_strategies.py`

```python
# 現状: 3種類の計算に抽象クラス + 3つの実装クラス
class MassBalanceStrategy(ABC):
    @abstractmethod
    def calculate(...) -> MassBalanceCalculationResult:
        pass
    @abstractmethod
    def supports_mole_fraction(self) -> bool:
        pass

class AdsorptionStrategy(MassBalanceStrategy):
    def __init__(self, tower_conds, state_manager, tower_num, 
                 external_inflow_gas=None, equalization_flow_rate=None,
                 residual_gas_composition=None):
        # ... 6つの引数を保持
    
    def calculate(self, stream, section, previous_result):
        # adsorption_base_models.calculate_mass_balance_for_adsorption()を呼ぶだけ
        ...

class DesorptionStrategy(MassBalanceStrategy): ...
class ValveClosedStrategy(MassBalanceStrategy): ...
```

**PSA担当者の視点での問題:**
- 「吸着の物質収支計算はどこ？」→ Strategy → base_models と2段階たどる必要
- 「Strategy」「Abstract」というソフトウェア用語が技術者には馴染みがない
- 3種類しかない分岐にデザインパターンを適用する必要があるか疑問

#### 改善後

**ファイル:** `physics/mass_balance.py`

```python
"""物質収支計算モジュール

PSA担当者向け説明:
- 吸着モード: CO2が吸着材に吸着される際の物質収支
- 脱着モード: 真空引きでCO2が脱着される際の物質収支
- 停止モード: バルブ閉鎖時（物質移動なし）
"""

from operation_modes.mode_types import OperationMode, ADSORPTION_MODES

def calculate_mass_balance(
    mode: OperationMode,
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    inflow_gas: Optional[GasFlow] = None,
    equalization_flow_rate: Optional[float] = None,
    residual_gas_composition: Optional[MassBalanceResults] = None,
    vacuum_pumping_results: Optional[VacuumPumpingResult] = None,
) -> MassBalanceCalculationResult:
    """
    物質収支を計算する
    
    Args:
        mode: 運転モード
        tower_conds: 塔条件
        stream: ストリーム番号 (1-indexed)
        section: セクション番号 (1-indexed)
        state_manager: 状態変数管理
        tower_num: 塔番号
        inflow_gas: 流入ガス情報（下流セクション・下流塔の場合）
        equalization_flow_rate: 均圧配管流量（均圧減圧モードの場合）
        residual_gas_composition: 残留ガス組成（バッチ吸着下流の場合）
        vacuum_pumping_results: 真空排気結果（脱着モードの場合）
    
    Returns:
        MassBalanceCalculationResult: 物質収支計算結果
    """
    # モードに応じて計算を分岐
    if mode in ADSORPTION_MODES:
        return _calculate_adsorption_mass_balance(
            tower_conds, stream, section, state_manager, tower_num,
            inflow_gas, equalization_flow_rate, residual_gas_composition
        )
    elif mode == OperationMode.VACUUM_DESORPTION:
        return _calculate_desorption_mass_balance(
            tower_conds, stream, section, state_manager, tower_num,
            vacuum_pumping_results
        )
    elif mode == OperationMode.STOP:
        return _calculate_valve_closed_mass_balance(
            stream, section, state_manager, tower_num
        )
    else:
        raise ValueError(f"未対応の運転モード: {mode}")


def _calculate_adsorption_mass_balance(...) -> MassBalanceCalculationResult:
    """吸着モードの物質収支計算"""
    # 現在の adsorption_base_models.calculate_mass_balance_for_adsorption() の内容
    ...


def _calculate_desorption_mass_balance(...) -> MassBalanceCalculationResult:
    """脱着モードの物質収支計算"""
    # 現在の adsorption_base_models.calculate_mass_balance_for_desorption() の内容
    ...


def _calculate_valve_closed_mass_balance(...) -> MassBalanceCalculationResult:
    """停止モードの物質収支計算（物質移動なし）"""
    # 現在の adsorption_base_models.calculate_mass_balance_for_valve_closed() の内容
    ...
```

**改善のポイント:**
- 抽象クラス・継承を廃止し、シンプルな関数に
- 1ファイルで物質収支計算が完結
- 「〇〇の計算はどこ？」→ `physics/mass_balance.py` と即答できる

---

### 2.2 Results系クラスの多層構造

#### 現状の問題

**ファイル:** `core/state/results.py`

```python
# 現状: 4層のクラス階層
class SectionResults:
    def __init__(self, section_data: Dict[int, Any]):
        self.section_data = section_data
    def get_cell_result(self, section_id: int) -> Any:
        return self.section_data.get(section_id)

class MaterialBalanceSectionResults(SectionResults):
    def get_material_balance_result(self, section_id: int) -> MaterialBalanceResult:
        return super().get_cell_result(section_id)

class MaterialBalanceStreamSectionResults(StreamSectionResults):
    def __init__(self, results_by_stream_section: Dict[int, Dict[int, MaterialBalanceResult]] = None):
        # ...
    def get_material_balance_result(self, stream_id: int, section_id: int) -> MaterialBalanceResult:
        return self.get_stream_data(stream_id).get_material_balance_result(section_id)

class MassBalanceResults:
    def __init__(self, material_balance_results_dict: Dict[int, Dict[int, MaterialBalanceResult]] = None):
        self.material_balance_stream_section_results = MaterialBalanceStreamSectionResults(...)
    
    def get_result(self, stream_id: int, section_id: int) -> MaterialBalanceResult:
        return self.material_balance_stream_section_results.get_material_balance_result(stream_id, section_id)

# 使用例: 4階層をたどる
result = mass_balance_results.get_result(stream, section)
```

**PSA担当者の視点での問題:**
- 「セクション3の吸着量は？」という単純な質問に4クラスの関係理解が必要
- `MaterialBalanceSectionResults`, `MaterialBalanceStreamSectionResults` など長い名前
- 同じパターンが `HeatBalance`, `MoleFraction` で3回繰り返し

#### 改善後

**ファイル:** `state/calculation_results.py`

```python
"""計算結果データクラス

PSA担当者向け説明:
- MaterialBalanceResult: 1セルの物質収支結果
- HeatBalanceResult: 1セルの熱収支結果
- TowerResults: 1塔の全セル結果をまとめたもの
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

# ============================================================
# 基本データクラス（1セル分の結果）
# ============================================================

@dataclass
class GasFlow:
    """ガス流量"""
    co2_volume: float      # CO2体積 [cm3]
    n2_volume: float       # N2体積 [cm3]
    co2_mole_fraction: float  # CO2モル分率 [-]
    n2_mole_fraction: float   # N2モル分率 [-]


@dataclass
class MaterialBalanceResult:
    """1セルの物質収支結果"""
    inlet_gas: GasFlow           # 流入ガス
    outlet_gas: GasFlow          # 流出ガス
    equilibrium_loading: float   # 平衡吸着量 [cm3/g-abs]
    actual_uptake_volume: float  # 実吸着量 [cm3]
    updated_loading: float       # 更新後吸着量 [cm3/g-abs]
    co2_partial_pressure: float  # CO2分圧 [MPa]
    outlet_co2_partial_pressure: float  # 流出CO2分圧 [MPa]
    gas_density: float           # ガス密度 [kg/m3]
    gas_specific_heat: float     # ガス比熱 [kJ/kg/K]


@dataclass
class HeatBalanceResult:
    """1セルの熱収支結果"""
    bed_temperature: float           # 層温度 [℃]
    thermocouple_temperature: float  # 熱電対温度 [℃]
    wall_to_bed_htc: float          # 壁-層伝熱係数 [W/m2/K]
    bed_to_bed_htc: float           # 層間伝熱係数 [W/m2/K]
    adsorption_heat: float          # 吸着熱 [J]
    heat_flux_downstream: float     # 下流への熱流束 [J]
    heat_flux_upstream: float       # 上流への熱流束 [J]


@dataclass
class WallHeatResult:
    """壁面の熱収支結果"""
    temperature: float           # 壁温度 [℃]
    heat_flux_downstream: float  # 下流への熱流束 [J]


@dataclass
class LidHeatResult:
    """蓋の熱収支結果"""
    temperature: float  # 蓋温度 [℃]


# ============================================================
# 塔全体の結果（簡素化版）
# ============================================================

@dataclass
class TowerResults:
    """
    1塔の計算結果
    
    使用例:
        # セクション温度を取得
        temp = results.get_temperature(stream=1, section=3)
        
        # 物質収支結果を取得
        mb = results.get_mass_balance(stream=1, section=3)
        print(f"吸着量: {mb.updated_loading}")
    """
    # (stream, section) をキーとする辞書
    _mass_balance: Dict[Tuple[int, int], MaterialBalanceResult] = field(default_factory=dict)
    _heat_balance: Dict[Tuple[int, int], HeatBalanceResult] = field(default_factory=dict)
    
    # section のみをキーとする辞書
    _wall_heat: Dict[int, WallHeatResult] = field(default_factory=dict)
    
    # "up" / "down" をキーとする辞書
    _lid_heat: Dict[str, LidHeatResult] = field(default_factory=dict)
    
    # --- 物質収支 ---
    def get_mass_balance(self, stream: int, section: int) -> MaterialBalanceResult:
        """物質収支結果を取得"""
        return self._mass_balance[(stream, section)]
    
    def set_mass_balance(self, stream: int, section: int, result: MaterialBalanceResult):
        """物質収支結果を設定"""
        self._mass_balance[(stream, section)] = result
    
    # --- 熱収支 ---
    def get_heat_balance(self, stream: int, section: int) -> HeatBalanceResult:
        """熱収支結果を取得"""
        return self._heat_balance[(stream, section)]
    
    def set_heat_balance(self, stream: int, section: int, result: HeatBalanceResult):
        """熱収支結果を設定"""
        self._heat_balance[(stream, section)] = result
    
    # --- 便利メソッド（よく使う値への直接アクセス） ---
    def get_temperature(self, stream: int, section: int) -> float:
        """セクション温度を取得 [℃]"""
        return self._heat_balance[(stream, section)].bed_temperature
    
    def get_loading(self, stream: int, section: int) -> float:
        """吸着量を取得 [cm3/g-abs]"""
        return self._mass_balance[(stream, section)].updated_loading
    
    def get_outlet_co2_flow(self, stream: int, section: int) -> float:
        """流出CO2流量を取得 [cm3]"""
        return self._mass_balance[(stream, section)].outlet_gas.co2_volume
    
    # --- 壁・蓋 ---
    def get_wall_heat(self, section: int) -> WallHeatResult:
        """壁面熱収支を取得"""
        return self._wall_heat[section]
    
    def set_wall_heat(self, section: int, result: WallHeatResult):
        """壁面熱収支を設定"""
        self._wall_heat[section] = result
    
    def get_lid_heat(self, position: str) -> LidHeatResult:
        """蓋熱収支を取得 (position: "up" or "down")"""
        return self._lid_heat[position]
    
    def set_lid_heat(self, position: str, result: LidHeatResult):
        """蓋熱収支を設定"""
        self._lid_heat[position] = result


# ============================================================
# 運転モード計算の統一結果
# ============================================================

@dataclass
class OperationResult:
    """
    運転モード計算結果（全モード共通）
    
    PSA担当者向け説明:
    - 全モードで共通: material, heat, heat_wall, heat_lid
    - 圧力変化があるモード: pressure_after に値が入る
    - 真空脱着モード: vacuum_pumping, mole_fraction に値が入る
    - 均圧減圧モード: downstream_flow に値が入る
    """
    # 全モード共通
    tower_results: TowerResults
    
    # 圧力関連（バッチ吸着、流通吸着、真空脱着、均圧で使用）
    pressure_after: Optional[float] = None
    
    # 均圧減圧モード用
    pressure_diff: Optional[float] = None
    downstream_flow: Optional[Dict] = None
    
    # 真空脱着モード用
    vacuum_pumping: Optional[Dict] = None
    mole_fraction: Optional[Dict[Tuple[int, int], Dict]] = None
```

**改善のポイント:**
- 4層の継承階層 → 1クラス `TowerResults` に統合
- `get_temperature(stream, section)` のような直感的なメソッド
- モード別の5種類のResultクラス → 1つの `OperationResult` に統合

---

## 3. 優先度「中」の問題と改善

### 3.1 CellCalculatorクラスの不要なクラス化

#### 現状の問題

**ファイル:** `core/physics/operation_models.py`

```python
class CellCalculator:
    """セル計算の共通処理を提供するクラス"""

    @staticmethod
    def calculate_mass_and_heat_balance(...): ...
    
    @staticmethod
    def calculate_wall_heat_balance(...): ...
    
    @staticmethod
    def calculate_lid_heat_balance(...): ...
    
    @staticmethod
    def distribute_inflow_gas(...): ...

# 使用側: 毎回インスタンス化しているが意味がない
def flow_adsorption_single_or_upstream(...):
    calculator = CellCalculator()
    balance_results = calculator.calculate_mass_and_heat_balance(...)
```

**問題点:**
- 全メソッドが `@staticmethod` → クラスにする意味がない
- 「セル計算」という名前だが、壁・蓋の計算も含む
- インスタンス化は無意味（状態を持たない）

#### 改善後

クラスを廃止し、各モジュールに関数として配置:

```python
# physics/heat_balance.py
def calculate_bed_heat_balance(...) -> HeatBalanceResult: ...
def calculate_wall_heat_balance(...) -> WallHeatResult: ...
def calculate_lid_heat_balance(...) -> LidHeatResult: ...

# physics/mass_balance.py
def calculate_mass_balance(...) -> MassBalanceCalculationResult: ...

# operation_modes/common.py
def distribute_inflow_gas(...) -> Dict[int, GasFlow]: ...
```

---

### 3.2 PhysicsCalculatorの抽象化

#### 現状の問題

**ファイル:** `core/physics/adsorption_base_models.py` (末尾)

```python
class OperationModePhysicsCalculator(ABC):
    @abstractmethod
    def calculate_adsorption_heat(...): ...
    @abstractmethod
    def calculate_inlet_gas_mass(...): ...
    @abstractmethod
    def get_gas_specific_heat(...): ...

class AdsorptionPhysicsCalculator(OperationModePhysicsCalculator):
    def calculate_adsorption_heat(self, material_output, tower_conds):
        return (material_output.adsorption_state.actual_uptake_volume
                / 1000 / STANDARD_MOLAR_VOLUME
                * tower_conds.feed_gas.co2_molecular_weight
                * tower_conds.feed_gas.co2_adsorption_heat)
    # ...

class ValveClosedPhysicsCalculator(OperationModePhysicsCalculator):
    def calculate_adsorption_heat(...): return 0
    def calculate_inlet_gas_mass(...): return 0
    def get_gas_specific_heat(...): return 0

class DesorptionPhysicsCalculator(OperationModePhysicsCalculator):
    # AdsorptionPhysicsCalculatorとほぼ同じ
    ...

def _get_physics_calculator(mode: int) -> OperationModePhysicsCalculator:
    calculators = {0: AdsorptionPhysicsCalculator(), 1: ValveClosedPhysicsCalculator(), 2: DesorptionPhysicsCalculator()}
    return calculators.get(mode, AdsorptionPhysicsCalculator())
```

**問題点:**
- 3メソッドだけの抽象クラスに3つの実装クラス → 過剰
- 1500行ファイルの末尾に隠れており発見しづらい
- 停止モードは全て0を返すだけ、吸着と脱着はほぼ同じ

#### 改善後

**ファイル:** `physics/heat_balance.py` 内にヘルパー関数として配置

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
    return (
        material_output.actual_uptake_volume
        / 1000 
        / STANDARD_MOLAR_VOLUME
        * tower_conds.feed_gas.co2_molecular_weight
        * tower_conds.feed_gas.co2_adsorption_heat
    )


def _get_inlet_gas_mass(
    mode: OperationMode,
    material_output: MaterialBalanceResult,
    tower_conds: TowerConditions
) -> float:
    """流入ガス質量を計算 [g]"""
    if mode in (OperationMode.STOP, OperationMode.VACUUM_DESORPTION):
        return 0.0
    
    return (
        material_output.inlet_gas.co2_volume / 1000 / STANDARD_MOLAR_VOLUME
        * tower_conds.feed_gas.co2_molecular_weight
        + material_output.inlet_gas.n2_volume / 1000 / STANDARD_MOLAR_VOLUME
        * tower_conds.feed_gas.n2_molecular_weight
    )


def _get_gas_specific_heat(
    mode: OperationMode,
    material_output: MaterialBalanceResult
) -> float:
    """ガス比熱を取得 [kJ/kg/K]"""
    if mode == OperationMode.STOP:
        return 0.0
    return material_output.gas_specific_heat
```

---

### 3.3 運転モードのEnum化

#### 現状の問題

**複数ファイル:** `simulator.py`, `state_variables.py`

```python
# simulator.py
if mode == "初回ガス導入":
    ...
elif mode == "停止":
    ...
elif mode == "流通吸着_単独/上流":
    ...
# 12種類のif-elif

# state_variables.py (同じ分岐が重複)
if mode == "停止":
    pass
elif mode in ["初回ガス導入", "流通吸着_単独/上流", ...]:
    ...
```

**問題点:**
- 文字列比較のためtypoしても検出不可
- 同じ分岐が複数箇所に重複
- 新モード追加時に複数箇所を修正必要

#### 改善後

**ファイル:** `operation_modes/mode_types.py`

```python
"""運転モード定義

PSA担当者向け説明:
各運転モードを定義しています。稼働工程表の「塔1」「塔2」「塔3」列に
記載されるモード名と対応しています。
"""

from enum import Enum

class OperationMode(Enum):
    """運転モード"""
    # 基本モード
    INITIAL_GAS_INTRODUCTION = "初回ガス導入"
    STOP = "停止"
    
    # 流通吸着
    FLOW_ADSORPTION_UPSTREAM = "流通吸着_単独/上流"
    FLOW_ADSORPTION_DOWNSTREAM = "流通吸着_下流"
    
    # バッチ吸着
    BATCH_ADSORPTION_UPSTREAM = "バッチ吸着_上流"
    BATCH_ADSORPTION_DOWNSTREAM = "バッチ吸着_下流"
    BATCH_ADSORPTION_UPSTREAM_WITH_VALVE = "バッチ吸着_上流（圧調弁あり）"
    BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE = "バッチ吸着_下流（圧調弁あり）"
    
    # 均圧
    EQUALIZATION_DEPRESSURIZATION = "均圧_減圧"
    EQUALIZATION_PRESSURIZATION = "均圧_加圧"
    
    # 真空脱着
    VACUUM_DESORPTION = "真空脱着"
    
    @classmethod
    def from_japanese(cls, name: str) -> "OperationMode":
        """日本語名からEnumに変換"""
        for mode in cls:
            if mode.value == name:
                return mode
        raise ValueError(f"未対応の運転モード: {name}")


# モードのグループ分け（計算ロジックで使用）
ADSORPTION_MODES = {
    OperationMode.INITIAL_GAS_INTRODUCTION,
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
    OperationMode.EQUALIZATION_DEPRESSURIZATION,
    OperationMode.EQUALIZATION_PRESSURIZATION,
}

UPSTREAM_MODES = {
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
}

DOWNSTREAM_MODES = {
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
}

# 圧力更新が必要なモード
PRESSURE_UPDATE_MODES = {
    OperationMode.INITIAL_GAS_INTRODUCTION,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
    OperationMode.EQUALIZATION_PRESSURIZATION,
    OperationMode.EQUALIZATION_DEPRESSURIZATION,
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.VACUUM_DESORPTION,
}
```

---

## 4. 移行手順

### Phase 1: 基盤整備（破壊的変更なし）
1. `operation_modes/mode_types.py` を新規作成（Enum定義）
2. `state/calculation_results.py` を新規作成（簡素化版Results）
3. `common/` フォルダを作成し、定数を分離

### Phase 2: 物理計算の移行
1. `physics/mass_balance.py` を新規作成
2. `physics/heat_balance.py` を新規作成
3. `physics/pressure.py` を新規作成
4. 旧 `adsorption_base_models.py` から段階的に移行

### Phase 3: 運転モードの移行
1. `operation_modes/` 配下に各モードファイルを作成
2. 旧 `operation_models.py` から段階的に移行
3. Strategyパターンを廃止

### Phase 4: プロセス制御の移行
1. `process/simulator.py` を新規作成（軽量化版）
2. `process/process_executor.py` を新規作成
3. `process/termination_conditions.py` を新規作成
4. 旧 `simulator.py` から段階的に移行

### Phase 5: クリーンアップ
1. 旧ファイル構造を削除
2. import文の整理
3. テスト実行・動作確認

---

## 5. 注意事項

### 互換性
- 外部インターフェース（`main.py` の呼び出し方、設定ファイル形式）は変更しない
- 出力ファイル形式（CSV, PNG, XLSX）は変更しない

### テスト
- 各Phase完了時に既存のシミュレーション結果と比較
- 数値の一致を確認

### ドキュメント
- 各モジュールの冒頭に「PSA担当者向け説明」を記載
- 複雑な計算には物理的な意味のコメントを追加
