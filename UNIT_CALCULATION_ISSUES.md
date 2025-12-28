# 単位計算に関する問題点

このドキュメントは、physicsモジュールの単位計算を確認した際に発見された問題点をまとめたものです。

## 1. `physics/mass_balance.py` - 脱着モードの単位不整合

### 問題箇所
`_calculate_desorption_mass_balance` 関数（363-366行目付近）

### 問題の内容
異なる単位の値を加算している。

```python
# 319-320行目: inlet_co2_volume を [cm³] に変換
inlet_co2_volume *= STANDARD_MOLAR_VOLUME * L_TO_CM3  # [mol] → [cm³]

# 361行目: desorp_mw_co2 は [mol] 単位
desorp_mw_co2 = -actual_uptake_volume / (L_TO_CM3 * STANDARD_MOLAR_VOLUME)  # [cm³] → [mol]

# 364行目: 単位不整合の加算
desorp_mw_co2_after_vacuum = inlet_co2_volume + desorp_mw_co2  # [cm³] + [mol] ← バグ
```

### 単位の追跡
1. `inlet_co2_volume` は314-316行目で `[mol]` として計算される
2. 319-320行目で `STANDARD_MOLAR_VOLUME * L_TO_CM3` を掛けて `[cm³]` に変換される
3. `desorp_mw_co2` は361行目で `[mol]` 単位として計算される
4. 364行目で `[cm³] + [mol]` を加算 → **単位不整合**

### 現状の挙動
- `inlet_co2_volume` は `[cm³]` 単位で数値が大きい（例: 6.72）
- `desorp_mw_co2` は `[mol]` 単位で数値が非常に小さい（例: 2.23e-5）
- 結果として加算結果はほぼ `inlet_co2_volume` の値そのままになる
- **偶然動いているだけ**で、脱着量が大きくなった場合に結果が破綻する可能性がある

### 修正案
以下のいずれかの方法で修正が必要：

**案1**: `inlet_co2_volume` を `[mol]` に戻してから加算
```python
# 気相放出後モル量 [mol]
inlet_co2_volume_mol = inlet_co2_volume / (STANDARD_MOLAR_VOLUME * L_TO_CM3)
desorp_mw_co2_after_vacuum = inlet_co2_volume_mol + desorp_mw_co2
```

**案2**: `desorp_mw_co2` を `[cm³]` に変換してから加算
```python
# 気相放出CO2量 [cm³]
desorp_co2_volume = -actual_uptake_volume  # すでに [cm³]
desorp_mw_co2_after_vacuum = inlet_co2_volume + desorp_co2_volume  # [cm³] + [cm³]
```

### 影響範囲
- 真空脱着モード（`OperationMode.VACUUM_DESORPTION`）の物質収支計算
- 工程表で「真空脱着」と指定された工程

---

## 2. `physics/mass_balance.py` - `_calculate_theoretical_uptake` の単位

### 問題箇所
`_calculate_theoretical_uptake` 関数（549-582行目）

### 問題の内容
LDFモデルによる吸着量計算式の単位が不明確。

```python
theoretical_loading_delta = (
    tower_conds.packed_bed.adsorption_mass_transfer_coef ** (current_loading / equilibrium_loading)
    / tower_conds.packed_bed.adsorbent_bulk_density
    * 6
    * (1 - tower_conds.packed_bed.average_porosity)
    * tower_conds.packed_bed.particle_shape_factor
    / tower_conds.packed_bed.average_particle_diameter
    * (equilibrium_loading - current_loading)
    * tower_conds.common.calculation_step_time
    * MINUTE_TO_SECOND
    / 1e6
)
```

### 単位の追跡
| パラメータ | 単位（定義より） |
|-----------|-----------------|
| `adsorption_mass_transfer_coef` | 1e-6/sec |
| `adsorbent_bulk_density` | g/cm³ |
| `average_porosity` | - (無次元) |
| `particle_shape_factor` | - (無次元) |
| `average_particle_diameter` | m |
| `equilibrium_loading`, `current_loading` | cm³/g-abs |
| `calculation_step_time` | min |
| `MINUTE_TO_SECOND` | 60 s/min |

### 疑問点
1. `coef ** (ratio)` の指数演算は物理的に非標準
   - 通常のLDFモデルは `k * (q_eq - q)` の形式
   - 指数演算は経験的な補正？
2. 単位を追跡すると整合しない
   - `[1/s] / [g/cm³] * [-] * [-] / [m] * [cm³/g-abs] * [s] / 1e6`
   - 期待される出力単位 `[cm³/g-abs]` と一致しない
3. `/1e6` の意味が不明確
   - `adsorption_mass_transfer_coef` の単位が `1e-6/sec` なので、実質的に `1/sec` に戻す操作？

### 現状
- シミュレーション結果は実験データと整合している（テストが通っている）
- パラメータが経験的に調整されている可能性が高い
- ドキュメント上の単位と実際の計算が一致しているか要確認

---

## 3. `physics/heat_balance.py` - 壁面熱伝導の距離項欠落

### 問題箇所
`calculate_wall_heat_balance` 関数（300-306行目、332-344行目）

### 問題の内容
フーリエの法則（熱伝導）において、伝熱距離 L での除算が抜けている。

**フーリエの法則**:
$$Q = k \cdot A \cdot \frac{\Delta T}{L} \cdot \Delta t$$

- $Q$: 熱量 [J]
- $k$: 熱伝導率 [W/(m·K)]
- $A$: 断面積 [m²]
- $\Delta T$: 温度差 [K]
- $L$: 伝熱距離 [m]
- $\Delta t$: 時間 [s]

**現在のコード（300-306行目）**:
```python
upstream_heat_flux = (
    tower_conds.vessel.wall_thermal_conductivity  # k [W/(m·K)]
    * stream_conds[wall_stream_1indexed].cross_section  # A [m²]
    * (temp_now - tower.lid_temperature)  # ΔT [K]
    * tower_conds.common.calculation_step_time  # dt [min]
    * MINUTE_TO_SECOND  # [s/min]
    # ← L での除算が抜けている
)
```

### 単位計算
現在のコード:
$$[W/(m \cdot K)] \cdot [m^2] \cdot [K] \cdot [s] = [J \cdot m]$$

期待される単位: $[J]$

**結果**: $[J \cdot m]$ となり、単位が整合しない。

### 追加の問題（340-344行目）
中間セクション間の下流熱流束計算では、Lでの除算に加えて時間項も抜けている:
```python
downstream_heat_flux = (
    tower_conds.vessel.wall_thermal_conductivity
    * stream_conds[wall_stream_1indexed].cross_section
    * (temp_now - tower.temp_wall[section + 1])
    # ← L での除算なし
    # ← dt * MINUTE_TO_SECOND もなし
)
```

### 修正案
伝熱距離 L（セクション長さ）を追加:
```python
section_length = tower_conds.packed_bed.height / tower_conds.common.num_sections  # [m]

upstream_heat_flux = (
    tower_conds.vessel.wall_thermal_conductivity
    * stream_conds[wall_stream_1indexed].cross_section
    * (temp_now - tower.lid_temperature)
    / section_length  # ← 追加
    * tower_conds.common.calculation_step_time
    * MINUTE_TO_SECOND
)
```

### 影響範囲
- 壁面の温度変化計算
- 全運転モードに影響

---

## 4. `physics/heat_transfer.py` - 熱伝導率とPrandtl数のスケーリング誤り

### 問題箇所
`compute_gas_k` 関数（19-27行目）および `calc_heat_transfer_coef` 関数（180行目）

### 問題の内容

#### 4-1. `compute_gas_k` の `/1000` 問題
CoolPropから取得した熱伝導率を誤って1000で除算している。

**現在のコード（27行目）**:
```python
def compute_gas_k(T_K: float, co2_mole_fraction: float, n2_mole_fraction: float) -> float:
    P_ATM = STANDARD_PRESSURE
    k_co2 = CP.PropsSI("L", "T", T_K, "P", P_ATM, "co2")  # [W/(m·K)]
    k_n2 = CP.PropsSI("L", "T", T_K, "P", P_ATM, "nitrogen")  # [W/(m·K)]
    return (k_co2 * co2_mole_fraction + k_n2 * n2_mole_fraction) / 1000  # ← 誤り
```

**検証結果**（25℃、30% CO2 / 70% N2）:
- `k_co2` (CoolProp) = 0.016633 W/(m·K)
- `k_n2` (CoolProp) = 0.025835 W/(m·K)
- `k_mix` (正しい値) = 0.023074 W/(m·K)
- `k_mix` (現在のコード) = 0.000023 W/(m·K) ← **1000倍の差**

#### 4-2. Prandtl数 (Pr) の計算誤り

**現在のコード（180行目）**:
```python
Pr = viscosity * 1000.0 * material_output.gas_properties.specific_heat / kf
```

**Prandtl数の定義**（無次元）:
$$Pr = \frac{\mu \cdot C_p}{k}$$

- $\mu$: 粘度 [Pa·s]
- $C_p$: 比熱 [J/(kg·K)]
- $k$: 熱伝導率 [W/(m·K)]

**問題点**:
- `specific_heat` が [kJ/(kg·K)] 単位であるため `* 1000.0` で [J/(kg·K)] に変換
- しかし `kf` が既に `/1000` されているため、変換が相殺されず **1000倍の誤差**

**検証結果**:
- `Pr` (正しい計算) = 0.72
- `Pr` (現在のコード) = 722 ← **1000倍の差**
- 典型的なガスのPr値: 0.7 ~ 0.8

### 単位計算の詳細

**正しいPr計算** (全SI単位):
```
Pr = viscosity [Pa·s] * cp [J/(kg·K)] / k [W/(m·K)]
   = [kg/(m·s)] * [J/(kg·K)] / [J/(s·m·K)]
   = [-] (無次元)
```

**現在のコードのPr計算**:
```
Pr = viscosity [Pa·s] * 1000 * specific_heat [kJ/(kg·K)] / kf [mW/(m·K)]
   = [Pa·s] * 1000 * [kJ/(kg·K)] / [W/(m·K) / 1000]
   = 正しいPr の約1000倍
```

### 影響範囲

この誤りは以下の計算に波及:

| 関数/変数 | 影響 |
|-----------|------|
| `kf` (`compute_gas_k`) | 1/1000 に |
| `ke0` (`_yagi_kunii_radiation`) | 1/1000 に |
| `Pr` (`calc_heat_transfer_coef`) | 1000倍に |
| `ke`, `hw1_raw` (`_axial_flow_correction`) | 誤った値に |
| `wall_to_bed_heat_transfer_coef` | 誤った値に |
| `bed_heat_transfer_coef` | 誤った値に |

### 修正案

**compute_gas_k の修正**:
```python
def compute_gas_k(T_K: float, co2_mole_fraction: float, n2_mole_fraction: float) -> float:
    P_ATM = STANDARD_PRESSURE
    k_co2 = CP.PropsSI("L", "T", T_K, "P", P_ATM, "co2")
    k_n2 = CP.PropsSI("L", "T", T_K, "P", P_ATM, "nitrogen")
    return k_co2 * co2_mole_fraction + k_n2 * n2_mole_fraction  # /1000 を削除
```

**Pr 計算の修正**（specific_heatの単位確認が必要）:
```python
# specific_heat が [kJ/(kg·K)] の場合
Pr = viscosity * 1000.0 * material_output.gas_properties.specific_heat / kf
# specific_heat が [J/(kg·K)] の場合
Pr = viscosity * material_output.gas_properties.specific_heat / kf
```

**注意**: 修正後はシミュレーション結果が変わる可能性があるため、全工程の検証が必要。

---

## 5. その他の確認結果

### `physics/adsorption_isotherm.py`
- **問題なし**: 経験式のため係数で単位調整済み
- 入力: `pressure_kpa` [kPaA], `temperature_k` [K]
- 出力: `equilibrium_loading` [cm³/g-abs]

### `physics/heat_balance.py` - 充填層熱収支
- **確認済み**: 主要な熱流束計算（対流伝熱）の単位は整合
- 熱流束: [J] = [W/m²/K] * [m²] * [K] * [min] * [s/min]
- 伝熱係数: [W/m²/K]
- ※壁面熱伝導は上記の問題あり

### `physics/pressure.py`
- **確認済み**: 理想気体の状態方程式に基づく計算
- 圧力変化: [MPaA] = [J/mol/K] * [K] / [m³] * [mol] * [Pa/MPa]

---

## 対応優先度

| 問題 | 優先度 | 理由 |
|------|--------|------|
| 1. 脱着モードの単位不整合 | **高** | 明確なバグ。脱着量が大きい条件で結果が破綻する可能性 |
| 2. LDFモデルの単位不明確 | 中 | 現状動作しているが、ドキュメント整備が必要 |
| 3. 壁面熱伝導の距離項欠落 | **高** | フーリエの法則に反する。壁面温度計算に影響 |
| 4. 熱伝導率/Prのスケーリング誤り | **高** | Prが1000倍誤り、熱伝達係数に影響 |

---

## 更新履歴

- 2025-12-28: 初版作成
- 2025-12-28: 熱伝導率とPrandtl数のスケーリング問題（4項）を追記
