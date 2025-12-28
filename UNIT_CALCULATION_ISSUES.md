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

## 3. その他の確認結果

### `physics/adsorption_isotherm.py`
- **問題なし**: 経験式のため係数で単位調整済み
- 入力: `pressure_kpa` [kPaA], `temperature_k` [K]
- 出力: `equilibrium_loading` [cm³/g-abs]

### `physics/heat_balance.py`
- **確認済み**: 主要な熱流束計算の単位は整合
- 熱流束: [J] = [W/m²/K] * [m²] * [K] * [min] * [s/min]
- 伝熱係数: [W/m²/K]

### `physics/heat_transfer.py`
- **確認済み**: Yagi-Kuniiモデルに基づく計算
- 出力: `wall_to_bed_heat_transfer_coef` [W/m²/K], `bed_heat_transfer_coef` [W/m²/K]

### `physics/pressure.py`
- **確認済み**: 理想気体の状態方程式に基づく計算
- 圧力変化: [MPaA] = [J/mol/K] * [K] / [m³] * [mol] * [Pa/MPa]

---

## 対応優先度

| 問題 | 優先度 | 理由 |
|------|--------|------|
| 1. 脱着モードの単位不整合 | **高** | 明確なバグ。脱着量が大きい条件で結果が破綻する可能性 |
| 2. LDFモデルの単位不明確 | 中 | 現状動作しているが、ドキュメント整備が必要 |

---

## 更新履歴

- 2025-12-28: 初版作成
