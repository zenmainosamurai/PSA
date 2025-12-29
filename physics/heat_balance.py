"""熱収支計算モジュール

このモジュールはPSAプロセスにおける熱収支（ヒートバランス）を計算します。

- 充填層の熱収支: 吸着熱、ガス流入熱、隣接セルとの熱交換
- 壁面の熱収支: 充填層からの熱流入、外気への放熱
- 蓋（上・下）の熱収支: 充填層・壁からの熱流入、外気への放熱

主要な関数:
- calculate_bed_heat_balance(): 充填層の熱収支計算
- calculate_wall_heat_balance(): 壁面の熱収支計算
- calculate_lid_heat_balance(): 蓋の熱収支計算
"""

from typing import Optional, Dict
import numpy as np
from scipy import optimize

from operation_modes.mode_types import HeatCalculationMode
from common.enums import LidPosition
from common.constants import (
    STANDARD_MOLAR_VOLUME,
    MINUTE_TO_SECOND,
)

from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MaterialBalanceResult,
    HeatBalanceResult,
    HeatBalanceResults,
    VacuumPumpingResult,
    HeatTransferCoefficients,
    HeatTransfer,
    CellTemperatures,
    WallHeatTransfer,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
)
from physics.heat_transfer import calc_heat_transfer_coef as _heat_transfer_coef


# ============================================================
# 充填層の熱収支計算
# ============================================================

def calculate_bed_heat_balance(
    tower_conds: TowerConditions,
    stream: int,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    mode: HeatCalculationMode,
    material_output: Optional[MaterialBalanceResult] = None,
    heat_output: Optional[HeatBalanceResult] = None,
    vacuum_pumping_results: Optional[VacuumPumpingResult] = None,
) -> HeatBalanceResult:
    """
    充填層セルの熱収支計算
    
    各セル（ストリーム×セクション）の温度変化を計算します。
    
    熱の出入り:
    - 吸着熱: CO2吸着時に発生する熱（発熱）
    - ガス流入熱: 流入ガスが持ち込む熱
    - 隣接セルとの熱交換: 内側・外側・上流・下流セルとの伝熱
    
    Args:
        tower_conds: 塔条件
        stream: ストリーム番号 (0-indexed, 内部インデックス)
        section: セクション番号 (0-indexed, 内部インデックス)
        state_manager: 状態変数管理
        tower_num: 塔番号 (1-indexed, I/O用)
        mode: 熱収支計算モード (ADSORPTION:吸着, VALVE_CLOSED:停止, DESORPTION:脱着)
        material_output: 物質収支計算結果
        heat_output: 上流セクションの熱収支結果 (section >= 1 の場合)
        vacuum_pumping_results: 真空排気結果（脱着時）
    
    Returns:
        HeatBalanceResult: 熱収支計算結果
    """
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions

    # === 現在温度の取得 ===
    temp_now = tower.cell(stream, section).temp
    
    # 内側セル温度 [℃] （stream=0は最内側）
    if stream == 0:
        temp_inside_cell = 18  # 中心部の想定温度
    else:
        temp_inside_cell = tower.cell(stream - 1, section).temp
    
    # 外側セル温度 [℃] （stream=num_streams-1は最外側）
    if stream != tower_conds.common.num_streams - 1:
        temp_outside_cell = tower.cell(stream + 1, section).temp
    else:
        temp_outside_cell = tower.temp_wall[section]
    
    # 下流セル温度 [℃] （section=num_sections-1は最下流）
    if section != tower_conds.common.num_sections - 1:
        temp_below_cell = tower.cell(stream, section + 1).temp

    # === 物理量の計算（モード依存）===
    adsorption_heat = _get_adsorption_heat(mode, material_output, tower_conds)
    inlet_gas_mass = _get_inlet_gas_mass(mode, material_output, tower_conds)
    gas_specific_heat = _get_gas_specific_heat(mode, material_output)

    # === 境界面積 ===
    section_inner_boundary_area = stream_conds[stream].inner_boundary_area / tower_conds.common.num_sections
    section_outer_boundary_area = stream_conds[stream].outer_boundary_area / tower_conds.common.num_sections
    cross_section_area = stream_conds[stream].cross_section

    # === 伝熱係数の計算 ===
    if mode == HeatCalculationMode.ADSORPTION:
        wall_to_bed_htc, bed_htc = _heat_transfer_coef(
            tower_conds, stream, section, temp_now, mode,
            state_manager, tower_num, material_output,
        )
    elif mode == HeatCalculationMode.VALVE_CLOSED:
        # 停止モードでは直前値を使用
        wall_to_bed_htc = tower.cell(stream, section).wall_to_bed_heat_transfer_coef
        bed_htc = tower.cell(stream, section).bed_heat_transfer_coef
    elif mode == HeatCalculationMode.DESORPTION:
        wall_to_bed_htc, bed_htc = _heat_transfer_coef(
            tower_conds, stream, section, temp_now, mode,
            state_manager, tower_num, material_output, vacuum_pumping_results,
        )
    else:
        raise ValueError(f"未対応のモード: {mode}")

    # === 熱量の計算 [J] ===
    
    # 内側境界からの熱量（stream=0は最内側）
    if stream == 0:
        from_inner_j = 0
    else:
        from_inner_j = (
            bed_htc
            * section_inner_boundary_area
            * (temp_inside_cell - temp_now)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    
    # 外側境界への熱量（stream=num_streams-1は最外側）
    if stream == tower_conds.common.num_streams - 1:
        to_outer_j = (
            wall_to_bed_htc
            * section_outer_boundary_area
            * (temp_now - temp_outside_cell)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    else:
        to_outer_j = (
            bed_htc
            * section_outer_boundary_area
            * (temp_now - temp_outside_cell)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    
    # 下流セルへの熱量（section=num_sections-1は最下流）
    if section == tower_conds.common.num_sections - 1:
        # 下蓋への熱量
        downstream_j = (
            wall_to_bed_htc
            * cross_section_area
            * (temp_now - tower.bottom_temperature)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    else:
        downstream_j = (
            bed_htc
            * cross_section_area
            * (temp_now - temp_below_cell)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    
    # 上流セルへの熱量（section=0は最上流）
    if section == 0:
        # 上蓋への熱量
        upstream_j = (
            wall_to_bed_htc
            * cross_section_area
            * (temp_now - tower.top_temperature)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    else:
        # 上流セルの下流熱量の負値
        upstream_j = -heat_output.heat_transfer.downstream_j

    # === 到達温度の計算（Newton法）===
    args = (
        tower_conds,
        gas_specific_heat,
        inlet_gas_mass,
        temp_now,
        adsorption_heat,
        from_inner_j,
        to_outer_j,
        downstream_j,
        upstream_j,
        stream,
    )
    temp_reached = optimize.newton(_optimize_bed_temperature, temp_now, args=args)

    # === 熱電対温度の計算 ===
    thermocouple_temp = _calculate_thermocouple_temperature(
        tower_conds, tower, stream, section, mode, wall_to_bed_htc
    )

    # === 結果オブジェクトの構築 ===
    return HeatBalanceResult(
        cell_temperatures=CellTemperatures(
            bed_temperature=temp_reached,
            thermocouple_temperature=thermocouple_temp,
        ),
        heat_transfer_coefficients=HeatTransferCoefficients(
            wall_to_bed=wall_to_bed_htc,
            bed_to_bed=bed_htc,
        ),
        heat_transfer=HeatTransfer(
            adsorption_j=adsorption_heat,
            from_inner_j=from_inner_j,
            to_outer_j=to_outer_j,
            downstream_j=downstream_j,
            upstream_j=upstream_j,
        ),
    )


# ============================================================
# 壁面の熱収支計算
# ============================================================

def calculate_wall_heat_balance(
    tower_conds: TowerConditions,
    section: int,
    state_manager: StateVariables,
    tower_num: int,
    heat_output: HeatBalanceResult,
    heat_wall_output: Optional[WallHeatBalanceResult] = None,
) -> WallHeatBalanceResult:
    """
    壁面の熱収支計算
    
    容器壁の温度変化を計算します。
    充填層からの熱流入と外気への放熱のバランスで決まります。
    
    Args:
        tower_conds: 塔条件
        section: セクション番号 (0-indexed, 内部インデックス)
        state_manager: 状態変数管理
        tower_num: 塔番号 (1-indexed, I/O用)
        heat_output: 隣接セル（最外ストリーム）の熱収支結果
        heat_wall_output: 上流セクションの壁面熱収支結果 (section >= 1 の場合)
    
    Returns:
        WallHeatBalanceResult: 壁面熱収支結果
    """
    tower = state_manager.towers[tower_num]
    stream_conds = tower_conds.stream_conditions
    num_streams = tower_conds.common.num_streams
    num_sections = tower_conds.common.num_sections
    wall_stream_idx = num_streams
    
    temp_now = tower.temp_wall[section]
    
    # 内側セル温度（最外ストリーム）
    temp_inside_cell = tower.cell(num_streams - 1, section).temp
    
    # 外側温度（外気）
    temp_outside = tower_conds.vessel.ambient_temperature
    
    # 下流壁温度（section=num_sections-1は最下流）
    if section != num_sections - 1:
        temp_below = tower.temp_wall[section + 1]

    # === 熱量の計算 [J] ===
    
    # 上流壁への熱量（section=0は最上流）
    if section == 0:
        upstream_j = (
            tower_conds.vessel.wall_thermal_conductivity
            * stream_conds[wall_stream_idx].cross_section
            * (temp_now - tower.top_temperature)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    else:
        upstream_j = heat_wall_output.heat_transfer.downstream_j
    
    # 内側境界からの熱量
    from_inner_j = (
        heat_output.heat_transfer_coefficients.wall_to_bed
        * stream_conds[wall_stream_idx].inner_boundary_area
        / num_sections
        * (temp_inside_cell - temp_now)
        * tower_conds.common.calculation_step_time
        * MINUTE_TO_SECOND
    )
    
    # 外側境界への熱量（外気への放熱）
    to_outer_j = (
        tower_conds.vessel.external_heat_transfer_coef
        * stream_conds[wall_stream_idx].outer_boundary_area
        / num_sections
        * (temp_now - temp_outside)
        * tower_conds.common.calculation_step_time
        * MINUTE_TO_SECOND
    )
    
    # 下流壁への熱量（section=num_sections-1は最下流）
    if section == num_sections - 1:
        downstream_j = (
            tower_conds.vessel.wall_thermal_conductivity
            * stream_conds[wall_stream_idx].cross_section
            * (temp_now - tower.bottom_temperature)
            * tower_conds.common.calculation_step_time
            * MINUTE_TO_SECOND
        )
    else:
        downstream_j = (
            tower_conds.vessel.wall_thermal_conductivity
            * stream_conds[wall_stream_idx].cross_section
            * (temp_now - tower.temp_wall[section + 1])
        )

    # === 到達温度の計算 ===
    args = (
        tower_conds,
        temp_now,
        from_inner_j,
        to_outer_j,
        downstream_j,
        upstream_j,
    )
    temp_reached = optimize.newton(_optimize_wall_temperature, temp_now, args=args)

    return WallHeatBalanceResult(
        temperature=temp_reached,
        heat_transfer=WallHeatTransfer(
            from_inner_j=from_inner_j,
            to_outer_j=to_outer_j,
            downstream_j=downstream_j,
            upstream_j=upstream_j,
        ),
    )


# ============================================================
# 蓋の熱収支計算
# ============================================================

def calculate_lid_heat_balance(
    tower_conds: TowerConditions,
    position: LidPosition,
    state_manager: StateVariables,
    tower_num: int,
    heat_output: HeatBalanceResults,
    heat_wall_output: Dict[int, WallHeatBalanceResult],
) -> LidHeatBalanceResult:
    """
    上下蓋の熱収支計算
    
    容器の上蓋・下蓋の温度変化を計算します。
    
    Args:
        tower_conds: 塔条件
        position: LidPosition.TOP（上蓋）または LidPosition.BOTTOM（下蓋）
        state_manager: 状態変数管理
        tower_num: 塔番号
        heat_output: 各セルの熱収支結果
        heat_wall_output: 壁面の熱収支結果
    
    Returns:
        LidHeatBalanceResult: 蓋の熱収支結果
    """
    tower = state_manager.towers[tower_num]
    is_top = (position == LidPosition.TOP)
    
    # 現在温度
    temp_now = tower.top_temperature if is_top else tower.bottom_temperature
    
    # 外気への熱量 [J]
    heat_to_ambient_j = (
        tower_conds.vessel.external_heat_transfer_coef
        * (temp_now - tower_conds.vessel.ambient_temperature)
        * tower_conds.common.calculation_step_time
        * MINUTE_TO_SECOND
    )
    
    if is_top:
        heat_to_ambient_j *= tower_conds.top.outer_flange_area
    else:
        heat_to_ambient_j *= tower_conds.bottom.outer_flange_area

    num_sections = tower_conds.common.num_sections
    if is_top:
        # stream=1, section=0（最上流）
        stream2_section1_upstream_j = heat_output.get_result(1, 0).heat_transfer.upstream_j
        stream1_section1_upstream_j = heat_output.get_result(0, 0).heat_transfer.upstream_j
        wall_section1_upstream_j = heat_wall_output[0].heat_transfer.upstream_j
        net_heat_input_j = (
            stream2_section1_upstream_j
            - stream1_section1_upstream_j
            - heat_to_ambient_j
            - wall_section1_upstream_j
        )
    else:
        # stream=1, section=num_sections-1（最下流）
        last_section = num_sections - 1
        stream2_lastsection_upstream_j = heat_output.get_result(1, last_section).heat_transfer.upstream_j
        stream1_lastsection_upstream_j = heat_output.get_result(0, last_section).heat_transfer.upstream_j
        net_heat_input_j = (
            stream2_lastsection_upstream_j
            - stream1_lastsection_upstream_j
            - heat_to_ambient_j
            - heat_wall_output[last_section].heat_transfer.downstream_j
        )

    # 到達温度の計算
    args = (tower_conds, temp_now, net_heat_input_j, position)
    temp_reached = optimize.newton(_optimize_top_bottom_temperature, temp_now, args=args)

    return LidHeatBalanceResult(temperature=temp_reached)


# ============================================================
# ヘルパー関数（モード依存の物理量計算）
# ============================================================

def _get_adsorption_heat(
    mode: HeatCalculationMode,
    material_output: Optional[MaterialBalanceResult],
    tower_conds: TowerConditions,
) -> float:
    """
    吸着熱を計算 [J]
    
    CO2が吸着材に吸着される際に発生する熱を計算します。
    停止モードでは吸着が起きないため0です。
    """
    if mode == HeatCalculationMode.VALVE_CLOSED:
        return 0.0
    
    if material_output is None:
        return 0.0
    
    # 吸着・脱着は同じ計算式
    return (
        material_output.adsorption_state.actual_uptake_volume
        / 1000
        / STANDARD_MOLAR_VOLUME
        * tower_conds.feed_gas.co2_molecular_weight
        * tower_conds.feed_gas.co2_adsorption_heat
    )


def _get_inlet_gas_mass(
    mode: HeatCalculationMode,
    material_output: Optional[MaterialBalanceResult],
    tower_conds: TowerConditions,
) -> float:
    """
    流入ガス質量を計算 [g]
    
    セルに流入するガスの質量を計算します。
    停止・脱着モードではガス流入がないため0です。
    """
    if mode in (HeatCalculationMode.VALVE_CLOSED, HeatCalculationMode.DESORPTION):
        return 0.0
    
    if material_output is None:
        return 0.0
    
    return (
        material_output.inlet_gas.co2_volume
        / 1000
        / STANDARD_MOLAR_VOLUME
        * tower_conds.feed_gas.co2_molecular_weight
        + material_output.inlet_gas.n2_volume
        / 1000
        / STANDARD_MOLAR_VOLUME
        * tower_conds.feed_gas.n2_molecular_weight
    )


def _get_gas_specific_heat(
    mode: HeatCalculationMode,
    material_output: Optional[MaterialBalanceResult],
) -> float:
    """
    ガス比熱を取得 [kJ/kg/K]
    
    停止モードでは0を返します。
    """
    if mode == HeatCalculationMode.VALVE_CLOSED:
        return 0.0
    
    if material_output is None:
        return 0.0
    
    return material_output.gas_properties.specific_heat


def _calculate_thermocouple_temperature(
    tower_conds: TowerConditions,
    tower,
    stream: int,
    section: int,
    mode: HeatCalculationMode,
    wall_to_bed_htc: float,
) -> float:
    """
    熱電対温度を計算
    
    熱電対は充填層内に設置されており、層温度に追従して変化します。
    熱電対と層の間の伝熱を考慮して計算します。
    """
    # 熱電対熱容量 [J/K]
    heat_capacity = tower_conds.thermocouple.specific_heat * tower_conds.thermocouple.weight
    
    # 熱電対側面積 [m2]
    S_side = 0.004 * np.pi * 0.1
    
    # 伝熱係数
    thermocouple_htc = wall_to_bed_htc
    
    # 補正係数（脱着モードでは異なる）
    if mode != HeatCalculationMode.DESORPTION:
        correction_factor = tower_conds.thermocouple.heat_transfer_correction_factor
    else:
        correction_factor = 100
    
    # 熱流束 [W]（時間あたりの熱量）
    heat_flux_w = (
        thermocouple_htc
        * correction_factor
        * S_side
        * (tower.cell(stream, section).temp - tower.cell(stream, section).thermocouple_temperature)
    )
    
    # 温度上昇 [℃]
    temp_increase = (
        heat_flux_w
        * tower_conds.common.calculation_step_time
        * MINUTE_TO_SECOND
        / heat_capacity
    )
    
    return tower.cell(stream, section).thermocouple_temperature + temp_increase


# ============================================================
# Newton法用の最適化関数
# ============================================================

def _optimize_bed_temperature(
    temp_reached: float,
    tower_conds: TowerConditions,
    gas_specific_heat: float,
    inlet_gas_mass: float,
    temp_now: float,
    adsorption_heat_j: float,
    from_inner_j: float,
    to_outer_j: float,
    downstream_j: float,
    upstream_j: float,
    stream: int,
) -> float:
    """
    充填層到達温度のソルバー用関数
    
    熱収支基準と時間基準の差分が0になる温度を求めます。
    """
    stream_conds = tower_conds.stream_conditions
    
    # 流入ガスが受け取る熱 [J]
    H_gas_j = gas_specific_heat * inlet_gas_mass * (temp_reached - temp_now)
    
    # 充填層が受け取る熱（時間基準）[J]
    H_bed_time_j = (
        tower_conds.packed_bed.heat_capacity
        * stream_conds[stream].area_fraction
        / tower_conds.common.num_sections
        * (temp_reached - temp_now)
    )
    
    # 充填層が受け取る熱（熱収支基準）[J]
    H_bed_balance_j = (
        adsorption_heat_j
        - H_gas_j
        + from_inner_j
        - to_outer_j
        - downstream_j
        - upstream_j
    )
    
    return H_bed_balance_j - H_bed_time_j


def _optimize_wall_temperature(
    temp_reached: float,
    tower_conds: TowerConditions,
    temp_now: float,
    from_inner_j: float,
    to_outer_j: float,
    downstream_j: float,
    upstream_j: float,
) -> float:
    """
    壁面到達温度のソルバー用関数
    """
    stream_conds = tower_conds.stream_conditions
    
    # 壁が受け取る熱（熱収支基準）[J]
    H_wall_balance_j = (
        from_inner_j
        - upstream_j
        - to_outer_j
        - downstream_j
    )
    
    # 壁が受け取る熱（時間基準）[J]
    H_wall_time_j = (
        tower_conds.vessel.wall_specific_heat_capacity
        * stream_conds[tower_conds.common.num_streams].wall_weight
        * (temp_reached - temp_now)
    )
    
    return H_wall_balance_j - H_wall_time_j


def _optimize_top_bottom_temperature(
    temp_reached: float,
    tower_conds: TowerConditions,
    temp_now: float,
    net_heat_input_j: float,
    position: LidPosition,
) -> float:
    """
    蓋到達温度のソルバー用関数
    """
    # 蓋が受け取る熱（時間基準）[J]
    H_lid_time_j = tower_conds.vessel.wall_specific_heat_capacity * (temp_reached - temp_now)
    
    if position == LidPosition.TOP:
        H_lid_time_j *= tower_conds.top.flange_total_weight
    else:
        H_lid_time_j *= tower_conds.bottom.flange_total_weight
    
    return net_heat_input_j - H_lid_time_j
