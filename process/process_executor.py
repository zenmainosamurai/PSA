"""工程実行ロジック

稼働工程表に従って各工程を実行するモジュールです。
塔間の依存関係（上流→下流、減圧→加圧）を考慮して、
適切な順序で各塔の計算を実行します。

計算順序の制御:
- 上流/下流ペア: 上流塔を先に計算し、結果を下流塔に渡す
- 均圧ペア: 減圧塔を先に計算し、流出ガス情報を加圧塔に渡す
- 独立運転: 各塔を順番に計算

稼働工程表の対応モード:
- 流通吸着: 単独/上流、下流
- バッチ吸着: 上流、下流、圧調弁あり
- 均圧: 減圧、加圧
- 真空脱着
- 停止
- 初回ガス導入
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from operation_modes import (
    OperationMode,
    UPSTREAM_DOWNSTREAM_PAIRS,
    # 停止モード
    execute_stop_mode,
    # 流通吸着
    execute_flow_adsorption_upstream,
    execute_flow_adsorption_downstream,
    # バッチ吸着
    execute_batch_adsorption_upstream,
    execute_batch_adsorption_upstream_with_valve,
    execute_batch_adsorption_downstream,
    execute_batch_adsorption_downstream_with_valve,
    # 均圧
    execute_equalization_depressurization,
    execute_equalization_pressurization,
    # 真空脱着
    execute_vacuum_desorption,
    # 初回ガス導入
    execute_initial_gas_introduction,
)
from config.sim_conditions import SimulationConditions, TowerConditions
from state import StateVariables, MassBalanceResults


@dataclass
class TowerCalculationOutput:
    """
    塔ごとの計算出力
    
    1つの塔の計算結果をまとめたものです。
    シミュレーション結果の記録に使用します。
    """
    material: MassBalanceResults  # 物質収支結果
    heat: dict  # 熱収支結果
    heat_wall: dict  # 壁面熱収支結果
    heat_lid: dict  # 蓋熱収支結果
    others: dict  # その他（圧力、モル分率など）


def execute_mode_list(
    sim_conds: SimulationConditions,
    mode_list: List[str],
    state_manager: StateVariables,
    residual_gas_composition: Optional[MassBalanceResults] = None,
    is_first_step: bool = False,
) -> Tuple[Dict[int, TowerCalculationOutput], Optional[MassBalanceResults]]:
    """
    全塔のモードリストを実行
    
    稼働工程表の1行（全塔のモード指定）を実行します。
    塔間の依存関係を考慮して適切な順序で計算します。
    
    Args:
        sim_conds: シミュレーション条件
        mode_list: 各塔のモードリスト ["流通吸着_単独/上流", "停止", "真空脱着"]
        state_manager: 状態変数管理
        residual_gas_composition: 残留ガス組成（バッチ吸着下流で使用）
        is_first_step: 工程の初回ステップかどうか（初期化処理の実行に使用）
    
    Returns:
        Tuple[Dict[int, TowerCalculationOutput], Optional[MassBalanceResults]]:
            - 各塔の計算出力
            - 更新された残留ガス組成（均圧加圧後に設定）
    
    使用例:
        outputs, residual = execute_mode_list(
            sim_conds, ["流通吸着_単独/上流", "流通吸着_下流", "停止"],
            state_manager,
            is_first_step=True,  # 工程の初回のみTrue
        )
    """
    # 工程初回の初期化処理
    if is_first_step:
        _initialize_process(state_manager, sim_conds, mode_list)
    
    num_towers = sim_conds.num_towers
    outputs: Dict[int, TowerCalculationOutput] = {}
    updated_residual = residual_gas_composition
    
    # 計算パターンの判定と実行
    pattern = _detect_calculation_pattern(mode_list)
    
    if pattern == "upstream_downstream":
        outputs, updated_residual = _execute_upstream_downstream(
            sim_conds, mode_list, state_manager, residual_gas_composition
        )
    elif pattern == "equalization":
        outputs, updated_residual = _execute_equalization(
            sim_conds, mode_list, state_manager
        )
    else:
        # 独立運転
        outputs = _execute_independent(sim_conds, mode_list, state_manager)
    
    return outputs, updated_residual


def _detect_calculation_pattern(mode_list: List[str]) -> str:
    """計算パターンを検出"""
    # 上流・下流ペアの確認
    for upstream_mode, downstream_mode in UPSTREAM_DOWNSTREAM_PAIRS:
        if upstream_mode.value in mode_list and downstream_mode.value in mode_list:
            return "upstream_downstream"
    
    # 均圧ペアの確認
    if "均圧_減圧" in mode_list and "均圧_加圧" in mode_list:
        return "equalization"
    
    return "independent"


def _execute_upstream_downstream(
    sim_conds: SimulationConditions,
    mode_list: List[str],
    state_manager: StateVariables,
    residual_gas_composition: Optional[MassBalanceResults],
) -> Tuple[Dict[int, TowerCalculationOutput], Optional[MassBalanceResults]]:
    """上流・下流ペアの実行"""
    outputs: Dict[int, TowerCalculationOutput] = {}
    num_towers = sim_conds.num_towers
    
    # 上流・下流の塔番号を特定
    upstream_tower_num = None
    downstream_tower_num = None
    upstream_mode = None
    downstream_mode = None
    
    for up_mode, down_mode in UPSTREAM_DOWNSTREAM_PAIRS:
        if up_mode.value in mode_list and down_mode.value in mode_list:
            upstream_tower_num = mode_list.index(up_mode.value) + 1
            downstream_tower_num = mode_list.index(down_mode.value) + 1
            upstream_mode = up_mode
            downstream_mode = down_mode
            break
    
    # 上流塔の計算
    upstream_output = _execute_single_tower(
        tower_conds=sim_conds.get_tower(upstream_tower_num),
        mode=upstream_mode,
        tower_num=upstream_tower_num,
        state_manager=state_manager,
    )
    outputs[upstream_tower_num] = upstream_output
    
    # 下流塔の計算（上流の結果を使用）
    downstream_output = _execute_single_tower(
        tower_conds=sim_conds.get_tower(downstream_tower_num),
        mode=downstream_mode,
        tower_num=downstream_tower_num,
        state_manager=state_manager,
        upstream_mass_balance=upstream_output.material,
        residual_gas_composition=residual_gas_composition,
    )
    outputs[downstream_tower_num] = downstream_output
    
    # 残りの塔
    for tower_num in range(1, num_towers + 1):
        if tower_num in [upstream_tower_num, downstream_tower_num]:
            continue
        mode_str = mode_list[tower_num - 1]
        mode = OperationMode.from_japanese(mode_str)
        output = _execute_single_tower(
            tower_conds=sim_conds.get_tower(tower_num),
            mode=mode,
            tower_num=tower_num,
            state_manager=state_manager,
        )
        outputs[tower_num] = output
    
    return outputs, residual_gas_composition


def _execute_equalization(
    sim_conds: SimulationConditions,
    mode_list: List[str],
    state_manager: StateVariables,
) -> Tuple[Dict[int, TowerCalculationOutput], Optional[MassBalanceResults]]:
    """均圧ペアの実行"""
    outputs: Dict[int, TowerCalculationOutput] = {}
    num_towers = sim_conds.num_towers
    
    # 減圧・加圧の塔番号を特定
    depressurization_tower_num = mode_list.index("均圧_減圧") + 1
    pressurization_tower_num = mode_list.index("均圧_加圧") + 1
    
    # 加圧側の圧力を取得
    pressurization_pressure = state_manager.towers[pressurization_tower_num].total_press
    
    # 減圧塔の計算
    depressurization_result = execute_equalization_depressurization(
        tower_conds=sim_conds.get_tower(depressurization_tower_num),
        state_manager=state_manager,
        tower_num=depressurization_tower_num,
        target_tower_pressure=pressurization_pressure,
    )
    
    # 状態更新
    _update_state_from_result(
        state_manager, depressurization_tower_num,
        OperationMode.EQUALIZATION_DEPRESSURIZATION, depressurization_result
    )
    
    outputs[depressurization_tower_num] = TowerCalculationOutput(
        material=depressurization_result.material,
        heat=depressurization_result.heat,
        heat_wall=depressurization_result.heat_wall,
        heat_lid=depressurization_result.heat_lid,
        others=_extract_others(state_manager, depressurization_tower_num),
    )
    
    # 加圧塔の計算（減圧結果を使用）
    pressurization_result = execute_equalization_pressurization(
        tower_conds=sim_conds.get_tower(pressurization_tower_num),
        state_manager=state_manager,
        tower_num=pressurization_tower_num,
        upstream_depressurization_result=depressurization_result,
    )
    
    # 状態更新
    _update_state_from_result(
        state_manager, pressurization_tower_num,
        OperationMode.EQUALIZATION_PRESSURIZATION, pressurization_result
    )
    
    outputs[pressurization_tower_num] = TowerCalculationOutput(
        material=pressurization_result.material,
        heat=pressurization_result.heat,
        heat_wall=pressurization_result.heat_wall,
        heat_lid=pressurization_result.heat_lid,
        others=_extract_others(state_manager, pressurization_tower_num),
    )
    
    # 残りの塔
    for tower_num in range(1, num_towers + 1):
        if tower_num in [depressurization_tower_num, pressurization_tower_num]:
            continue
        mode_str = mode_list[tower_num - 1]
        mode = OperationMode.from_japanese(mode_str)
        output = _execute_single_tower(
            tower_conds=sim_conds.get_tower(tower_num),
            mode=mode,
            tower_num=tower_num,
            state_manager=state_manager,
        )
        outputs[tower_num] = output
    
    # 均圧加圧後の残留ガス組成を返す
    return outputs, pressurization_result.material


def _execute_independent(
    sim_conds: SimulationConditions,
    mode_list: List[str],
    state_manager: StateVariables,
) -> Dict[int, TowerCalculationOutput]:
    """独立運転の実行"""
    outputs: Dict[int, TowerCalculationOutput] = {}
    num_towers = sim_conds.num_towers
    
    for tower_num in range(1, num_towers + 1):
        mode_str = mode_list[tower_num - 1]
        mode = OperationMode.from_japanese(mode_str)
        output = _execute_single_tower(
            tower_conds=sim_conds.get_tower(tower_num),
            mode=mode,
            tower_num=tower_num,
            state_manager=state_manager,
        )
        outputs[tower_num] = output
    
    return outputs


def _execute_single_tower(
    tower_conds: TowerConditions,
    mode: OperationMode,
    tower_num: int,
    state_manager: StateVariables,
    upstream_mass_balance: Optional[MassBalanceResults] = None,
    residual_gas_composition: Optional[MassBalanceResults] = None,
) -> TowerCalculationOutput:
    """単一塔の計算を実行"""
    result = None
    
    # モードに応じた計算を実行
    if mode == OperationMode.INITIAL_GAS_INTRODUCTION:
        # 初回ガス導入は特殊な条件設定が必要
        tower_conds_copy = deepcopy(tower_conds)
        tower_conds_copy.feed_gas.co2_flow_rate = 20
        tower_conds_copy.feed_gas.n2_flow_rate = 25.2
        result = execute_initial_gas_introduction(
            tower_conds=tower_conds_copy,
            state_manager=state_manager,
            tower_num=tower_num,
        )
    
    elif mode == OperationMode.STOP:
        result = execute_stop_mode(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
        )
    
    elif mode == OperationMode.FLOW_ADSORPTION_UPSTREAM:
        result = execute_flow_adsorption_upstream(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
        )
    
    elif mode == OperationMode.FLOW_ADSORPTION_DOWNSTREAM:
        result = execute_flow_adsorption_downstream(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
            upstream_mass_balance=upstream_mass_balance,
        )
    
    elif mode == OperationMode.BATCH_ADSORPTION_UPSTREAM:
        result = execute_batch_adsorption_upstream(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
            is_series_operation=True,
        )
    
    elif mode == OperationMode.BATCH_ADSORPTION_DOWNSTREAM:
        result = execute_batch_adsorption_downstream(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
            is_series_operation=True,
            upstream_mass_balance=upstream_mass_balance,
            residual_gas_composition=residual_gas_composition,
        )
    
    elif mode == OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE:
        result = execute_batch_adsorption_upstream_with_valve(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
        )
    
    elif mode == OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE:
        result = execute_batch_adsorption_downstream_with_valve(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
            upstream_mass_balance=upstream_mass_balance,
            residual_gas_composition=residual_gas_composition,
        )
    
    elif mode == OperationMode.VACUUM_DESORPTION:
        result = execute_vacuum_desorption(
            tower_conds=tower_conds,
            state_manager=state_manager,
            tower_num=tower_num,
        )
    
    else:
        raise ValueError(f"未対応の運転モード: {mode}")
    
    # 状態更新
    _update_state_from_result(state_manager, tower_num, mode, result)
    
    # 出力オブジェクトの構築
    return TowerCalculationOutput(
        material=result.material,
        heat=result.heat,
        heat_wall=result.heat_wall,
        heat_lid=result.heat_lid,
        others=_extract_others(state_manager, tower_num),
    )


def _update_state_from_result(
    state_manager: StateVariables,
    tower_num: int,
    mode: OperationMode,
    result,
) -> None:
    """計算結果から状態変数を更新"""
    # 既存のstate_manager.update_from_calc_outputを使用
    # モード名を日本語文字列に変換して渡す
    state_manager.update_from_calc_output(tower_num, mode.value, result)


def _extract_others(state_manager: StateVariables, tower_num: int) -> dict:
    """その他の状態変数を抽出"""
    tower = state_manager.towers[tower_num]
    return {
        "total_pressure": tower.total_press,
        "co2_mole_fraction": tower.co2_mole_fraction.copy(),
        "n2_mole_fraction": tower.n2_mole_fraction.copy(),
        "cumulative_co2_recovered": tower.cumulative_co2_recovered,
        "cumulative_n2_recovered": tower.cumulative_n2_recovered,
    }


def _initialize_process(
    state_manager: StateVariables,
    sim_conds: SimulationConditions,
    mode_list: List[str],
) -> None:
    """
    工程開始時の初期化処理
    
    各工程の開始時に必要な初期化処理を行います。
    現在はバッチ吸着の圧力平均化のみ実装。
    
    Args:
        state_manager: 状態変数管理
        sim_conds: シミュレーション条件
        mode_list: モードリスト
    """
    # バッチ吸着（上流・下流）の圧力平均化
    if "バッチ吸着_上流" in mode_list and "バッチ吸着_下流" in mode_list:
        upstream_tower_num = mode_list.index("バッチ吸着_上流") + 1
        downstream_tower_num = mode_list.index("バッチ吸着_下流") + 1
        
        upstream_state = state_manager.towers[upstream_tower_num]
        downstream_state = state_manager.towers[downstream_tower_num]
        
        upstream_void = sim_conds.get_tower(upstream_tower_num).packed_bed.void_volume
        downstream_void = sim_conds.get_tower(downstream_tower_num).packed_bed.void_volume
        
        # 圧力の重み付け平均
        total_press_mean = (
            upstream_state.total_press * upstream_void +
            downstream_state.total_press * downstream_void
        ) / (upstream_void + downstream_void)
        
        upstream_state.total_press = total_press_mean
        downstream_state.total_press = total_press_mean
