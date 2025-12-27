"""停止モード

PSA担当者向け説明:
バルブが閉じている状態（停止モード）の計算を行います。

停止モードとは:
- 入口・出口バルブが閉じた状態
- ガスの流入・流出がない
- 吸着量は変化しない
- 熱は隣接セル・壁・蓋との間で移動する

稼働工程表での対応:
- 塔の運転モードが「停止」の場合に使用
"""

from dataclasses import dataclass
from typing import Dict

from operation_modes.mode_types import OperationMode
from operation_modes.common import calculate_full_tower
from config.sim_conditions import TowerConditions
from core.state import (
    StateVariables,
    MassBalanceResults,
    HeatBalanceResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
)


@dataclass
class StopModeResult:
    """
    停止モードの計算結果
    
    PSA担当者向け説明:
    停止モードでは圧力変化がないため、
    物質収支・熱収支の結果のみを保持します。
    """
    material: MassBalanceResults
    heat: HeatBalanceResults
    heat_wall: Dict[int, WallHeatBalanceResult]
    heat_lid: Dict[str, LidHeatBalanceResult]


def execute_stop_mode(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
) -> StopModeResult:
    """
    停止モードの計算を実行
    
    PSA担当者向け説明:
    バルブ閉鎖状態での温度変化を計算します。
    ガス流入がないため吸着量は変化しませんが、
    熱は隣接セル・壁・外気との間で移動します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
    
    Returns:
        StopModeResult: 停止モードの計算結果
    
    使用例:
        result = execute_stop_mode(tower_conds, state_manager, tower_num=1)
        # 熱収支結果を確認
        temp = result.heat.get_result(stream=1, section=3).cell_temperatures.bed_temperature
    """
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.STOP,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
    )
    
    return StopModeResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
    )
