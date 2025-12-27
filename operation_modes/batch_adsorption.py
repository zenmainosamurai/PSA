"""バッチ吸着モード

PSA担当者向け説明:
密閉状態でガスを導入しながら吸着を行うモードの計算を行います。

バッチ吸着とは:
- 出口バルブが閉じた状態でガスを導入
- 塔内圧力が上昇する
- 圧力上昇に伴い吸着が進行

運転パターン:
- 上流: 導入ガスを直接受け取る塔
- 下流: 上流塔からのガスを受け取る塔
- 圧調弁あり: 圧力調整弁がある場合

稼働工程表での対応:
- 「バッチ吸着_上流」
- 「バッチ吸着_下流」
- 「バッチ吸着_上流（圧調弁あり）」
- 「バッチ吸着_下流（圧調弁あり）」
"""

from dataclasses import dataclass
from typing import Dict, Optional

from operation_modes.mode_types import OperationMode
from operation_modes.common import calculate_full_tower, distribute_inflow_gas
from config.sim_conditions import TowerConditions
from core.state import (
    StateVariables,
    MassBalanceResults,
    HeatBalanceResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
)
from physics.pressure import calculate_pressure_after_batch_adsorption


@dataclass
class BatchAdsorptionResult:
    """
    バッチ吸着モードの計算結果
    
    PSA担当者向け説明:
    バッチ吸着では圧力が変化するため、
    pressure_after_batch_adsorption に吸着後の圧力が入ります。
    """
    material: MassBalanceResults
    heat: HeatBalanceResults
    heat_wall: Dict[int, WallHeatBalanceResult]
    heat_lid: Dict[str, LidHeatBalanceResult]
    pressure_after_batch_adsorption: float  # バッチ吸着後の圧力 [MPaA]


def execute_batch_adsorption_upstream(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    is_series_operation: bool = False,
) -> BatchAdsorptionResult:
    """
    バッチ吸着（上流）の計算を実行
    
    PSA担当者向け説明:
    導入ガスを直接受け取る塔でのバッチ吸着を計算します。
    密閉状態でガスを導入するため、圧力が上昇します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        is_series_operation: 直列運転かどうか
    
    Returns:
        BatchAdsorptionResult: バッチ吸着の計算結果
    """
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.BATCH_ADSORPTION_UPSTREAM,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
    )
    
    # バッチ吸着後の圧力計算
    pressure_after = calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=is_series_operation,
        has_pressure_valve=False,
    )
    
    return BatchAdsorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        pressure_after_batch_adsorption=pressure_after,
    )


def execute_batch_adsorption_upstream_with_valve(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
) -> BatchAdsorptionResult:
    """
    バッチ吸着（上流・圧調弁あり）の計算を実行
    
    PSA担当者向け説明:
    圧力調整弁がある場合のバッチ吸着を計算します。
    圧調弁により圧力は導入ガス圧力に維持されます。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
    
    Returns:
        BatchAdsorptionResult: バッチ吸着の計算結果
    """
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
    )
    
    # 圧調弁ありの場合は導入ガス圧力に維持
    pressure_after = tower_conds.feed_gas.total_pressure
    
    return BatchAdsorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        pressure_after_batch_adsorption=pressure_after,
    )


def execute_batch_adsorption_downstream(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    is_series_operation: bool,
    upstream_mass_balance: MassBalanceResults,
    residual_gas_composition: Optional[MassBalanceResults] = None,
) -> BatchAdsorptionResult:
    """
    バッチ吸着（下流）の計算を実行
    
    PSA担当者向け説明:
    直列運転において、上流塔から流出したガスを受け取る
    下流塔でのバッチ吸着を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        is_series_operation: 直列運転かどうか
        upstream_mass_balance: 上流塔の物質収支結果
        residual_gas_composition: 残留ガス組成（オプション）
    
    Returns:
        BatchAdsorptionResult: バッチ吸着の計算結果
    """
    # 上流からのガスを各ストリームに分配
    distributed_inflows = distribute_inflow_gas(tower_conds, upstream_mass_balance)
    
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        external_inflow_gas=distributed_inflows,
        residual_gas_composition=residual_gas_composition,
    )
    
    # バッチ吸着後の圧力計算
    pressure_after = calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=is_series_operation,
        has_pressure_valve=False,
    )
    
    return BatchAdsorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        pressure_after_batch_adsorption=pressure_after,
    )


def execute_batch_adsorption_downstream_with_valve(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    upstream_mass_balance: MassBalanceResults,
    residual_gas_composition: Optional[MassBalanceResults] = None,
) -> BatchAdsorptionResult:
    """
    バッチ吸着（下流・圧調弁あり）の計算を実行
    
    PSA担当者向け説明:
    圧力調整弁がある場合の下流塔でのバッチ吸着を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        upstream_mass_balance: 上流塔の物質収支結果
        residual_gas_composition: 残留ガス組成（オプション）
    
    Returns:
        BatchAdsorptionResult: バッチ吸着の計算結果
    """
    # 上流からのガスを各ストリームに分配
    distributed_inflows = distribute_inflow_gas(tower_conds, upstream_mass_balance)
    
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        external_inflow_gas=distributed_inflows,
        residual_gas_composition=residual_gas_composition,
    )
    
    # 圧調弁ありの場合の圧力計算
    pressure_after = calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=True,
        has_pressure_valve=True,
    )
    
    return BatchAdsorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        pressure_after_batch_adsorption=pressure_after,
    )
