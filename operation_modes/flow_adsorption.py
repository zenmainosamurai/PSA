"""流通吸着モード

PSA担当者向け説明:
ガスを連続的に流通させながら吸着を行うモードの計算を行います。

流通吸着とは:
- 入口・出口バルブが開いた状態
- ガスが塔内を連続的に流れる
- 流通中にCO2が吸着される
- 圧力は導入ガス圧力に維持される

運転パターン:
- 単独/上流: 1塔のみ、または直列運転の上流塔
- 下流: 直列運転の下流塔（上流塔からのガスを受け取る）

稼働工程表での対応:
- 「流通吸着_単独/上流」
- 「流通吸着_下流」
"""

from dataclasses import dataclass
from typing import Dict

from operation_modes.mode_types import OperationMode
from operation_modes.common import calculate_full_tower, distribute_inflow_gas
from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MassBalanceResults,
    HeatBalanceResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
)


@dataclass
class FlowAdsorptionResult:
    """
    流通吸着モードの計算結果
    
    PSA担当者向け説明:
    流通吸着では導入ガス圧力が維持されるため、
    total_pressure に現在の圧力が入ります。
    """
    material: MassBalanceResults
    heat: HeatBalanceResults
    heat_wall: Dict[int, WallHeatBalanceResult]
    heat_lid: Dict[str, LidHeatBalanceResult]
    total_pressure: float  # 全圧 [MPaA]


def execute_flow_adsorption_upstream(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
) -> FlowAdsorptionResult:
    """
    流通吸着（単独/上流）の計算を実行
    
    PSA担当者向け説明:
    導入ガスを直接受け取る塔での流通吸着を計算します。
    圧力は導入ガス圧力に維持されます。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
    
    Returns:
        FlowAdsorptionResult: 流通吸着の計算結果
    
    使用例:
        result = execute_flow_adsorption_upstream(tower_conds, state_manager, tower_num=1)
        # 吸着後の吸着量を確認
        loading = result.material.get_result(1, 3).adsorption_state.updated_loading
    """
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.FLOW_ADSORPTION_UPSTREAM,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
    )
    
    # 圧力は導入ガス圧力に維持
    total_pressure = tower_conds.feed_gas.total_pressure
    
    return FlowAdsorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        total_pressure=total_pressure,
    )


def execute_flow_adsorption_downstream(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    upstream_mass_balance: MassBalanceResults,
) -> FlowAdsorptionResult:
    """
    流通吸着（下流）の計算を実行
    
    PSA担当者向け説明:
    直列運転において、上流塔から流出したガスを受け取る
    下流塔での流通吸着を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        upstream_mass_balance: 上流塔の物質収支結果
    
    Returns:
        FlowAdsorptionResult: 流通吸着の計算結果
    
    使用例:
        # 上流塔の計算結果を使用
        result = execute_flow_adsorption_downstream(
            tower_conds, state_manager, tower_num=2,
            upstream_mass_balance=upstream_result.material
        )
    """
    # 上流からのガスを各ストリームに分配
    distributed_inflows = distribute_inflow_gas(tower_conds, upstream_mass_balance)
    
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        external_inflow_gas=distributed_inflows,
    )
    
    # 圧力は導入ガス圧力に維持
    total_pressure = tower_conds.feed_gas.total_pressure
    
    return FlowAdsorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        total_pressure=total_pressure,
    )
