"""真空脱着モード

PSA担当者向け説明:
真空ポンプで塔内を減圧し、吸着剤からCO2を脱着させるモードの計算を行います。

真空脱着とは:
- 真空ポンプで塔内を減圧
- 平衡吸着量が低下し、吸着剤からCO2が脱着
- 脱着したCO2は真空ポンプで排出
- PSAサイクルの中でCO2を回収する重要な工程

計算の流れ:
1. 真空ポンプによる排気量計算
2. 脱着に伴う物質収支計算
3. 脱着熱による温度変化計算
4. 圧力変化計算

稼働工程表での対応:
- 「真空脱着」
"""

from dataclasses import dataclass
from typing import Dict, Optional

from operation_modes.mode_types import OperationMode
from operation_modes.common import calculate_full_tower
from config.sim_conditions import TowerConditions
from core.state import (
    StateVariables,
    MassBalanceResults,
    HeatBalanceResults,
    MoleFractionResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    VacuumPumpingResult,
)
from physics.pressure import (
    calculate_vacuum_pumping_result,
    calculate_pressure_after_vacuum_desorption,
)


@dataclass
class VacuumDesorptionResult:
    """
    真空脱着モードの計算結果
    
    PSA担当者向け説明:
    真空脱着の結果には以下が含まれます:
    - 物質収支・熱収支結果
    - 脱着ガスのモル分率
    - 真空排気による累積排気量
    - 脱着後の圧力
    """
    material: MassBalanceResults
    heat: HeatBalanceResults
    heat_wall: Dict[int, WallHeatBalanceResult]
    heat_lid: Dict[str, LidHeatBalanceResult]
    mole_fraction: MoleFractionResults
    accumulative_vacuum_amount: float  # 累積真空排気量 [Nm3]
    pressure_after_vacuum_desorption: float  # 脱着後の圧力 [MPaA]


def execute_vacuum_desorption(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    previous_accumulative_vacuum_amount: float = 0.0,
) -> VacuumDesorptionResult:
    """
    真空脱着の計算を実行
    
    PSA担当者向け説明:
    真空脱着では、以下の順序で計算を行います:
    1. 真空ポンプの排気量計算（圧力・温度依存）
    2. 脱着による吸着量減少の計算
    3. 脱着熱による温度低下の計算
    4. 圧力変化の計算
    
    真空ポンプの特性:
    - 排気量は吸入圧力に依存（低圧ほど低下）
    - 実験データに基づく特性曲線を使用
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        previous_accumulative_vacuum_amount: 前ステップまでの累積排気量 [Nm3]
    
    Returns:
        VacuumDesorptionResult: 真空脱着の計算結果
    
    使用例:
        result = execute_vacuum_desorption(
            tower_conds, state_manager, tower_num=1,
            previous_accumulative_vacuum_amount=0.5
        )
        # 脱着後の圧力を確認
        print(f"脱着後圧力: {result.pressure_after_vacuum_desorption} MPaA")
        # 累積排気量を確認
        print(f"累積排気量: {result.accumulative_vacuum_amount} Nm3")
    """
    # 真空排気結果の計算
    vacuum_pumping_results = calculate_vacuum_pumping_result(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
    )
    
    # 塔全体の計算を実行（真空排気結果を使用）
    tower_results = calculate_full_tower(
        mode=OperationMode.VACUUM_DESORPTION,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        vacuum_pumping_results=vacuum_pumping_results,
    )
    
    # モル分率結果の取得（脱着モードでは必ず存在）
    mole_fraction = tower_results.mole_fraction
    if mole_fraction is None:
        raise RuntimeError("真空脱着モードではモル分率結果が必要です")
    
    # 脱着後の圧力計算
    pressure_after = calculate_pressure_after_vacuum_desorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        vacuum_pumping_results=vacuum_pumping_results,
    )
    
    # 累積排気量の更新
    accumulative = (
        previous_accumulative_vacuum_amount +
        vacuum_pumping_results.total_pumped_amount
    )
    
    return VacuumDesorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        mole_fraction=mole_fraction,
        accumulative_vacuum_amount=accumulative,
        pressure_after_vacuum_desorption=pressure_after,
    )
