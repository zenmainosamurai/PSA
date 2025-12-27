"""初回ガス導入モード

PSA担当者向け説明:
シミュレーション開始時に、塔にガスを導入する初期化工程の計算を行います。

初回ガス導入とは:
- PSAサイクル開始前の初期化工程
- 塔内に導入ガスを充填
- 初期吸着状態を形成
- 通常のバッチ吸着と同様の計算を行う

計算内容:
- 物質収支（吸着量の変化）
- 熱収支（吸着熱による温度上昇）
- 圧力変化（ガス導入による加圧）

稼働工程表での対応:
- 「初回ガス導入」
"""

from dataclasses import dataclass
from typing import Dict

from operation_modes.mode_types import OperationMode
from operation_modes.common import calculate_full_tower
from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MassBalanceResults,
    HeatBalanceResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
)
from physics.pressure import calculate_pressure_after_batch_adsorption


@dataclass
class InitialGasIntroductionResult:
    """
    初回ガス導入モードの計算結果
    
    PSA担当者向け説明:
    初回ガス導入はバッチ吸着と同様の結果形式です。
    pressure_after_introduction にガス導入後の圧力が入ります。
    """
    material: MassBalanceResults
    heat: HeatBalanceResults
    heat_wall: Dict[int, WallHeatBalanceResult]
    heat_lid: Dict[str, LidHeatBalanceResult]
    pressure_after_introduction: float  # ガス導入後の圧力 [MPaA]
    
    @property
    def pressure_after_batch_adsorption(self) -> float:
        """互換性のための別名（state_variables.update_from_calc_outputで使用）"""
        return self.pressure_after_introduction


def execute_initial_gas_introduction(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
) -> InitialGasIntroductionResult:
    """
    初回ガス導入の計算を実行
    
    PSA担当者向け説明:
    シミュレーション開始時に塔にガスを導入する工程を計算します。
    バッチ吸着（上流）と同様の計算を行いますが、
    初期状態からの計算であることを明示するために
    別のモードとして扱っています。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
    
    Returns:
        InitialGasIntroductionResult: 初回ガス導入の計算結果
    
    使用例:
        result = execute_initial_gas_introduction(
            tower_conds, state_manager, tower_num=1
        )
        # 導入後の圧力を確認
        print(f"導入後圧力: {result.pressure_after_introduction} MPaA")
    """
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.INITIAL_GAS_INTRODUCTION,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
    )
    
    # ガス導入後の圧力計算（バッチ吸着と同様）
    pressure_after = calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=False,
        has_pressure_valve=False,
    )
    
    return InitialGasIntroductionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        pressure_after_introduction=pressure_after,
    )
