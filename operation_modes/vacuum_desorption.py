"""真空脱着モード

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
from typing import Dict

from common.constants import CELSIUS_TO_KELVIN_OFFSET, MPA_TO_PA

from operation_modes.mode_types import OperationMode
from operation_modes.common import calculate_full_tower
from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MassBalanceResults,
    HeatBalanceResults,
    MoleFractionResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    VacuumPumpingResult,
)
from physics.pressure import (
    calculate_pressure_after_vacuum_desorption,
    _calculate_average_temperature,
    _calculate_average_mole_fractions,
)
from physics.gas_properties import (
    calculate_mixed_gas_viscosity,
    calculate_mixed_gas_density,
)
from physics.pipe_flow import (
    calculate_vacuum_pump_flow,
    calculate_pressure_from_moles,
)
from physics.recovery import (
    calculate_desorption_amount,
    calculate_co2_recovery_concentration,
)


@dataclass
class VacuumDesorptionResult:
    """
    真空脱着モードの計算結果
    
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
    
    @property
    def mol_fraction(self) -> MoleFractionResults:
        """互換性のための別名（state_variables.update_from_calc_outputで使用）"""
        return self.mole_fraction
    accumulative_vacuum_amount: float  # 累積真空排気量 [Nm3]
    pressure_after_vacuum_desorption: float  # 脱着後の圧力 [MPaA]
    accum_vacuum_amt: VacuumPumpingResult = None  # 互換性のため（state_variables.update_from_calc_outputで使用）


def execute_vacuum_desorption(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    previous_accumulative_vacuum_amount: float = 0.0,
) -> VacuumDesorptionResult:
    """
    真空脱着の計算を実行
    
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
    # 真空排気結果の計算（物理計算の組み合わせ）
    vacuum_pumping_results = _calculate_vacuum_pumping(
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
        mole_fraction_results=mole_fraction,
        vacuum_pumping_results=vacuum_pumping_results,
    )
    
    # 累積排気量の更新（CO2とN2の合計）
    accumulative = (
        previous_accumulative_vacuum_amount +
        vacuum_pumping_results.cumulative_co2_recovered +
        vacuum_pumping_results.cumulative_n2_recovered
    )
    
    return VacuumDesorptionResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        mole_fraction=mole_fraction,
        accumulative_vacuum_amount=accumulative,
        pressure_after_vacuum_desorption=pressure_after,
        accum_vacuum_amt=vacuum_pumping_results,  # 互換性のため
    )


# ============================================================
# ヘルパー関数（物理計算のオーケストレーション）
# ============================================================

def _calculate_vacuum_pumping(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
) -> VacuumPumpingResult:
    """
    真空排気計算（物理計算の組み合わせ）
    
    純粋な物理計算を組み合わせて真空排気の結果を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
    
    Returns:
        VacuumPumpingResult: 真空排気計算結果
    """
    tower = state_manager.towers[tower_num]
    
    # === 1. 状態量の取得 ===
    T_K = _calculate_average_temperature(tower, tower_conds) + CELSIUS_TO_KELVIN_OFFSET
    P_Pa = tower.total_press * MPA_TO_PA
    avg_co2_mf, avg_n2_mf = _calculate_average_mole_fractions(tower, tower_conds)
    
    # === 2. ガス物性計算 ===
    viscosity = calculate_mixed_gas_viscosity(T_K, avg_co2_mf, avg_n2_mf)
    density = calculate_mixed_gas_density(T_K, P_Pa, avg_co2_mf, avg_n2_mf)
    
    # === 3. 配管流量・圧力損失計算 ===
    flow_result = calculate_vacuum_pump_flow(
        tower_conds=tower_conds,
        current_pressure=tower.total_press,
        T_K=T_K,
        viscosity=viscosity,
        density=density,
    )
    
    # === 4. 回収量計算 ===
    cumulative_co2, cumulative_n2 = calculate_desorption_amount(
        tower_conds=tower_conds,
        tower=tower,
        avg_co2_mole_fraction=avg_co2_mf,
        avg_n2_mole_fraction=avg_n2_mf,
    )
    
    co2_concentration = calculate_co2_recovery_concentration(cumulative_co2, cumulative_n2)
    
    # === 5. 排気後圧力計算 ===
    # 排気量 [mol]
    moles_pumped = flow_result.molar_flow_rate * tower_conds.common.calculation_step_time
    
    # 排気前の物質量 [mol]
    P_PUMP = max(0, (tower.total_press - flow_result.pressure_loss) * MPA_TO_PA)
    from common.constants import GAS_CONSTANT
    case_inner_mol_amt = (
        (P_PUMP + flow_result.pressure_loss * MPA_TO_PA)
        * tower_conds.vacuum_piping.space_volume
        / (GAS_CONSTANT * T_K)
    )
    
    # 排気後の物質量 [mol]
    remaining_moles = max(0, case_inner_mol_amt - moles_pumped)
    
    # 排気後の圧力 [MPaA]
    final_pressure = calculate_pressure_from_moles(
        total_moles=remaining_moles,
        T_K=T_K,
        volume=tower_conds.vacuum_piping.space_volume,
    )
    
    return VacuumPumpingResult(
        pressure_loss=flow_result.pressure_loss,
        cumulative_co2_recovered=cumulative_co2,
        cumulative_n2_recovered=cumulative_n2,
        co2_recovery_concentration=co2_concentration,
        volumetric_flow_rate=flow_result.volumetric_flow_rate,
        remaining_moles=remaining_moles,
        final_pressure=final_pressure,
    )
