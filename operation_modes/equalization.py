"""均圧モード

PSA担当者向け説明:
2つの塔間でガスを移動させて圧力を均一化するモードの計算を行います。

均圧とは:
- 高圧塔と低圧塔を配管で接続
- ガスが高圧側から低圧側へ流れる
- 最終的に両塔の圧力が近づく
- エネルギー回収の効果がある

運転パターン:
- 減圧側: 圧力が下がる塔（ガスが流出）
- 加圧側: 圧力が上がる塔（ガスが流入）

稼働工程表での対応:
- 「均圧_減圧」
- 「均圧_加圧」
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
    GasFlow,
)
from physics.pressure import (
    calculate_depressurization,
    calculate_downstream_flow,
    calculate_pressure_after_batch_adsorption,
)


@dataclass
class EqualizationDepressurizationResult:
    """
    均圧減圧モードの計算結果
    
    PSA担当者向け説明:
    均圧減圧では塔から流出するガス量と、
    減圧後の最終圧力を計算します。
    """
    material: MassBalanceResults
    heat: HeatBalanceResults
    heat_wall: Dict[int, WallHeatBalanceResult]
    heat_lid: Dict[str, LidHeatBalanceResult]
    final_total_pressure: float  # 減圧後の圧力 [MPaA]
    pressure_difference: float  # 圧力差 [MPa]
    downstream_flow_co2: float  # 下流への流出CO2量 [m3]
    downstream_flow_n2: float  # 下流への流出N2量 [m3]
    
    @property
    def total_pressure(self) -> float:
        """互換性のための別名（state_variables.update_from_calc_outputで使用）"""
        return self.final_total_pressure


@dataclass
class EqualizationPressurizationResult:
    """
    均圧加圧モードの計算結果
    
    PSA担当者向け説明:
    均圧加圧では上流塔からのガス流入による圧力上昇と、
    それに伴う吸着量変化を計算します。
    """
    material: MassBalanceResults
    heat: HeatBalanceResults
    heat_wall: Dict[int, WallHeatBalanceResult]
    heat_lid: Dict[str, LidHeatBalanceResult]
    pressure_after_batch_adsorption: float  # 加圧後の圧力 [MPaA]


def execute_equalization_depressurization(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    target_tower_pressure: float,
) -> EqualizationDepressurizationResult:
    """
    均圧減圧の計算を実行
    
    PSA担当者向け説明:
    均圧減圧では、以下の順序で計算を行います:
    1. 減圧に伴うガス流出量の計算
    2. 流出に伴う物質収支・熱収支の計算
    3. 最終圧力の計算
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        target_tower_pressure: 目標圧力（均圧相手塔の圧力）[MPaA]
    
    Returns:
        EqualizationDepressurizationResult: 均圧減圧の計算結果
    
    使用例:
        result = execute_equalization_depressurization(
            tower_conds, state_manager, tower_num=1,
            target_tower_pressure=0.15  # 均圧相手塔の圧力
        )
        # 流出ガス量を確認
        print(f"CO2流出量: {result.downstream_flow_co2} m3")
    """
    # 現在の圧力を取得
    current_pressure = state_manager.towers[tower_num].total_press
    
    # 減圧結果の計算（圧力差と均圧流量）
    depressurization_result = calculate_depressurization(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        downstream_tower_pressure=target_tower_pressure,  # target_tower_pressureをdownstream_tower_pressureとして渡す
    )
    
    equalization_flow_rate = depressurization_result.flow_rate
    
    # 塔全体の計算を実行（均圧流量を使用）
    tower_results = calculate_full_tower(
        mode=OperationMode.EQUALIZATION_DEPRESSURIZATION,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        equalization_flow_rate=equalization_flow_rate,
    )
    
    # 下流への流出ガス量計算
    downstream_flow_result = calculate_downstream_flow(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        mass_balance_results=tower_results.mass_balance,
        downstream_tower_pressure=target_tower_pressure,
    )
    
    # 全ストリームの流出量を合計
    total_co2_flow = sum(
        flow.co2_volume for flow in downstream_flow_result.outlet_flows.values()
    )
    total_n2_flow = sum(
        flow.n2_volume for flow in downstream_flow_result.outlet_flows.values()
    )
    
    # 圧力差
    pressure_difference = current_pressure - depressurization_result.final_pressure
    
    return EqualizationDepressurizationResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        final_total_pressure=depressurization_result.final_pressure,
        pressure_difference=pressure_difference,
        downstream_flow_co2=total_co2_flow,
        downstream_flow_n2=total_n2_flow,
    )


def execute_equalization_pressurization(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    upstream_depressurization_result: EqualizationDepressurizationResult,
) -> EqualizationPressurizationResult:
    """
    均圧加圧の計算を実行
    
    PSA担当者向け説明:
    均圧加圧では、減圧側塔から流出したガスを受け取り、
    圧力を上昇させます。加圧に伴い吸着も進行します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        upstream_depressurization_result: 減圧側塔の計算結果
    
    Returns:
        EqualizationPressurizationResult: 均圧加圧の計算結果
    
    使用例:
        # 減圧側の計算結果を使用
        result = execute_equalization_pressurization(
            tower_conds, state_manager, tower_num=2,
            upstream_depressurization_result=depressurization_result
        )
    """
    # 減圧側からの流入ガスを作成
    inflow_gas = _create_equalization_inflow(upstream_depressurization_result)
    
    # 外部流入ガスとして分配
    distributed_inflows = _distribute_equalization_inflow(
        tower_conds, inflow_gas
    )
    
    # 塔全体の計算を実行
    tower_results = calculate_full_tower(
        mode=OperationMode.EQUALIZATION_PRESSURIZATION,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        external_inflow_gas=distributed_inflows,
    )
    
    # 加圧後の圧力計算
    pressure_after = calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=False,
        has_pressure_valve=False,
    )
    
    return EqualizationPressurizationResult(
        material=tower_results.mass_balance,
        heat=tower_results.heat_balance,
        heat_wall=tower_results.wall_heat,
        heat_lid=tower_results.lid_heat,
        pressure_after_batch_adsorption=pressure_after,
    )


# ============================================================
# ヘルパー関数
# ============================================================

def _create_equalization_inflow(
    depressurization_result: EqualizationDepressurizationResult,
) -> GasFlow:
    """減圧側の流出ガスから流入ガスを作成"""
    co2_vol = depressurization_result.downstream_flow_co2
    n2_vol = depressurization_result.downstream_flow_n2
    total_vol = co2_vol + n2_vol
    
    return GasFlow(
        co2_volume=co2_vol,
        n2_volume=n2_vol,
        co2_mole_fraction=co2_vol / total_vol if total_vol > 0 else 0,
        n2_mole_fraction=n2_vol / total_vol if total_vol > 0 else 0,
    )


def _distribute_equalization_inflow(
    tower_conds: TowerConditions,
    inflow_gas: GasFlow,
) -> Dict[int, GasFlow]:
    """均圧流入ガスを各ストリームに分配"""
    stream_conds = tower_conds.stream_conditions
    num_streams = tower_conds.common.num_streams
    
    distributed: Dict[int, GasFlow] = {}
    
    for stream in range(1, 1 + num_streams):
        area_frac = stream_conds[stream].area_fraction
        distributed[stream] = GasFlow(
            co2_volume=inflow_gas.co2_volume * area_frac,
            n2_volume=inflow_gas.n2_volume * area_frac,
            co2_mole_fraction=inflow_gas.co2_mole_fraction,
            n2_mole_fraction=inflow_gas.n2_mole_fraction,
        )
    
    return distributed
