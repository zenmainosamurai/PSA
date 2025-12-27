"""運転モード共通処理

PSA担当者向け説明:
各運転モードで共通して使用する計算処理を提供します。

主要な関数:
- calculate_all_cells(): 全セルの物質・熱収支を計算
- calculate_wall_heat(): 壁面の熱収支を計算
- calculate_lid_heat(): 蓋の熱収支を計算
- distribute_inflow_gas(): 上流からの流入ガスを各ストリームに分配
"""

from typing import Dict, Optional
from dataclasses import dataclass

from operation_modes.mode_types import OperationMode
from config.sim_conditions import TowerConditions
from state import (
    StateVariables,
    MassBalanceResults,
    HeatBalanceResults,
    MoleFractionResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    VacuumPumpingResult,
    GasFlow,
)

# Phase2で作成した物理計算モジュールをインポート
from physics.mass_balance import calculate_mass_balance
from physics.heat_balance import (
    calculate_bed_heat_balance,
    calculate_wall_heat_balance,
    calculate_lid_heat_balance,
)

# 熱収支計算のモード定数
MODE_ADSORPTION = 0
MODE_VALVE_CLOSED = 1
MODE_DESORPTION = 2


@dataclass
class CellCalculationResults:
    """全セル計算結果
    
    PSA担当者向け説明:
    全ストリーム×全セクションの物質収支・熱収支計算結果をまとめたものです。
    """
    mass_balance: MassBalanceResults
    heat_balance: HeatBalanceResults
    mole_fraction: Optional[MoleFractionResults] = None


@dataclass
class FullTowerResults:
    """塔全体の計算結果
    
    PSA担当者向け説明:
    1つの塔の全計算結果（セル・壁・蓋）をまとめたものです。
    """
    mass_balance: MassBalanceResults
    heat_balance: HeatBalanceResults
    wall_heat: Dict[int, WallHeatBalanceResult]
    lid_heat: Dict[str, LidHeatBalanceResult]
    mole_fraction: Optional[MoleFractionResults] = None


def calculate_all_cells(
    mode: OperationMode,
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    external_inflow_gas: Optional[Dict[int, GasFlow]] = None,
    equalization_flow_rate: Optional[float] = None,
    residual_gas_composition: Optional[MassBalanceResults] = None,
    vacuum_pumping_results: Optional[VacuumPumpingResult] = None,
) -> CellCalculationResults:
    """
    全セルの物質収支・熱収支を計算
    
    PSA担当者向け説明:
    塔内の全セル（ストリーム×セクション）について、
    物質収支と熱収支を順番に計算します。
    
    計算順序:
    1. ストリーム1のセクション1→2→...→N
    2. ストリーム2のセクション1→2→...→N
    3. ...
    
    Args:
        mode: 運転モード
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        external_inflow_gas: 外部からの流入ガス（下流塔の場合）
        equalization_flow_rate: 均圧配管流量（均圧モードの場合）
        residual_gas_composition: 残留ガス組成（バッチ吸着下流の場合）
        vacuum_pumping_results: 真空排気結果（脱着モードの場合）
    
    Returns:
        CellCalculationResults: 全セルの計算結果
    """
    num_streams = tower_conds.common.num_streams
    num_sections = tower_conds.common.num_sections
    
    # 熱収支計算用のモード番号に変換
    heat_mode = _get_heat_mode(mode)
    
    # 結果格納用
    mass_balance_results: Dict[int, Dict] = {}
    heat_balance_results: Dict[int, Dict] = {}
    mole_fraction_results: Dict[int, Dict] = {} if mode == OperationMode.VACUUM_DESORPTION else None
    
    # 全セルを計算
    for stream in range(1, 1 + num_streams):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        if mole_fraction_results is not None:
            mole_fraction_results[stream] = {}
        
        for section in range(1, 1 + num_sections):
            # 流入ガスの決定
            inflow_gas = _get_inflow_gas(
                stream, section, external_inflow_gas, mass_balance_results
            )
            
            # 前セクションの計算結果
            previous_result = mass_balance_results[stream].get(section - 1)
            
            # 物質収支計算
            mass_result = calculate_mass_balance(
                mode=mode,
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                inflow_gas=inflow_gas,
                equalization_flow_rate=equalization_flow_rate if section == 1 else None,
                residual_gas_composition=residual_gas_composition,
                vacuum_pumping_results=vacuum_pumping_results,
                previous_result=previous_result,
            )
            
            mass_balance_results[stream][section] = mass_result.material_balance
            
            # モル分率データ（脱着モードのみ）
            if mole_fraction_results is not None and mass_result.mole_fraction_data is not None:
                mole_fraction_results[stream][section] = mass_result.mole_fraction_data
            
            # 熱収支計算
            heat_result = calculate_bed_heat_balance(
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                mode=heat_mode,
                material_output=mass_result.material_balance,
                heat_output=heat_balance_results[stream].get(section - 1),
                vacuum_pumping_results=vacuum_pumping_results,
            )
            
            heat_balance_results[stream][section] = heat_result
    
    # 結果オブジェクトの構築
    return CellCalculationResults(
        mass_balance=MassBalanceResults(material_balance_results_dict=mass_balance_results),
        heat_balance=HeatBalanceResults(heat_balance_results_dict=heat_balance_results),
        mole_fraction=(
            MoleFractionResults(mole_fraction_results_dict=mole_fraction_results)
            if mole_fraction_results else None
        ),
    )


def calculate_wall_heat(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    heat_balance_results: HeatBalanceResults,
) -> Dict[int, WallHeatBalanceResult]:
    """
    壁面の熱収支を計算
    
    PSA担当者向け説明:
    容器壁の各セクションの温度変化を計算します。
    セクション1から順に計算し、上流セクションの結果を使用します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        heat_balance_results: 全セルの熱収支結果
    
    Returns:
        Dict[int, WallHeatBalanceResult]: セクション番号 -> 壁面熱収支結果
    """
    num_sections = tower_conds.common.num_sections
    num_streams = tower_conds.common.num_streams
    
    wall_results: Dict[int, WallHeatBalanceResult] = {}
    
    # セクション1
    wall_results[1] = calculate_wall_heat_balance(
        tower_conds=tower_conds,
        section=1,
        state_manager=state_manager,
        tower_num=tower_num,
        heat_output=heat_balance_results.get_result(num_streams, 1),
        heat_wall_output=None,
    )
    
    # セクション2以降
    for section in range(2, 1 + num_sections):
        wall_results[section] = calculate_wall_heat_balance(
            tower_conds=tower_conds,
            section=section,
            state_manager=state_manager,
            tower_num=tower_num,
            heat_output=heat_balance_results.get_result(num_streams, section),
            heat_wall_output=wall_results[section - 1],
        )
    
    return wall_results


def calculate_lid_heat(
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    heat_balance_results: HeatBalanceResults,
    wall_heat_results: Dict[int, WallHeatBalanceResult],
) -> Dict[str, LidHeatBalanceResult]:
    """
    蓋の熱収支を計算
    
    PSA担当者向け説明:
    上蓋・下蓋の温度変化を計算します。
    
    Args:
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        heat_balance_results: 全セルの熱収支結果
        wall_heat_results: 壁面の熱収支結果
    
    Returns:
        Dict[str, LidHeatBalanceResult]: "up" or "down" -> 蓋熱収支結果
    """
    lid_results: Dict[str, LidHeatBalanceResult] = {}
    
    for position in ["up", "down"]:
        lid_results[position] = calculate_lid_heat_balance(
            tower_conds=tower_conds,
            position=position,
            state_manager=state_manager,
            tower_num=tower_num,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_results,
        )
    
    return lid_results


def calculate_full_tower(
    mode: OperationMode,
    tower_conds: TowerConditions,
    state_manager: StateVariables,
    tower_num: int,
    external_inflow_gas: Optional[Dict[int, GasFlow]] = None,
    equalization_flow_rate: Optional[float] = None,
    residual_gas_composition: Optional[MassBalanceResults] = None,
    vacuum_pumping_results: Optional[VacuumPumpingResult] = None,
) -> FullTowerResults:
    """
    塔全体（セル・壁・蓋）の計算を実行
    
    PSA担当者向け説明:
    1つの塔の全計算（物質収支、熱収支、壁、蓋）を一括で実行します。
    各運転モードで共通して使用する処理です。
    
    Args:
        mode: 運転モード
        tower_conds: 塔条件
        state_manager: 状態変数管理
        tower_num: 塔番号
        external_inflow_gas: 外部からの流入ガス
        equalization_flow_rate: 均圧配管流量
        residual_gas_composition: 残留ガス組成
        vacuum_pumping_results: 真空排気結果
    
    Returns:
        FullTowerResults: 塔全体の計算結果
    """
    # 全セルの計算
    cell_results = calculate_all_cells(
        mode=mode,
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        external_inflow_gas=external_inflow_gas,
        equalization_flow_rate=equalization_flow_rate,
        residual_gas_composition=residual_gas_composition,
        vacuum_pumping_results=vacuum_pumping_results,
    )
    
    # 壁面の計算
    wall_results = calculate_wall_heat(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        heat_balance_results=cell_results.heat_balance,
    )
    
    # 蓋の計算
    lid_results = calculate_lid_heat(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        heat_balance_results=cell_results.heat_balance,
        wall_heat_results=wall_results,
    )
    
    return FullTowerResults(
        mass_balance=cell_results.mass_balance,
        heat_balance=cell_results.heat_balance,
        wall_heat=wall_results,
        lid_heat=lid_results,
        mole_fraction=cell_results.mole_fraction,
    )


def distribute_inflow_gas(
    tower_conds: TowerConditions,
    inflow_gas: MassBalanceResults,
) -> Dict[int, GasFlow]:
    """
    上流からの流入ガスを各ストリームに分配
    
    PSA担当者向け説明:
    上流塔から流出したガスを、面積比率に応じて各ストリームに分配します。
    下流塔での吸着計算で使用します。
    
    Args:
        tower_conds: 塔条件
        inflow_gas: 上流塔の物質収支結果
    
    Returns:
        Dict[int, GasFlow]: ストリーム番号 -> 流入ガス
    """
    stream_conds = tower_conds.stream_conditions
    most_down_section = tower_conds.common.num_sections
    num_streams = tower_conds.common.num_streams
    
    # 最下流セクションからの流出量合計
    total_outflow_co2 = sum(
        inflow_gas.get_result(stream, most_down_section).outlet_gas.co2_volume
        for stream in range(1, 1 + num_streams)
    )
    total_outflow_n2 = sum(
        inflow_gas.get_result(stream, most_down_section).outlet_gas.n2_volume
        for stream in range(1, 1 + num_streams)
    )
    
    # 各ストリームに分配
    distributed: Dict[int, GasFlow] = {}
    total_outflow = total_outflow_co2 + total_outflow_n2
    
    for stream in range(1, 1 + num_streams):
        distributed[stream] = GasFlow(
            co2_volume=total_outflow_co2 * stream_conds[stream].area_fraction,
            n2_volume=total_outflow_n2 * stream_conds[stream].area_fraction,
            co2_mole_fraction=(total_outflow_co2 / total_outflow) if total_outflow > 0 else 0,
            n2_mole_fraction=(total_outflow_n2 / total_outflow) if total_outflow > 0 else 0,
        )
    
    return distributed


# ============================================================
# ヘルパー関数
# ============================================================

def _get_heat_mode(mode: OperationMode) -> int:
    """運転モードを熱収支計算用のモード番号に変換"""
    if mode == OperationMode.STOP:
        return MODE_VALVE_CLOSED
    elif mode == OperationMode.VACUUM_DESORPTION:
        return MODE_DESORPTION
    else:
        return MODE_ADSORPTION


def _get_inflow_gas(
    stream: int,
    section: int,
    external_inflow_gas: Optional[Dict[int, GasFlow]],
    mass_balance_results: Dict[int, Dict],
) -> Optional[GasFlow]:
    """セルへの流入ガスを決定"""
    if section == 1:
        # 最上流セクション: 外部流入ガスを使用（あれば）
        if external_inflow_gas is not None:
            return external_inflow_gas.get(stream)
        return None
    else:
        # それ以外: 上流セクションの流出ガスを使用
        previous_result = mass_balance_results[stream].get(section - 1)
        if previous_result is not None:
            return GasFlow(
                co2_volume=previous_result.outlet_gas.co2_volume,
                n2_volume=previous_result.outlet_gas.n2_volume,
                co2_mole_fraction=previous_result.outlet_gas.co2_mole_fraction,
                n2_mole_fraction=previous_result.outlet_gas.n2_mole_fraction,
            )
        return None
