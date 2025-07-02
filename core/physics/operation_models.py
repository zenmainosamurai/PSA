import core.physics.adsorption_base_models as adsorption_base_models
import warnings
from typing import Dict

from config.sim_conditions import TowerConditions
from core.state.state_variables import StateVariables
from core.physics.mass_balance_strategies import (
    MassBalanceStrategy,
    AdsorptionStrategy,
    DesorptionStrategy,
    ValveClosedStrategy,
)
from core.state.results import (
    MassAndHeatBalanceResults,
    MassBalanceResults,
    HeatBalanceResults,
    MoleFractionResults,
    WallHeatBalanceResult,
    DownstreamFlowResult,
    GasFlow,
)

warnings.simplefilter("ignore")


class CellCalculator:
    """セル計算の共通処理を提供するクラス"""

    @staticmethod
    def calculate_mass_and_heat_balance(
        tower_conds: TowerConditions,
        state_manager: StateVariables,
        tower_num: int,
        mode: int,
        mass_strategy: MassBalanceStrategy,
        vacuum_pumping_results=None,
    ) -> MassAndHeatBalanceResults:
        """マスバランスと熱バランスの計算を統一的に処理"""
        mass_balance_results = {}
        heat_balance_results = {}
        mole_fraction_results = {} if mass_strategy.supports_mole_fraction() else None
        num_streams = tower_conds.common.num_streams
        num_sections = tower_conds.common.num_sections
        for stream in range(1, 1 + num_streams):
            mass_balance_results[stream] = {}
            heat_balance_results[stream] = {}
            if mole_fraction_results is not None:
                mole_fraction_results[stream] = {}
            for section in range(1, 1 + num_sections):
                previous_material_result = mass_balance_results[stream].get(section - 1)
                mass_calculation_result = mass_strategy.calculate(stream, section, previous_material_result)
                mass_balance_results[stream][section] = mass_calculation_result.material_balance
                if mass_calculation_result.has_mole_fraction_data() and mole_fraction_results is not None:
                    mole_fraction_results[stream][section] = mass_calculation_result.mole_fraction_data

                heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                    tower_conds=tower_conds,
                    stream=stream,
                    section=section,
                    state_manager=state_manager,
                    tower_num=tower_num,
                    mode=mode,
                    material_output=mass_calculation_result.material_balance,  # 統一されたインターフェース
                    heat_output=heat_balance_results[stream].get(section - 1),
                    vacuum_pumping_results=vacuum_pumping_results,
                )

        mass_balance_data = MassBalanceResults(material_balance_results_dict=mass_balance_results)
        heat_balance_data = HeatBalanceResults(heat_balance_results_dict=heat_balance_results)
        mole_fraction_data = (
            MoleFractionResults(mole_fraction_results_dict=mole_fraction_results) if mole_fraction_results else None
        )

        return MassAndHeatBalanceResults(
            mass_balance_results=mass_balance_data,
            heat_balance_results=heat_balance_data,
            mole_fraction_results=mole_fraction_data,
        )

    @staticmethod
    def calculate_wall_heat_balance(
        tower_conds: TowerConditions,
        state_manager: StateVariables,
        tower_num: int,
        heat_balance_results: HeatBalanceResults,
    ):
        """壁面熱バランスの計算"""
        wall_heat_balance_results = {}

        wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
            tower_conds=tower_conds,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            heat_output=heat_balance_results.get_result(tower_conds.common.num_streams, 1),
            heat_wall_output=None,
        )

        num_sections = tower_conds.common.num_sections
        num_streams = tower_conds.common.num_streams
        for section in range(2, 1 + num_sections):
            wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
                tower_conds=tower_conds,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                heat_output=heat_balance_results.get_result(num_streams, section),
                heat_wall_output=wall_heat_balance_results[section - 1],
            )

        return wall_heat_balance_results

    @staticmethod
    def calculate_lid_heat_balance(
        tower_conds: TowerConditions,
        state_manager: StateVariables,
        tower_num: int,
        heat_balance_results: HeatBalanceResults,
        wall_heat_balance_results: Dict[int, WallHeatBalanceResult],
    ):
        """蓋熱バランスの計算"""
        lid_heat_balance_results = {}

        for position in ["up", "down"]:
            lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
                tower_conds=tower_conds,
                position=position,
                state_manager=state_manager,
                tower_num=tower_num,
                heat_output=heat_balance_results,
                heat_wall_output=wall_heat_balance_results,
            )

        return lid_heat_balance_results

    @staticmethod
    def distribute_inflow_gas(tower_conds: TowerConditions, inflow_gas: MassBalanceResults):
        """上流からの流入ガスを各ストリームに分配"""
        stream_conds = tower_conds.stream_conditions
        most_down_section = tower_conds.common.num_sections
        num_streams = tower_conds.common.num_streams

        total_outflow_co2 = sum(
            inflow_gas.get_result(stream, most_down_section).outlet_gas.co2_volume
            for stream in range(1, 1 + num_streams)
        )
        total_outflow_n2 = sum(
            inflow_gas.get_result(stream, most_down_section).outlet_gas.n2_volume
            for stream in range(1, 1 + num_streams)
        )

        distributed_inflows = {}
        for stream in range(1, 1 + num_streams):
            distributed_inflows[stream] = GasFlow(
                co2_volume=total_outflow_co2 * stream_conds[stream].area_fraction,
                n2_volume=total_outflow_n2 * stream_conds[stream].area_fraction,
                co2_mole_fraction=(total_outflow_co2 / (total_outflow_co2 + total_outflow_n2)),
                n2_mole_fraction=(total_outflow_n2 / (total_outflow_co2 + total_outflow_n2)),
            )

        return distributed_inflows


def initial_adsorption(tower_conds: TowerConditions, state_manager, tower_num, is_series_operation=False):
    """吸着開始時の圧力調整"""
    mode = 0  # 吸着
    calculator = CellCalculator()
    mass_strategy = AdsorptionStrategy(tower_conds, state_manager, tower_num)

    # マスバランス・熱バランス計算
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )

    # バッチ吸着後圧力変化
    pressure_after_batch_adsorption = adsorption_base_models.calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=is_series_operation,
    )

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "pressure_after_batch_adsorption": pressure_after_batch_adsorption,
    }


def stop_mode(tower_conds: TowerConditions, state_manager, tower_num):
    """停止モード"""
    mode = 1  # 弁停止モード
    calculator = CellCalculator()
    mass_strategy = ValveClosedStrategy(state_manager, tower_num)

    # マスバランス・熱バランス計算（弁閉鎖用の関数を使用）
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
    }


def flow_adsorption_single_or_upstream(tower_conds: TowerConditions, state_manager, tower_num):
    """流通吸着（単独/直列吸着の上流）"""
    mode = 0  # 吸着
    calculator = CellCalculator()
    mass_strategy = AdsorptionStrategy(tower_conds, state_manager, tower_num)

    # マスバランス・熱バランス計算
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )

    # 圧力計算
    pressure_after_flow_adsorption = tower_conds.feed_gas.total_pressure

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "total_pressure": pressure_after_flow_adsorption,
    }


def batch_adsorption_upstream(tower_conds: TowerConditions, state_manager, tower_num, is_series_operation):
    """バッチ吸着（上流）"""
    # 初期のバッチ吸着と同じ仕組み
    return initial_adsorption(tower_conds, state_manager, tower_num, is_series_operation)


def equalization_pressure_depressurization(
    tower_conds: TowerConditions, state_manager, tower_num, downstream_tower_pressure
):
    """バッチ均圧（減圧）"""
    mode = 0  # 吸着
    calculator = CellCalculator()

    # 上流管からの流入計算
    depressurization_results = adsorption_base_models.calculate_pressure_after_depressurization(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        downstream_tower_pressure=downstream_tower_pressure,
    )
    mass_strategy = AdsorptionStrategy(
        tower_conds, state_manager, tower_num, equalization_flow_rate=depressurization_results.flow_rate
    )

    # マスバランス・熱バランス計算（均圧配管流量を考慮）
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )

    # 下流塔の圧力と流入量計算
    downstream_flow_and_pressure = adsorption_base_models.calculate_downstream_flow_after_depressurization(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        mass_balance_results=balance_results.mass_balance_results,
        downstream_tower_pressure=downstream_tower_pressure,
    )

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "total_pressure": depressurization_results.final_pressure,
        "diff_press": depressurization_results.pressure_differential,
        "downflow_params": downstream_flow_and_pressure,
    }


def desorption_by_vacuuming(tower_conds: TowerConditions, state_manager, tower_num):
    """真空脱着"""
    mode = 2  # 脱着
    calculator = CellCalculator()

    # 排気後圧力の計算
    vacuum_pumping_results = adsorption_base_models.calculate_pressure_after_vacuum_pumping(
        tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
    )
    mass_strategy = DesorptionStrategy(tower_conds, state_manager, tower_num, vacuum_pumping_results)

    # マスバランス・熱バランス計算（脱着用）
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
        vacuum_pumping_results=vacuum_pumping_results,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )

    # 脱着後の全圧
    pressure_after_desorption = adsorption_base_models.calculate_pressure_after_desorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        mole_fraction_results=(
            balance_results.mole_fraction_results if balance_results.mole_fraction_results else None
        ),
        vacuum_pumping_results=vacuum_pumping_results,
    )

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "mol_fraction": (balance_results.mole_fraction_results if balance_results.mole_fraction_results else None),
        "accum_vacuum_amt": vacuum_pumping_results,
        "pressure_after_desorption": pressure_after_desorption,
    }


def equalization_pressure_pressurization(
    tower_conds: TowerConditions, state_manager: StateVariables, tower_num: int, upstream_params: DownstreamFlowResult
):
    """バッチ均圧（加圧）"""
    mode = 0  # 吸着
    calculator = CellCalculator()

    # 上流側で計算した流出量を各セクションの流入ガスとして設定
    inflow_gas_dict = {}
    num_streams = tower_conds.common.num_streams
    for stream in range(1, 1 + num_streams):
        inflow_gas_dict[stream] = upstream_params.outlet_flows[stream]

    mass_strategy = AdsorptionStrategy(tower_conds, state_manager, tower_num, external_inflow_gas=inflow_gas_dict)
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
    )
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )
    pressure_after_batch_adsorption = upstream_params.final_pressure

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "pressure_after_batch_adsorption": pressure_after_batch_adsorption,
    }


def batch_adsorption_downstream(
    tower_conds: TowerConditions, state_manager, tower_num, is_series_operation, inflow_gas, residual_gas_composition
):
    """バッチ吸着（下流）"""
    mode = 0  # 吸着
    calculator = CellCalculator()

    # 流入ガスを各ストリームに分配
    distributed_inflows = calculator.distribute_inflow_gas(tower_conds, inflow_gas)

    mass_strategy = AdsorptionStrategy(
        tower_conds,
        state_manager,
        tower_num,
        external_inflow_gas=distributed_inflows,
        residual_gas_composition=residual_gas_composition,
    )
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )

    # バッチ吸着後圧力変化
    pressure_after_batch_adsorption = adsorption_base_models.calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=is_series_operation,
    )

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "pressure_after_batch_adsorption": pressure_after_batch_adsorption,
    }


def flow_adsorption_downstream(tower_conds: TowerConditions, state_manager, tower_num, inflow_gas):
    """流通吸着（下流）"""
    mode = 0  # 吸着
    calculator = CellCalculator()

    # 流入ガスを各ストリームに分配
    distributed_inflows = calculator.distribute_inflow_gas(tower_conds, inflow_gas)

    mass_strategy = AdsorptionStrategy(
        tower_conds,
        state_manager,
        tower_num,
        external_inflow_gas=distributed_inflows,
    )
    balance_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_strategy=mass_strategy,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, balance_results.heat_balance_results, wall_heat_balance_results
    )

    # 全圧
    pressure_after_flow_adsorption = tower_conds.feed_gas.total_pressure

    return {
        "material": balance_results.mass_balance_results,
        "heat": balance_results.heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "total_pressure": pressure_after_flow_adsorption,
    }
