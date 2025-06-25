import numpy as np
import pandas as pd
import adsorption_base_models
import warnings

from sim_conditions import TowerConditions

warnings.simplefilter("ignore")


class CellCalculator:
    """セル計算の共通処理を提供するクラス"""

    @staticmethod
    def calculate_mass_and_heat_balance(
        tower_conds: TowerConditions,
        state_manager,
        tower_num,
        mode,
        mass_balance_func,
        heat_balance_func,
        inflow_gas_sec1=None,
        flow_amt_depress=None,
        stagnant_mode=None,
        vacuum_pumping_results=None,
        is_valve_closed=False,
    ):
        """マスバランスと熱バランスの計算を統一的に処理"""
        mass_balance_results = {}
        heat_balance_results = {}
        mole_fraction_results = {} if mode == 2 else None  # 脱着モードの場合のみ

        num_streams = tower_conds.common.num_streams
        for stream in range(1, 1 + num_streams):
            mass_balance_results[stream] = {}
            heat_balance_results[stream] = {}
            if mole_fraction_results is not None:
                mole_fraction_results[stream] = {}

            # Section 1の計算
            if is_valve_closed:
                # 弁閉鎖モードの場合は特別な引数セット
                result = mass_balance_func(stream=stream, section=1, state_manager=state_manager, tower_num=tower_num)
            else:
                # 通常のマスバランス計算
                kwargs = {
                    "tower_conds": tower_conds,
                    "stream": stream,
                    "section": 1,
                    "state_manager": state_manager,
                    "tower_num": tower_num,
                }

                # オプショナルパラメータの追加
                if mass_balance_func.__name__ == "calculate_mass_balance_for_adsorption":
                    kwargs["inflow_gas"] = inflow_gas_sec1
                    if flow_amt_depress is not None:
                        kwargs["flow_amt_depress"] = flow_amt_depress
                    if stagnant_mode is not None:
                        kwargs["stagnant_mode"] = stagnant_mode
                elif mass_balance_func.__name__ == "calculate_mass_balance_for_desorption":
                    kwargs["vacuum_pumping_results"] = vacuum_pumping_results

                result = mass_balance_func(**kwargs)

            # 脱着モードの場合はタプルで返される
            if isinstance(result, tuple):
                mass_balance_results[stream][1], mole_fraction_results[stream][1] = result
            else:
                mass_balance_results[stream][1] = result

            heat_balance_results[stream][1] = heat_balance_func(
                tower_conds=tower_conds,
                stream=stream,
                section=1,
                state_manager=state_manager,
                tower_num=tower_num,
                mode=mode,
                material_output=mass_balance_results[stream][1],
                heat_output=None,
                vacuum_pumping_results=vacuum_pumping_results,
            )

            # Section 2以降の計算
            num_sections = tower_conds.common.num_sections
            for section in range(2, 1 + num_sections):
                if is_valve_closed:
                    # 弁閉鎖モードの場合
                    result = mass_balance_func(
                        stream=stream, section=section, state_manager=state_manager, tower_num=tower_num
                    )
                else:
                    # 通常のマスバランス計算
                    kwargs = {
                        "tower_conds": tower_conds,
                        "stream": stream,
                        "section": section,
                        "state_manager": state_manager,
                        "tower_num": tower_num,
                    }

                    if mass_balance_func.__name__ == "calculate_mass_balance_for_adsorption":
                        kwargs["inflow_gas"] = mass_balance_results[stream][section - 1]
                    elif mass_balance_func.__name__ == "calculate_mass_balance_for_desorption":
                        kwargs["vacuum_pumping_results"] = vacuum_pumping_results

                    result = mass_balance_func(**kwargs)

                if isinstance(result, tuple):
                    mass_balance_results[stream][section], mole_fraction_results[stream][section] = result
                else:
                    mass_balance_results[stream][section] = result

                heat_balance_results[stream][section] = heat_balance_func(
                    tower_conds=tower_conds,
                    stream=stream,
                    section=section,
                    state_manager=state_manager,
                    tower_num=tower_num,
                    mode=mode,
                    material_output=mass_balance_results[stream][section],
                    heat_output=heat_balance_results[stream][section - 1],
                    vacuum_pumping_results=vacuum_pumping_results,
                )

        return mass_balance_results, heat_balance_results, mole_fraction_results

    @staticmethod
    def calculate_wall_heat_balance(tower_conds: TowerConditions, state_manager, tower_num, heat_balance_results):
        """壁面熱バランスの計算"""
        wall_heat_balance_results = {}

        wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
            tower_conds=tower_conds,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            heat_output=heat_balance_results[tower_conds.common.num_streams][1],
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
                heat_output=heat_balance_results[num_streams][section],
                heat_wall_output=wall_heat_balance_results[section - 1],
            )

        return wall_heat_balance_results

    @staticmethod
    def calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
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
    def distribute_inflow_gas(tower_conds: TowerConditions, inflow_gas):
        """上流からの流入ガスを各ストリームに分配"""
        stream_conds = tower_conds.stream_conditions
        most_down_section = tower_conds.common.num_sections
        num_streams = tower_conds.common.num_streams

        total_outflow_co2 = sum(
            inflow_gas[stream][most_down_section]["outlet_co2_volume"] for stream in range(1, 1 + num_streams)
        )
        total_outflow_n2 = sum(
            inflow_gas[stream][most_down_section]["outlet_n2_volume"] for stream in range(1, 1 + num_streams)
        )

        distributed_inflows = {}
        for stream in range(1, 1 + num_streams):
            distributed_inflows[stream] = {
                "outlet_co2_volume": total_outflow_co2 * stream_conds[stream].area_fraction,
                "outlet_n2_volume": total_outflow_n2 * stream_conds[stream].area_fraction,
            }

        return distributed_inflows


def initial_adsorption(tower_conds: TowerConditions, state_manager, tower_num, is_series_operation=False):
    """吸着開始時の圧力調整"""
    mode = 0  # 吸着
    calculator = CellCalculator()

    # マスバランス・熱バランス計算
    mass_balance_results, heat_balance_results, _ = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_balance_func=adsorption_base_models.calculate_mass_balance_for_adsorption,
        heat_balance_func=adsorption_base_models.calculate_heat_balance_for_bed,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    # バッチ吸着後圧力変化
    pressure_after_batch_adsorption = adsorption_base_models.calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=is_series_operation,
    )

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "pressure_after_batch_adsorption": pressure_after_batch_adsorption,
    }


def stop_mode(tower_conds: TowerConditions, state_manager, tower_num):
    """停止モード"""
    mode = 1  # 弁停止モード
    calculator = CellCalculator()

    # マスバランス・熱バランス計算（弁閉鎖用の関数を使用）
    mass_balance_results, heat_balance_results, _ = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_balance_func=adsorption_base_models.calculate_mass_balance_for_valve_closed,
        heat_balance_func=adsorption_base_models.calculate_heat_balance_for_bed,
        is_valve_closed=True,  # 弁閉鎖モードフラグを追加
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
    }


def flow_adsorption_single_or_upstream(tower_conds: TowerConditions, state_manager, tower_num):
    """流通吸着（単独/直列吸着の上流）"""
    mode = 0  # 吸着
    calculator = CellCalculator()

    # マスバランス・熱バランス計算
    mass_balance_results, heat_balance_results, _ = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_balance_func=adsorption_base_models.calculate_mass_balance_for_adsorption,
        heat_balance_func=adsorption_base_models.calculate_heat_balance_for_bed,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    # 圧力計算
    pressure_after_flow_adsorption = tower_conds.feed_gas.total_pressure

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
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

    # マスバランス・熱バランス計算（均圧配管流量を考慮）
    mass_balance_results, heat_balance_results, _ = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_balance_func=adsorption_base_models.calculate_mass_balance_for_adsorption,
        heat_balance_func=adsorption_base_models.calculate_heat_balance_for_bed,
        flow_amt_depress=depressurization_results["flow_amount_l"],
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    # 下流塔の圧力と流入量計算
    downstream_flow_and_pressure = adsorption_base_models.calculate_downstream_flow_after_depressurization(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        mass_balance_results=mass_balance_results,
        downstream_tower_pressure=downstream_tower_pressure,
    )

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "total_pressure": depressurization_results["total_press_after_depressure"],
        "diff_press": depressurization_results["diff_press"],
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

    # マスバランス・熱バランス計算（脱着用）
    mass_balance_results, heat_balance_results, mole_fraction_results = calculator.calculate_mass_and_heat_balance(
        tower_conds,
        state_manager,
        tower_num,
        mode,
        mass_balance_func=adsorption_base_models.calculate_mass_balance_for_desorption,
        heat_balance_func=adsorption_base_models.calculate_heat_balance_for_bed,
        vacuum_pumping_results=vacuum_pumping_results,
    )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    # 脱着後の全圧
    pressure_after_desorption = adsorption_base_models.calculate_pressure_after_desorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        mole_fraction_results=mole_fraction_results,
        vacuum_pumping_results=vacuum_pumping_results,
    )

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "mol_fraction": mole_fraction_results,
        "accum_vacuum_amt": vacuum_pumping_results,
        "pressure_after_desorption": pressure_after_desorption,
    }


def equalization_pressure_pressurization(tower_conds: TowerConditions, state_manager, tower_num, upstream_params):
    """バッチ均圧（加圧）"""
    mode = 0  # 吸着
    calculator = CellCalculator()

    # 上流側で計算した流出量を各セクションの流入ガスとして設定
    inflow_gas_dict = {}
    num_streams = tower_conds.common.num_streams
    for stream in range(1, 1 + num_streams):
        inflow_gas_dict[stream] = upstream_params["outflow_fr"][stream]

    # マスバランス・熱バランス計算（特殊な流入ガス処理）
    mass_balance_results = {}
    heat_balance_results = {}

    for stream in range(1, 1 + num_streams):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}

        # Section 1（上流からの流入）
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            tower_conds=tower_conds,
            stream=stream,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            inflow_gas=inflow_gas_dict[stream],
        )

        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            tower_conds=tower_conds,
            stream=stream,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )

        # Section 2以降
        num_sections = tower_conds.common.num_sections
        for section in range(2, 1 + num_sections):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                inflow_gas=mass_balance_results[stream][section - 1],
            )

            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    # バッチ吸着後圧力変化（上流側で計算済みの値を使用）
    pressure_after_batch_adsorption = upstream_params["total_press_after_depressure_downflow"]

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
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

    # マスバランス・熱バランス計算（分配された流入ガスを使用）
    mass_balance_results = {}
    heat_balance_results = {}

    num_streams = tower_conds.common.num_streams
    for stream in range(1, 1 + num_streams):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}

        # Section 1（上流からの流入）
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            tower_conds=tower_conds,
            stream=stream,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            inflow_gas=distributed_inflows[stream],
            residual_gas_composition=residual_gas_composition,
        )

        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            tower_conds=tower_conds,
            stream=stream,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )

        # Section 2以降
        num_sections = tower_conds.common.num_sections
        for section in range(2, 1 + num_sections):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                inflow_gas=mass_balance_results[stream][section - 1],
            )

            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    # バッチ吸着後圧力変化
    pressure_after_batch_adsorption = adsorption_base_models.calculate_pressure_after_batch_adsorption(
        tower_conds=tower_conds,
        state_manager=state_manager,
        tower_num=tower_num,
        is_series_operation=is_series_operation,
    )

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
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

    # マスバランス・熱バランス計算（分配された流入ガスを使用）
    mass_balance_results = {}
    heat_balance_results = {}

    num_streams = tower_conds.common.num_streams
    for stream in range(1, 1 + num_streams):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}

        # Section 1（上流からの流入）
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            tower_conds=tower_conds,
            stream=stream,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            inflow_gas=distributed_inflows[stream],
        )

        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            tower_conds=tower_conds,
            stream=stream,
            section=1,
            state_manager=state_manager,
            tower_num=tower_num,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )

        # Section 2以降
        num_sections = tower_conds.common.num_sections
        for section in range(2, 1 + num_sections):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                inflow_gas=mass_balance_results[stream][section - 1],
            )

            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                tower_conds=tower_conds,
                stream=stream,
                section=section,
                state_manager=state_manager,
                tower_num=tower_num,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )

    # 壁面熱バランス計算
    wall_heat_balance_results = calculator.calculate_wall_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results
    )

    # 蓋熱バランス計算
    lid_heat_balance_results = calculator.calculate_lid_heat_balance(
        tower_conds, state_manager, tower_num, heat_balance_results, wall_heat_balance_results
    )

    # 全圧
    pressure_after_flow_adsorption = tower_conds.feed_gas.total_pressure

    return {
        "material": mass_balance_results,
        "heat": heat_balance_results,
        "heat_wall": wall_heat_balance_results,
        "heat_lid": lid_heat_balance_results,
        "total_pressure": pressure_after_flow_adsorption,
    }
