import numpy as np
import pandas as pd

import adsorption_base_models

import warnings

warnings.simplefilter("ignore")


def initial_adsorption(sim_conds, stream_conds, variables, is_series_operation=False):
    """吸着開始時の圧力調整
        説明: 規定圧力に達するまでガスを導入する（吸着も起こる）
        ベースモデル: バッチ吸着モデル
        終了条件: 圧力到達

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        variables (dict): 状態変数
        is_series_operation (bool): 直列か単独か

    Returns:
        output (dict): 各モデルの計算結果
    """
    mass_balance_results = {}
    heat_balance_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 0  # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        # sec_1は手動で実施
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            inflow_gas=None,
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                inflow_gas=mass_balance_results[stream][section - 1],
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 4. バッチ吸着後圧力変化
    pressure_after_batch_adsorption = adsorption_base_models.calculate_pressure_after_batch_adsorption(
        sim_conds=sim_conds, variables=variables, is_series_operation=is_series_operation
    )
    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
        "pressure_after_batch_adsorption": pressure_after_batch_adsorption,  # バッチ吸着後圧力
    }

    return output


def stop_mode(sim_conds, stream_conds, variables):
    """停止
        説明: モードとモードの間の吸着停止期間
        ベースモデル: 脱着モデル
        終了条件: 時間経過
        補足: 真空脱着との違いは排気後圧力計算で排気速度が0である点のみ

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mass_balance_results = {}
    heat_balance_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 1  # 弁停止モード

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        # sec_1は手動で実施
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_valve_closed(
            stream=stream, section=1, variables=variables
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_valve_closed(
                stream=stream, section=section, variables=variables
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
    }

    return output


def flow_adsorption_single_or_upstream(sim_conds, stream_conds, variables):
    """流通吸着_単独/直列吸着_上流
        説明: 1つの吸着塔の流通吸着 or 2つの吸着塔の直列吸着における上流
        ベースモデル: 流通吸着モデル
        終了条件: 時間経過・温度上昇

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mass_balance_results = {}
    heat_balance_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 0  # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        # sec_1は手動で実施
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            inflow_gas=None,
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                inflow_gas=mass_balance_results[stream][section - 1],
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 4. 圧力計算
    pressure_after_flow_adsorption = sim_conds["INFLOW_GAS_COND"]["total_press"]

    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
        "total_press": pressure_after_flow_adsorption,  # 圧力
    }

    return output


def batch_adsorption_upstream(sim_conds, stream_conds, variables, is_series_operation):
    """バッチ吸着_上流
        説明: ２つの吸着塔のバッチ吸着における上流側（下流側の圧力回復が目的）
        ベースモデル: バッチ吸着モデル
        終了条件: 下流の温度上昇

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        timestamp (float): 時刻
    """
    # 初期のバッチ吸着と同じ仕組み
    return initial_adsorption(sim_conds, stream_conds, variables, is_series_operation)


def equalization_pressure_depressurization(sim_conds, stream_conds, variables, downstream_tower_pressure):
    """バッチ均圧（減圧）
        説明: 均圧における減圧側
        ベースモデル: 吸着モデル
        終了条件: 圧力到達・時間経過
        補足１: 基本は「流通吸着_上流」と同じ
        補足２: 上流配管からの流入ガスがあるので、sec1は特有の流入あり
        補足３: 下流塔への流出ガス計算も追加で必要（下流塔の圧力計算も実施）

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻
    """
    mass_balance_results = {}
    heat_balance_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 0  # 吸着

    ### セル計算 --------------------------------------------------------------

    # 0. 上流管からの流入計算
    depressurization_results = adsorption_base_models.calculate_pressure_after_depressurization(
        sim_conds=sim_conds,
        variables=variables,
        downstream_tower_pressure=downstream_tower_pressure,
    )
    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        # sec_1は手動で実施
        # NOTE: 均圧配管流量を引数として与える
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            inflow_gas=None,
            flow_amt_depress=depressurization_results["flow_amount_l"],
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                inflow_gas=mass_balance_results[stream][section - 1],
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 4. 下流塔の圧力と流入量計算
    downstream_flow_and_pressure = adsorption_base_models.calculate_downstream_flow_after_depressurization(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        variables=variables,
        mass_balance_results=mass_balance_results,
        downstream_tower_pressure=downstream_tower_pressure,
    )

    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
        "total_press": depressurization_results["total_press_after_depressure"],  # 減圧後の全圧
        "diff_press": depressurization_results["diff_press"],
        "downflow_params": downstream_flow_and_pressure,  # 下流塔の全圧と流入量
    }

    return output


def desorption_by_vacuuming(sim_conds, stream_conds, variables):
    """真空脱着
        説明: 真空引きによる脱着
        ベースモデル: 脱着モデル
        終了条件: 時間経過
        補足: 停止モードとほぼ同じだが、排気速度が0ではない点のみ異なる

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻
    """
    mass_balance_results = {}
    heat_balance_results = {}
    mole_fraction_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 2  # 脱着

    ### セル計算 --------------------------------------------------------------

    # 0. 排気後圧力の計算
    vacuum_pumping_results = adsorption_base_models.calculate_pressure_after_vacuum_pumping(
        sim_conds=sim_conds, variables=variables
    )
    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        mole_fraction_results[stream] = {}
        # sec_1は手動で実施
        mass_balance_results[stream][1], mole_fraction_results[stream][1] = (
            adsorption_base_models.calculate_mass_balance_for_desorption(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=1,
                variables=variables,
                vacuum_pumping_results=vacuum_pumping_results,
            )
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=vacuum_pumping_results,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section], mole_fraction_results[stream][section] = (
                adsorption_base_models.calculate_mass_balance_for_desorption(
                    sim_conds=sim_conds,
                    stream_conds=stream_conds,
                    stream=stream,
                    section=section,
                    variables=variables,
                    vacuum_pumping_results=vacuum_pumping_results,
                )
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=vacuum_pumping_results,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 4. 脱着後の全圧
    pressure_after_desorption = adsorption_base_models.calculate_pressure_after_desorption(
        sim_conds=sim_conds,
        variables=variables,
        mole_fraction_results=mole_fraction_results,
        vacuum_pumping_results=vacuum_pumping_results,
    )
    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
        "mol_fraction": mole_fraction_results,  # モル分率
        "accum_vacuum_amt": vacuum_pumping_results,  # 積算CO2, N2回収量
        "pressure_after_desorption": pressure_after_desorption,  # 脱着後の全圧
    }

    return output


def equalization_pressure_pressurization(sim_conds, stream_conds, variables, upstream_params):
    """バッチ均圧（加圧）
        説明: 均圧における加圧側
        ベースモデル: バッチ吸着モデル
        終了条件: 圧力到達・時間経過
        補足１: バッチ吸着とほぼ同じだが、最上流セルの流入量に上流塔の出力値をそのまま使用
        補足２: また、同じく上流塔で計算した次時刻全圧値をそのまま使用

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mass_balance_results = {}
    heat_balance_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 0  # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        # sec_1は手動で実施
        # NOTE: 上流側で計算した流出量を引数に与える
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            inflow_gas=upstream_params["outflow_fr"][stream],
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                inflow_gas=mass_balance_results[stream][section - 1],
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 4. バッチ吸着後圧力変化
    pressure_after_batch_adsorption = upstream_params["total_press_after_depressure_downflow"]
    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
        "pressure_after_batch_adsorption": pressure_after_batch_adsorption,  # バッチ吸着後圧力
    }

    return output


def batch_adsorption_downstream(sim_conds, stream_conds, variables, is_series_operation, inflow_gas, stagnant_mf):
    """バッチ吸着（下流）
        説明: ２つの吸着塔のバッチ吸着における下流側（下流側の圧力回復が目的）
        ベースモデル: バッチ吸着モデル
        終了条件: 下流の温度上昇
        補足: 通常のバッチ吸着とほぼ同じだが、section1の流入ガスが上流の流出ガスになっている点のみ異なる

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        variables (dict): 状態変数
        is_series_operation (bool): 直列か単独か

    Returns:
        output (dict): 各モデルの計算結果
    """
    mass_balance_results = {}
    heat_balance_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 0  # 吸着

    most_down_section = sim_conds["COMMON_COND"]["NUM_SEC"]
    total_outflow_co2 = sum(
        [
            inflow_gas[stream][most_down_section]["outflow_fr_co2"]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"])
        ]
    )
    total_outflow_n2 = sum(
        [
            inflow_gas[stream][most_down_section]["outflow_fr_n2"]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"])
        ]
    )

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        distributed_inflow = {
            "outflow_fr_co2": total_outflow_co2 * stream_conds[stream]["streamratio"],
            "outflow_fr_n2": total_outflow_n2 * stream_conds[stream]["streamratio"],
        }
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        # sec_1は手動で実施
        # NOTE: # 流入ガスが上流の流出ガス
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            inflow_gas=distributed_inflow,
            stagnant_mode=stagnant_mf,
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                inflow_gas=mass_balance_results[stream][section - 1],
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 4. バッチ吸着後圧力変化
    pressure_after_batch_adsorption = adsorption_base_models.calculate_pressure_after_batch_adsorption(
        sim_conds=sim_conds, variables=variables, is_series_operation=is_series_operation
    )
    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
        "pressure_after_batch_adsorption": pressure_after_batch_adsorption,  # バッチ吸着後圧力
    }

    return output


def flow_adsorption_downstream(sim_conds, stream_conds, variables, inflow_gas):
    """流通吸着（下流）
        説明: ２つの吸着塔による流通吸着における下流側
        ベースモデル: 流通吸着モデル
        終了条件: 上流塔の温度上昇、時間経過
        補足: 通常の流通吸着とほぼ同じだが、section1の流入ガスが上流の流出ガスになっている点のみ異なる

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mass_balance_results = {}
    heat_balance_results = {}
    wall_heat_balance_results = {}
    lid_heat_balance_results = {}
    mode = 0  # 吸着

    most_down_section = sim_conds["COMMON_COND"]["NUM_SEC"]
    total_outflow_co2 = sum(
        [
            inflow_gas[stream][most_down_section]["outflow_fr_co2"]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"])
        ]
    )
    total_outflow_n2 = sum(
        [
            inflow_gas[stream][most_down_section]["outflow_fr_n2"]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"])
        ]
    )

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["NUM_STR"]):
        distributed_inflow = {
            "outflow_fr_co2": total_outflow_co2 * stream_conds[stream]["streamratio"],
            "outflow_fr_n2": total_outflow_n2 * stream_conds[stream]["streamratio"],
        }
        mass_balance_results[stream] = {}
        heat_balance_results[stream] = {}
        # sec_1は手動で実施
        # NOTE: # 流入ガスが上流の流出ガス
        mass_balance_results[stream][1] = adsorption_base_models.calculate_mass_balance_for_adsorption(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            inflow_gas=distributed_inflow,
        )
        heat_balance_results[stream][1] = adsorption_base_models.calculate_heat_balance_for_bed(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            stream=stream,
            section=1,
            variables=variables,
            mode=mode,
            material_output=mass_balance_results[stream][1],
            heat_output=None,
            vacuum_pumping_results=None,
        )
        # sec_2以降は自動で実施
        for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
            mass_balance_results[stream][section] = adsorption_base_models.calculate_mass_balance_for_adsorption(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                inflow_gas=mass_balance_results[stream][section - 1],
            )
            heat_balance_results[stream][section] = adsorption_base_models.calculate_heat_balance_for_bed(
                sim_conds=sim_conds,
                stream_conds=stream_conds,
                stream=stream,
                section=section,
                variables=variables,
                mode=mode,
                material_output=mass_balance_results[stream][section],
                heat_output=heat_balance_results[stream][section - 1],
                vacuum_pumping_results=None,
            )
    # 2. 壁面熱バラ（stream = 1+sim_conds["COMMON_COND"]["NUM_STR"]）
    wall_heat_balance_results[1] = adsorption_base_models.calculate_heat_balance_for_wall(
        sim_conds=sim_conds,
        stream_conds=stream_conds,
        section=1,
        variables=variables,
        heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][1],
        heat_wall_output=None,
    )
    for section in range(2, 1 + sim_conds["COMMON_COND"]["NUM_SEC"]):
        wall_heat_balance_results[section] = adsorption_base_models.calculate_heat_balance_for_wall(
            sim_conds=sim_conds,
            stream_conds=stream_conds,
            section=section,
            variables=variables,
            heat_output=heat_balance_results[sim_conds["COMMON_COND"]["NUM_STR"]][section],
            heat_wall_output=wall_heat_balance_results[section - 1],
        )
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        lid_heat_balance_results[position] = adsorption_base_models.calculate_heat_balance_for_lid(
            sim_conds=sim_conds,
            position=position,
            variables=variables,
            heat_output=heat_balance_results,
            heat_wall_output=wall_heat_balance_results,
        )
    # 4. 全圧
    pressure_after_flow_adsorption = sim_conds["INFLOW_GAS_COND"]["total_press"]

    # 出力
    output = {
        "material": mass_balance_results,  # マテバラ
        "heat": heat_balance_results,  # 熱バラ
        "heat_wall": wall_heat_balance_results,  # 熱バラ（壁面）
        "heat_lid": lid_heat_balance_results,  # 熱バラ（蓋）
        "total_press": pressure_after_flow_adsorption,  # 全圧
    }

    return output
