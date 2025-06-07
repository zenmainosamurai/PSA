import os
import sys
import datetime
import yaml
import math
import copy
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import CoolProp.CoolProp as CP

from utils import const, init_functions, plot_csv, other_utils
from utils.heat_transfer import calc_heat_transfer_coef as _heat_transfer_coef
from state_variables import StateVariables


import warnings

warnings.simplefilter("error")


def calculate_mass_balance_for_adsorption(
    sim_conds,
    stream_conds,
    stream,
    section,
    state_manager: StateVariables,
    tower_num,
    inflow_gas=None,
    flow_amt_depress=None,
    stagnant_mode=None,
):
    """任意セルのマテリアルバランスを計算する
        吸着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        inflow_gas (dict): 上部セルの出力値
        flow_amt_depress (float): 上流均圧菅からの全流量

    Returns:
        dict: 対象セルの計算結果
    """
    ### マテバラ計算開始 ------------------------------------------------
    tower = state_manager.towers[tower_num]

    # セクション吸着材量 [g]
    Mabs = stream_conds[stream]["adsorbent_mass"] / sim_conds["COMMON_COND"]["num_sections"]
    # 流入CO2, N2流量 [cm3]
    if (section == 1) & (inflow_gas is None) & (flow_amt_depress is None):  # 最上流セルの流入ガスによる吸着など
        inflow_fr_co2 = (
            sim_conds["FEED_GAS_COND"]["co2_flow_rate"]
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * stream_conds[stream]["area_fraction"]
            * 1000
        )
        inflow_fr_n2 = (
            sim_conds["FEED_GAS_COND"]["n2_flow_rate"]
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * stream_conds[stream]["area_fraction"]
            * 1000
        )
    elif (section == 1) & (flow_amt_depress is not None):  # 減圧時の最上流セルのみ対象
        inflow_fr_co2 = (
            sim_conds["FEED_GAS_COND"]["co2_mole_fraction"]
            * flow_amt_depress
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * stream_conds[stream]["area_fraction"]
            * 1000
        )
        inflow_fr_n2 = (
            sim_conds["FEED_GAS_COND"]["n2_mole_fraction"]
            * flow_amt_depress
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * stream_conds[stream]["area_fraction"]
            * 1000
        )
    elif inflow_gas is not None:  # 下流セクションや下流塔での吸着など
        inflow_fr_co2 = inflow_gas["outflow_fr_co2"]
        inflow_fr_n2 = inflow_gas["outflow_fr_n2"]
    # 流入ガスモル分率
    if (stagnant_mode is None) | (section != 1):
        inflow_mf_co2 = inflow_fr_co2 / (inflow_fr_co2 + inflow_fr_n2)
        inflow_mf_n2 = inflow_fr_n2 / (inflow_fr_co2 + inflow_fr_n2)
    else:
        inflow_mf_co2 = stagnant_mode[stream][1]["inflow_mf_co2"]
        inflow_mf_n2 = stagnant_mode[stream][1]["inflow_mf_n2"]
    # 全圧 [MPaA]
    total_press = tower.total_press
    # CO2分圧 [MPaA]
    p_co2 = total_press * inflow_mf_co2
    # 現在温度 [℃]
    temp = tower.temp[stream - 1, section - 1]
    # ガス密度 [kg/m3]
    gas_density = (
        sim_conds["FEED_GAS_COND"]["co2_density"] * inflow_mf_co2
        + sim_conds["FEED_GAS_COND"]["n2_density"] * inflow_mf_n2
    )
    # ガス比熱 [kJ/kg/K]
    gas_cp = (
        sim_conds["FEED_GAS_COND"]["co2_specific_heat_capacity"] * inflow_mf_co2
        + sim_conds["FEED_GAS_COND"]["n2_specific_heat_capacity"] * inflow_mf_n2
    )
    # 現在雰囲気の平衡吸着量 [cm3/g-abs]
    P_KPA = p_co2 * 1000  # [kPaA]
    T_K = temp + 273.15  # [K]
    adsorp_amt_equilibrium = max(0.1, _calculate_equilibrium_adsorption_amount(P_KPA, T_K))
    # 現在の既存吸着量 [cm3/g-abs]
    adsorp_amt_current = tower.adsorp_amt[stream - 1, section - 1]
    # 理論新規吸着量 [cm3/g-abs]
    if adsorp_amt_equilibrium >= adsorp_amt_current:
        adsorp_amt_estimate_abs = (
            sim_conds["PACKED_BED_COND"]["adsorption_mass_transfer_coeff"]
            ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / sim_conds["PACKED_BED_COND"]["adsorbent_bulk_density"]
            * 6
            * (1 - sim_conds["PACKED_BED_COND"]["average_porosity"])
            * sim_conds["PACKED_BED_COND"]["particle_shape_factor"]
            / sim_conds["PACKED_BED_COND"]["average_particle_diameter"]
            * (adsorp_amt_equilibrium - adsorp_amt_current)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
        # 実際のセクション新規吸着量 [cm3]
        adsorp_amt_estimate = min(adsorp_amt_estimate, inflow_fr_co2)
    else:
        adsorp_amt_estimate_abs = (
            sim_conds["PACKED_BED_COND"]["desorption_mass_transfer_coeff"]
            ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / sim_conds["PACKED_BED_COND"]["adsorbent_bulk_density"]
            * 6
            * (1 - sim_conds["PACKED_BED_COND"]["average_porosity"])
            * sim_conds["PACKED_BED_COND"]["particle_shape_factor"]
            / sim_conds["PACKED_BED_COND"]["average_particle_diameter"]
            * (adsorp_amt_equilibrium - adsorp_amt_current)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
        # 実際のセクション新規吸着量 [cm3]
        adsorp_amt_estimate = max(adsorp_amt_estimate, -adsorp_amt_current)
    # 実際の新規吸着量 [cm3/g-abs]
    adsorp_amt_estimate_abs = adsorp_amt_estimate / Mabs
    # 時間経過後吸着量 [cm3/g-abs]
    accum_adsorp_amt = adsorp_amt_current + adsorp_amt_estimate_abs
    # 下流流出CO2流量 [cm3]
    outflow_fr_co2 = inflow_fr_co2 - adsorp_amt_estimate
    # 下流流出N2流量 [cm3]
    outflow_fr_n2 = inflow_fr_n2
    # 流出CO2分率
    outflow_mf_co2 = outflow_fr_co2 / (outflow_fr_co2 + outflow_fr_n2)
    # 流出N2分率
    outflow_mf_n2 = outflow_fr_n2 / (outflow_fr_co2 + outflow_fr_n2)

    ### 流出CO2分圧のつじつま合わせ ------------------------------

    # 流出CO2分圧 [MPaA]
    outflow_pco2 = total_press * outflow_mf_co2
    # 直前値より低い場合は直前値と同じになるように吸着量を変更
    previous_outflow_pco2 = tower.outflow_pco2[stream - 1, section - 1]
    if p_co2 >= previous_outflow_pco2:
        if outflow_pco2 < previous_outflow_pco2:
            # 直前値に置換
            outflow_pco2 = previous_outflow_pco2
            # セクション新規吸着量とそれに伴う各変数を逆算
            adsorp_amt_estimate = inflow_fr_co2 - outflow_pco2 * inflow_fr_n2 / (total_press - outflow_pco2)
            # 実際の新規吸着量 [cm3/g-abs]
            adsorp_amt_estimate_abs = adsorp_amt_estimate / Mabs
            # 時間経過後吸着量 [cm3/g-abs]
            accum_adsorp_amt = adsorp_amt_current + adsorp_amt_estimate_abs
            # 下流流出CO2流量 [cm3]
            outflow_fr_co2 = inflow_fr_co2 - adsorp_amt_estimate
            # 下流流出N2流量 [cm3]
            outflow_fr_n2 = inflow_fr_n2
            # 流出CO2分率
            outflow_mf_co2 = outflow_fr_co2 / (outflow_fr_co2 + outflow_fr_n2)
            # 流出N2分率
            outflow_mf_n2 = outflow_fr_n2 / (outflow_fr_co2 + outflow_fr_n2)
    else:
        if outflow_pco2 < p_co2:
            # co2分圧に置換
            outflow_pco2 = p_co2
            # セクション新規吸着量とそれに伴う各変数を逆算
            adsorp_amt_estimate = inflow_fr_co2 - outflow_pco2 * inflow_fr_n2 / (total_press - outflow_pco2)
            # 実際の新規吸着量 [cm3/g-abs]
            adsorp_amt_estimate_abs = adsorp_amt_estimate / Mabs
            # 時間経過後吸着量 [cm3/g-abs]
            accum_adsorp_amt = adsorp_amt_current + adsorp_amt_estimate_abs
            # 下流流出CO2流量 [cm3]
            outflow_fr_co2 = inflow_fr_co2 - adsorp_amt_estimate
            # 下流流出N2流量 [cm3]
            outflow_fr_n2 = inflow_fr_n2
            # 流出CO2分率
            outflow_mf_co2 = outflow_fr_co2 / (outflow_fr_co2 + outflow_fr_n2)
            # 流出N2分率
            outflow_mf_n2 = outflow_fr_n2 / (outflow_fr_co2 + outflow_fr_n2)

    output = {
        "inflow_fr_co2": inflow_fr_co2,
        "inflow_fr_n2": inflow_fr_n2,
        "inflow_mf_co2": inflow_mf_co2,
        "inflow_mf_n2": inflow_mf_n2,
        "gas_density": gas_density,
        "gas_cp": gas_cp,
        "adsorp_amt_equilibrium": adsorp_amt_equilibrium,  # 平衡吸着量
        "adsorp_amt_estimate": adsorp_amt_estimate,
        "accum_adsorp_amt": accum_adsorp_amt,
        "outflow_fr_co2": outflow_fr_co2,
        "outflow_fr_n2": outflow_fr_n2,
        "outflow_mf_co2": outflow_mf_co2,
        "outflow_mf_n2": outflow_mf_n2,
        "adsorp_amt_estimate_abs": adsorp_amt_estimate_abs,
        "p_co2": p_co2,
        "outflow_pco2": outflow_pco2,
    }

    return output


def calculate_mass_balance_for_desorption(
    sim_conds,
    stream_conds,
    stream,
    section,
    state_manager: StateVariables,
    tower_num: int,
    vacuum_pumping_results,
):
    """任意セルのマテリアルバランスを計算する
        脱着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vacuum_pumping_results (dict): 排気後圧力計算の出力

    Returns:
        dict: 対象セルの計算結果
    """
    ### 現在気相モル量 = 流入モル量の計算 -------------------
    tower = state_manager.towers[tower_num]
    # セクション空間割合
    space_ratio_section = (
        stream_conds[stream]["area_fraction"]
        / sim_conds["COMMON_COND"]["num_sections"]
        * sim_conds["PACKED_BED_COND"]["void_volume"]
        / sim_conds["VACUUM_PIPING_COND"]["space_volume"]
    )
    # セクション空間現在物質量 [mol]
    mol_amt_section = vacuum_pumping_results["case_inner_mol_amt_after_vacuum"] * space_ratio_section
    # 現在気相モル量 [mol]
    inflow_fr_co2 = mol_amt_section * tower.mf_co2[stream - 1, section - 1]
    inflow_fr_n2 = mol_amt_section * tower.mf_n2[stream - 1, section - 1]
    # 現在気相ノルマル体積(=流入量) [cm3]
    inflow_fr_co2 *= 22.4 * 1000
    inflow_fr_n2 *= 22.4 * 1000

    ### 気相放出後モル量の計算 -----------------------------
    # 現在温度 [℃]
    T_K = tower.temp[stream - 1, section - 1] + 273.15
    # モル分率
    mf_co2 = tower.mf_co2[stream - 1, section - 1]
    mf_n2 = tower.mf_n2[stream - 1, section - 1]
    # CO2分圧 [MPaA]
    p_co2 = max(2.5e-3, vacuum_pumping_results["total_press_after_vacuum"] * mf_co2)
    # セクション吸着材量 [g]
    Mabs = stream_conds[stream]["adsorbent_mass"] / sim_conds["COMMON_COND"]["num_sections"]
    # 現在雰囲気の平衡吸着量 [cm3/g-abs]
    P_KPA = p_co2 * 1000  # [MPaA] → [kPaA]
    adsorp_amt_equilibrium = max(0.1, _calculate_equilibrium_adsorption_amount(P_KPA, T_K))
    # 現在の既存吸着量 [cm3/g-abs]
    adsorp_amt_current = tower.adsorp_amt[stream - 1, section - 1]
    # 理論新規吸着量 [cm3/g-abs]
    if adsorp_amt_equilibrium >= adsorp_amt_current:
        adsorp_amt_estimate_abs = (
            sim_conds["PACKED_BED_COND"]["adsorption_mass_transfer_coeff"]
            ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / sim_conds["PACKED_BED_COND"]["adsorbent_bulk_density"]
            * 6
            * (1 - sim_conds["PACKED_BED_COND"]["average_porosity"])
            * sim_conds["PACKED_BED_COND"]["particle_shape_factor"]
            / sim_conds["PACKED_BED_COND"]["average_particle_diameter"]
            * (adsorp_amt_equilibrium - adsorp_amt_current)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
        # 実際のセクション新規吸着量 [cm3]
        adsorp_amt_estimate = min(adsorp_amt_estimate, inflow_fr_co2)
    else:
        adsorp_amt_estimate_abs = (
            sim_conds["PACKED_BED_COND"]["desorption_mass_transfer_coeff"]
            ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / sim_conds["PACKED_BED_COND"]["adsorbent_bulk_density"]
            * 6
            * (1 - sim_conds["PACKED_BED_COND"]["average_porosity"])
            * sim_conds["PACKED_BED_COND"]["particle_shape_factor"]
            / sim_conds["PACKED_BED_COND"]["average_particle_diameter"]
            * (adsorp_amt_equilibrium - adsorp_amt_current)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            / 1e6
            * 60
        )
        # セクション理論新規吸着量 [cm3]
        adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
        # 実際のセクション新規吸着量 [cm3]
        adsorp_amt_estimate = max(adsorp_amt_estimate, -adsorp_amt_current)
    # 実際の新規吸着量 [cm3/g-abs]
    adsorp_amt_estimate_abs = adsorp_amt_estimate / Mabs
    # 時間経過後吸着量 [cm3/g-abs]
    accum_adsorp_amt = adsorp_amt_current + adsorp_amt_estimate_abs
    # 気相放出CO2量 [cm3]
    desorp_mw_co2 = -1 * adsorp_amt_estimate
    # 気相放出CO2量 [mol]
    desorp_mw_co2 = desorp_mw_co2 / 1000 / 22.4
    # 気相放出後モル量 [mol]
    desorp_mw_co2_after_vacuum = inflow_fr_co2 + desorp_mw_co2
    desorp_mw_n2_after_vacuum = inflow_fr_n2
    # 気相放出後全物質量 [mol]
    desorp_mw_all_after_vacuum = desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum
    # 気相放出後モル分率
    desorp_mf_co2_after_vacuum = desorp_mw_co2_after_vacuum / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)
    desorp_mf_n2_after_vacuum = desorp_mw_n2_after_vacuum / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)

    ### その他（熱バラ渡す用） ---------------------------------------
    P = vacuum_pumping_results["total_press_after_vacuum"] * 1e6
    P_ATM = 0.101325 * 1e6
    # ガス密度 [kg/m3]
    gas_density = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * mf_co2 + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * mf_n2
    )
    # ガス比熱 [kJ/kg/K]
    # NOTE: 比熱と熱伝導率と粘度は大気圧を使用
    gas_cp = (
        CP.PropsSI("CPMASS", "T", T_K, "P", P_ATM, "co2") * mf_co2
        + CP.PropsSI("CPMASS", "T", T_K, "P", P_ATM, "nitrogen") * mf_n2
    ) * 1e-3

    # 出力
    output = {
        "inflow_fr_co2": 0,
        "inflow_fr_n2": 0,
        "inflow_mf_co2": 0,
        "inflow_mf_n2": 0,
        "gas_density": gas_density,
        "gas_cp": gas_cp,
        "adsorp_amt_equilibrium": adsorp_amt_equilibrium,  # 平衡吸着量
        "adsorp_amt_estimate": adsorp_amt_estimate,
        "accum_adsorp_amt": accum_adsorp_amt,
        "outflow_fr_co2": 0,
        "outflow_fr_n2": 0,
        "outflow_mf_co2": 0,
        "outflow_mf_n2": 0,
        "adsorp_amt_estimate_abs": adsorp_amt_estimate_abs,
        "p_co2": p_co2,
        "outflow_pco2": tower.outflow_pco2[stream - 1, section - 1],
    }

    # 出力２（モル分率）
    output2 = {
        "mf_co2_after_vacuum": desorp_mf_co2_after_vacuum,  # 気相放出後CO2モル分率
        "mf_n2_after_vacuum": desorp_mf_n2_after_vacuum,  # 気相放出後N2モル分率
        "desorp_mw_all_after_vacuum": desorp_mw_all_after_vacuum,  # 気相放出後物質量 [mol]
    }

    return output, output2


def calculate_mass_balance_for_valve_closed(stream, section, state_manager: StateVariables, tower_num: int):
    """任意セルのマテリアルバランスを計算する
        弁停止モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vacuum_pumping_results (dict): 排気後圧力計算の出力

    Returns:
        dict: 対象セルの計算結果
    """
    tower = state_manager.towers[tower_num]
    # 何の反応もしない
    output = {
        "inflow_fr_co2": 0,
        "inflow_fr_n2": 0,
        "inflow_mf_co2": 0,
        "inflow_mf_n2": 0,
        "gas_density": 0,
        "gas_cp": 0,
        "adsorp_amt_equilibrium": 0,
        "adsorp_amt_estimate": 0,
        "accum_adsorp_amt": tower.adsorp_amt[stream - 1, section - 1],
        "outflow_fr_co2": 0,
        "outflow_fr_n2": 0,
        "outflow_mf_co2": 0,
        "outflow_mf_n2": 0,
        "adsorp_amt_estimate_abs": 0,
        "p_co2": 0,
        "outflow_pco2": tower.outflow_pco2[stream - 1, section - 1],
    }

    return output


def calculate_heat_balance_for_bed(
    sim_conds,
    stream_conds,
    stream,
    section,
    state_manager: StateVariables,
    tower_num,
    mode,
    material_output=None,
    heat_output=None,
    vacuum_pumping_results=None,
):
    """対象セルの熱バランスを計算する

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 状態変数
        mode (int): 吸着・脱着等の運転モード
        material_output (dict): 対象セルのマテバラ出力
        heat_output (dict): 上流セクションの熱バラ出力
        vacuum_pumping_results (dict): 排気後圧力計算の出力（脱着時）

    Returns:
        dict: 対象セルの熱バラ出力
    """
    ### 前準備 ------------------------------------------------------
    tower = state_manager.towers[tower_num]

    # セクション現在温度 [℃]
    temp_now = tower.temp[stream - 1, section - 1]
    # 内側セクション温度 [℃]
    if stream == 1:
        temp_inside_cell = 18
    else:
        temp_inside_cell = tower.temp[stream - 2, section - 1]
    # 外側セクション温度 [℃]
    if stream != sim_conds["COMMON_COND"]["num_streams"]:
        temp_outside_cell = tower.temp[stream, section - 1]
    else:
        temp_outside_cell = tower.temp_wall[section - 1]
    # 下流セクション温度 [℃]
    if section != sim_conds["COMMON_COND"]["num_sections"]:
        temp_below_cell = tower.temp[stream - 1, section]
    # 発生する吸着熱 [J]
    if mode == 1:
        Habs = 0  # 弁停止モードでは0
    else:
        Habs = (
            material_output["adsorp_amt_estimate"]
            / 1000
            / 22.4
            * sim_conds["FEED_GAS_COND"]["co2_molecular_weight"]
            * sim_conds["FEED_GAS_COND"]["co2_adsorption_heat"]
        )
    # 流入ガス質量 [g]
    if mode in [1, 2]:
        Mgas = 0  # 弁停止・脱着モードでは0
    else:
        Mgas = (
            material_output["inflow_fr_co2"] / 1000 / 22.4 * sim_conds["FEED_GAS_COND"]["co2_molecular_weight"]
            + material_output["inflow_fr_n2"] / 1000 / 22.4 * sim_conds["FEED_GAS_COND"]["n2_molecular_weight"]
        )
    # 流入ガス比熱 [J/g/K]
    if mode == 1:
        gas_cp = 0  # 弁停止モードでは0
    else:
        gas_cp = material_output["gas_cp"]
    # 内側境界面積 [m2]
    Ain = stream_conds[stream]["inner_boundary_area"] / sim_conds["COMMON_COND"]["num_sections"]
    # 外側境界面積 [m2]
    Aout = stream_conds[stream]["outer_boundary_area"] / sim_conds["COMMON_COND"]["num_sections"]
    # 下流セル境界面積 [m2]
    Abb = stream_conds[stream]["cross_section"]
    # 壁-層伝熱係数、層伝熱係数
    if mode == 0:
        hw1, u1 = _heat_transfer_coef(
            sim_conds,
            stream_conds,
            stream,
            section,
            temp_now,
            mode,
            state_manager,
            tower_num,
            material_output,
        )
    elif mode == 1:  # 弁停止モードでは直前値に置換
        hw1 = tower.heat_t_coef[stream - 1, section - 1]
        u1 = tower.heat_t_coef_wall[stream - 1, section - 1]
    elif mode == 2:  # 脱着モードでは入力に排気後圧力計算の出力を使用
        # hw1 = variables["heat_t_coef"][stream][section]
        # u1 = variables["heat_t_coef_wall"][stream][section]
        hw1, u1 = _heat_transfer_coef(
            sim_conds,
            stream_conds,
            stream,
            section,
            temp_now,
            mode,
            state_manager,
            tower_num,
            material_output,
            vacuum_pumping_results,
        )

    ### 熱流束計算 ---------------------------------------------------

    # 内側境界からの熱流束 [J]
    if stream == 1:
        Hwin = 0
    else:
        Hwin = u1 * Ain * (temp_inside_cell - temp_now) * sim_conds["COMMON_COND"]["calculation_step_time"] * 60
    # 外側境界への熱流束 [J]
    if stream == sim_conds["COMMON_COND"]["num_streams"]:
        Hwout = hw1 * Aout * (temp_now - temp_outside_cell) * sim_conds["COMMON_COND"]["calculation_step_time"] * 60
    else:
        Hwout = u1 * Aout * (temp_now - temp_outside_cell) * sim_conds["COMMON_COND"]["calculation_step_time"] * 60
    # 下流セルへの熱流束 [J]
    if section == sim_conds["COMMON_COND"]["num_sections"]:
        Hbb = (  # 下蓋への熱流束
            hw1
            * stream_conds[stream]["cross_section"]
            * (temp_now - tower.temp_lid_down)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * 60
        )
    else:
        Hbb = (
            u1 * Abb * (temp_now - temp_below_cell) * sim_conds["COMMON_COND"]["calculation_step_time"] * 60
        )  # 下流セルへの熱流束
    # 上流セルヘの熱流束 [J]
    if section == 1:  # 上蓋からの熱流束
        Hroof = (
            hw1
            * stream_conds[stream]["cross_section"]
            * (temp_now - tower.temp_lid_up)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * 60
        )
    else:  # 上流セルヘの熱流束 = -1 * 上流セルの「下流セルへの熱流束」
        Hroof = -heat_output["Hbb"]

    ### 到達温度計算 --------------------------------------------------------------

    # セクション到達温度 [℃]
    args = {
        "sim_conds": sim_conds,
        "stream_conds": stream_conds,
        "gas_cp": gas_cp,
        "Mgas": Mgas,
        "temp_now": temp_now,
        "Habs": Habs,
        "Hwin": Hwin,
        "Hwout": Hwout,
        "Hbb": Hbb,
        "Hroof": Hroof,
        "stream": stream,
    }
    temp_reached = optimize.newton(_optimize_bed_temperature, temp_now, args=args.values())

    ### 熱電対温度の計算 --------------------------------------------------------------

    # 熱電対熱容量 [J/K]
    heat_capacity = sim_conds["THERMOCOUPLE_COND"]["specific_heat"] * sim_conds["THERMOCOUPLE_COND"]["weight"]
    # 熱電対側面積 [m2]
    S_side = 0.004 * np.pi * 0.1
    # 熱電対伝熱係数 [W/m2/K]
    heat_transfer = hw1
    # 熱電対熱流束 [W]
    if mode != 2:
        heat_flux = (
            heat_transfer
            * sim_conds["THERMOCOUPLE_COND"]["heat_transfer_correction_factor"]
            * S_side
            * (tower.temp[stream - 1, section - 1] - tower.temp_thermo[stream - 1, section - 1])
        )
    else:
        heat_flux = (
            heat_transfer
            * 100
            * S_side
            * (tower.temp[stream - 1, section - 1] - tower.temp_thermo[stream - 1, section - 1])
        )
    # 熱電対上昇温度 [℃]
    temp_increase = heat_flux * sim_conds["COMMON_COND"]["calculation_step_time"] * 60 / heat_capacity
    # 次時刻熱電対温度 [℃]
    temp_thermocouple_reached = tower.temp_thermo[stream - 1, section - 1] + temp_increase

    output = {
        "temp_reached": temp_reached,
        "temp_thermocouple_reached": temp_thermocouple_reached,
        "hw1": hw1,  # 壁-層伝熱係数
        "Hroof": Hroof,
        "Hbb": Hbb,  # 下流セルへの熱流束 [J]
        ### 以下確認用
        "Habs": Habs,  # 発生する吸着熱 [J]
        "Hwin": Hwin,  # 内側境界からの熱流束 [J]
        "Hwout": Hwout,  # 外側境界への熱流束 [J]
        "u1": u1,  # 層伝熱係数 [W/m2/K]
    }

    return output


def calculate_heat_balance_for_wall(
    sim_conds,
    stream_conds,
    section,
    state_manager: StateVariables,
    tower_num,
    heat_output,
    heat_wall_output=None,
):
    """壁面の熱バランス計算

    Args:
        section (int): 対象のセクション番号
        variables (dict): 各セルの変数
        heat_output (dict): 隣接セルの熱バラ出力
        heat_wall_output (dict, optional): 上流セルの壁面熱バラ出力. Defaults to None.

    Returns:
        dict: 壁面熱バラ出力
    """
    tower = state_manager.towers[tower_num]
    # セクション現在温度 [℃]
    temp_now = tower.temp_wall[section - 1]
    # 内側セクション温度 [℃]
    temp_inside_cell = tower.temp[sim_conds["COMMON_COND"]["num_streams"] - 1, section - 1]
    # 外側セクション温度 [℃]
    temp_outside_cell = sim_conds["VESSEL_COND"]["ambient_temperature"]
    # 下流セクション温度 [℃]
    if section != sim_conds["COMMON_COND"]["num_sections"]:
        temp_below_cell = tower.temp_wall[section]
    # 上流壁への熱流束 [J]
    if section == 1:
        Hroof = (
            sim_conds["VESSEL_COND"]["wall_thermal_conductivity"]
            * stream_conds[3]["cross_section"]
            * (temp_now - tower.temp_lid_up)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * 60
        )
    else:
        Hroof = heat_wall_output["Hbb"]
    # 内側境界からの熱流束 [J]
    Hwin = (
        heat_output["hw1"]
        * stream_conds[3]["inner_boundary_area"]
        / sim_conds["COMMON_COND"]["num_sections"]
        * (temp_inside_cell - temp_now)
        * sim_conds["COMMON_COND"]["calculation_step_time"]
        * 60
    )
    # 外側境界への熱流束 [J]
    Hwout = (
        sim_conds["VESSEL_COND"]["external_heat_transfer_coef"]
        * stream_conds[3]["outer_boundary_area"]
        / sim_conds["COMMON_COND"]["num_sections"]
        * (temp_now - temp_outside_cell)
        * sim_conds["COMMON_COND"]["calculation_step_time"]
        * 60
    )
    # 下流壁への熱流束 [J]
    if section == sim_conds["COMMON_COND"]["num_sections"]:
        Hbb = (
            sim_conds["VESSEL_COND"]["wall_thermal_conductivity"]
            * stream_conds[3]["cross_section"]
            * (temp_now - tower.temp_lid_down)
            * sim_conds["COMMON_COND"]["calculation_step_time"]
            * 60
        )
    else:
        Hbb = (
            sim_conds["VESSEL_COND"]["wall_thermal_conductivity"]
            * stream_conds[3]["cross_section"]
            * (temp_now - tower.temp_wall[section])
        )
    # セクション到達温度 [℃]
    args = {
        "sim_conds": sim_conds,
        "stream_conds": stream_conds,
        "temp_now": temp_now,
        "Hwin": Hwin,
        "Hwout": Hwout,
        "Hbb": Hbb,
        "Hroof": Hroof,
    }
    temp_reached = optimize.newton(_optimize_wall_temperature, temp_now, args=args.values())

    output = {
        "Hbb": Hbb,
        "temp_reached": temp_reached,
        "Hroof": Hroof,  # 上流壁への熱流束 [J]
        ### 以下記録用
        "Hwin": Hwin,  # 内側境界からの熱流束 [J]
        "Hwout": Hwout,  # 外側境界への熱流束 [J]
        "Hbb": Hbb,  # 下流壁への熱流束 [J]
    }

    return output


def calculate_heat_balance_for_lid(
    sim_conds, position, state_manager: StateVariables, tower_num: int, heat_output, heat_wall_output
):
    """上下蓋の熱バランス計算

    Args:
        position (str): 上と下のどちらの蓋か
        variables (dict): 各セルの変数
        heat_output (dict): 各セルの熱バラ出力

    Returns:
        dict: 熱バラ出力
    """
    tower = state_manager.towers[tower_num]
    # セクション現在温度 [℃]
    temp_now = tower.temp_lid_up if position == "up" else tower.temp_lid_down
    # 外気への熱流束 [J]
    Hout = (
        sim_conds["VESSEL_COND"]["external_heat_transfer_coef"]
        * (temp_now - sim_conds["VESSEL_COND"]["ambient_temperature"])
        * sim_conds["COMMON_COND"]["calculation_step_time"]
        * 60
    )
    if position == "up":
        Hout *= sim_conds["BOTTOM_COND"]["outer_flange_area"]
    elif position == "down":
        Hout *= sim_conds["LID_COND"]["outer_flange_area"]
    # 底が受け取る熱(熱収支基準)
    if position == "up":
        Hlidall_heat = heat_output[2][1]["Hroof"] - heat_output[1][1]["Hroof"] - Hout - heat_wall_output[1]["Hroof"]
    else:
        Hlidall_heat = (
            heat_output[2][sim_conds["COMMON_COND"]["num_sections"]]["Hroof"]
            - heat_output[1][sim_conds["COMMON_COND"]["num_sections"]]["Hroof"]
            - Hout
            - heat_wall_output[sim_conds["COMMON_COND"]["num_sections"]]["Hbb"]
        )
    # セクション到達温度 [℃]
    args = {
        "sim_conds": sim_conds,
        "temp_now": temp_now,
        "Hlidall_heat": Hlidall_heat,
        "position": position,
    }
    temp_reached = optimize.newton(_optimize_lid_temperature, temp_now, args=args.values())

    output = {
        "temp_reached": temp_reached,
        # "Hlidall_heat": Hlidall_heat,
        # "Hlid_time": Hlid_time,
    }

    return output


def calculate_pressure_after_vacuum_pumping(sim_conds, state_manager: StateVariables, tower_num: int):
    """排気後圧力とCO2回収濃度の計算
        真空脱着モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 圧損, 積算CO2・N2回収量, CO2回収濃度, 排気後圧力,
    """
    tower = state_manager.towers[tower_num]
    ### 前準備 --------------------------------------
    # 容器内平均温度 [K]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
                for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
            ]
        )
        + 273.15
    )
    # 全圧 [PaA]
    P = tower.total_press * 1e6  # [MPaA]→[Pa]
    # 平均co2分率
    _mean_mf_co2 = np.mean(
        [
            tower.mf_co2[stream - 1, section - 1]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
            for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
        ]
    )
    # 平均n2分率
    _mean_mf_n2 = np.mean(
        [
            tower.mf_n2[stream - 1, section - 1]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
            for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
        ]
    )
    # 真空ポンプ排気ガス粘度 [Pa・s]
    # NOTE: 比熱と熱伝導率と粘度は大気圧を使用
    P_ATM = 0.101325 * 1e6
    mu = (
        CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * _mean_mf_co2
        + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * _mean_mf_n2
    )
    # 真空ポンプ排気ガス密度 [kg/m3]
    rho = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * _mean_mf_co2
        + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * _mean_mf_n2
    )

    ### 圧損計算 --------------------------------------
    _max_iteration = 1000
    P_resist = 0
    tolerance = 1e-6
    for iter in range(_max_iteration):
        P_resist_old = P_resist
        # ポンプ見せかけの全圧 [PaA]
        P_PUMP = (tower.total_press - P_resist) * 1e6
        P_PUMP = max(0, P_PUMP)
        # 真空ポンプ排気速度 [m3/min]
        vacuum_rate = 25 * (sim_conds["VACUUM_PIPING_COND"]["diameter"] ** 4) * P_PUMP / 2
        # 真空ポンプ排気ノルマル流量 [m3/min]
        vacuum_rate_N = vacuum_rate / 0.1013 * P_PUMP * 1e-6
        # 真空ポンプ排気線流速 [m/3]
        linear_velocity = vacuum_rate / sim_conds["VACUUM_PIPING_COND"]["cross_section"]
        # 真空ポンプ排気レイノルズ数
        Re = rho * linear_velocity * sim_conds["VACUUM_PIPING_COND"]["diameter"] / mu
        # 真空ポンプ排気管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        # 均圧配管圧力損失 [MPaA]
        P_resist = (
            lambda_f
            * sim_conds["VACUUM_PIPING_COND"]["length"]
            / sim_conds["VACUUM_PIPING_COND"]["diameter"]
            * linear_velocity**2
            / (2 * 9.81)
        ) * 1e-6
        # 収束判定
        if np.abs(P_resist - P_resist_old) < tolerance:
            break
        if pd.isna(P_resist):
            break
    if iter == _max_iteration - 1:
        print("収束せず: 見せかけの全圧 =", np.abs(P_resist - P_resist_old))

    ### CO2回収濃度計算 --------------------------------------
    # 排気速度 [mol/min]
    vacuum_rate_mol = 101325 * vacuum_rate_N / 8.314 / T_K
    # 排気量 [mol]
    vacuum_amt = vacuum_rate_mol * sim_conds["COMMON_COND"]["calculation_step_time"]
    # 排気CO2量 [mol]
    vacuum_amt_co2 = vacuum_amt * _mean_mf_co2
    # 排気N2量 [mol]
    vacuum_amt_n2 = vacuum_amt * _mean_mf_n2
    # 積算排気CO2量 [mol]
    accum_vacuum_amt_co2 = tower.vacuum_amt_co2 + vacuum_amt_co2
    # 積算排気N2量 [mol]
    accum_vacuum_amt_n2 = tower.vacuum_amt_n2 + vacuum_amt_n2
    # CO2回収濃度 [%]
    vacuum_co2_mf = (accum_vacuum_amt_co2 / (accum_vacuum_amt_co2 + accum_vacuum_amt_n2)) * 100

    ### 排気後圧力計算 --------------------------------
    # 排気"前"の真空排気空間の現在物質量 [mol]
    case_inner_mol_amt = (
        # P_PUMP * sim_conds["VACUUM_PIPING_COND"]["space_volume"]
        (P_PUMP + P_resist * 1e6)
        * sim_conds["VACUUM_PIPING_COND"]["space_volume"]
        / 8.314
        / T_K
    )
    # 排気"後"の現在物質量 [mol]
    case_inner_mol_amt_after_vacuum = max(0, case_inner_mol_amt - vacuum_amt)
    # 排気"後"の容器内部圧力 [MPaA]
    total_press_after_vacuum = (
        case_inner_mol_amt_after_vacuum * 8.314 * T_K / sim_conds["VACUUM_PIPING_COND"]["space_volume"] * 1e-6
    )

    # 出力
    output = {
        "P_resist": P_resist,  # 圧損 [MPaA]
        "accum_vacuum_amt_co2": accum_vacuum_amt_co2,  # 積算排気CO2量 [mol]
        "accum_vacuum_amt_n2": accum_vacuum_amt_n2,  # 積算排気N2量 [mol]
        "vacuum_co2_mf": vacuum_co2_mf,  # CO2回収濃度 [%]
        "vacuum_rate_N": vacuum_rate_N,  # 真空ポンプ排気速度 [m3/min]
        "case_inner_mol_amt_after_vacuum": case_inner_mol_amt_after_vacuum,  # 排気内部物質量 [mol]
        "total_press_after_vacuum": total_press_after_vacuum,  # 排気後圧力 [MPaA]
    }

    return output


def calculate_pressure_after_depressurization(sim_conds, state_manager, tower_num, downstream_tower_pressure):
    """減圧時の上流からの均圧配管流量計算
        バッチ均圧(上流側)モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 排気後圧力と排気後CO2・N2モル量
    """
    tower = state_manager.towers[tower_num]
    ### 上流均圧配管流量の計算 ----------------------------------------
    # 容器内平均温度 [℃]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
                for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
            ]
        )
        + 273.15
    )
    # 全圧 [PaA]
    P = tower.total_press * 1e6  # [MPaA]→[Pa]
    # 平均co2分率
    _mean_mf_co2 = np.mean(
        [
            tower.mf_co2[stream - 1, section - 1]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
            for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
        ]
    )
    # 平均n2分率
    _mean_mf_n2 = np.mean(
        [
            tower.mf_n2[stream - 1, section - 1]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
            for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
        ]
    )
    # 上流均圧管ガス粘度 [Pa・s]
    # NOTE: 比熱と熱伝導率と粘度は大気圧を使用
    P_ATM = 0.101325 * 1e6
    mu = (
        CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * _mean_mf_co2
        + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * _mean_mf_n2
    )
    # 上流均圧管ガス密度 [kg/m3]
    rho = (
        CP.PropsSI("D", "T", T_K, "P", P, "co2") * _mean_mf_co2
        + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * _mean_mf_n2
    )
    # 上流均圧配管圧力損失 [MPaA]
    _max_iteration = 1000
    P_resist = 0
    tolerance = 1e-6
    for iter in range(_max_iteration):
        P_resist_old = P_resist
        # 塔間の圧力差 [PaA]
        dP = (tower.total_press - downstream_tower_pressure - P_resist) * 1e6
        if np.abs(dP) < 1:
            dP = 0
        # 配管流速 [m/s]
        flow_rate = (
            dP
            * sim_conds["EQUALIZING_PIPING_COND"]["diameter"] ** 2
            / (32 * mu * sim_conds["EQUALIZING_PIPING_COND"]["length"])
        )
        flow_rate = max(1e-8, flow_rate)
        # 均圧管レイノルズ数
        Re = rho * abs(flow_rate) * sim_conds["EQUALIZING_PIPING_COND"]["diameter"] / mu
        # 管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        # 均圧配管圧力損失 [MPaA]
        P_resist = (
            lambda_f
            * sim_conds["EQUALIZING_PIPING_COND"]["length"]
            / sim_conds["EQUALIZING_PIPING_COND"]["diameter"]
            * flow_rate**2
            / (2 * 9.81)
        ) * 1e-6
        # 収束判定
        if np.abs(P_resist - P_resist_old) < tolerance:
            break
        if pd.isna(P_resist):
            break
    if iter == _max_iteration - 1:
        print("収束せず: 圧力差 =", np.abs(P_resist - P_resist_old))
    # 均圧配管流量 [m3/min]
    flow_amount_m3 = (
        sim_conds["EQUALIZING_PIPING_COND"]["cross_section"]
        * flow_rate
        / 60
        * sim_conds["EQUALIZING_PIPING_COND"]["flow_velocity_correction_factor"]
        * 5
    )
    # 均圧配管ノルマル流量 [m3/min]
    flow_amount_m3_N = flow_amount_m3 * tower.total_press / 0.1013
    # 均圧配管流量 [L/min] (下流塔への入力)
    flow_amount_l = flow_amount_m3_N * 1e3

    ### 次時刻の圧力計算 ----------------------------------------
    # 容器上流空間を移動する物質量 [mol]
    mw_upper_space = flow_amount_m3_N * 1000 * sim_conds["COMMON_COND"]["calculation_step_time"] / 22.4
    # 上流側の合計体積 [m3]
    V_upper_tower = (
        sim_conds["PACKED_BED_COND"]["vessel_internal_void_volume"] + sim_conds["PACKED_BED_COND"]["void_volume"]
    )
    # 上流容器圧力変化 [MPaA]
    dP_upper = 8.314 * T_K / V_upper_tower * mw_upper_space * 1e-6
    # 次時刻の容器圧力 [MPaA]
    total_press_after_depressure = tower.total_press - dP_upper

    # 出力
    output = {
        "total_press_after_depressure": total_press_after_depressure,  # 減圧後の全圧 [MPaA]
        "flow_amount_l": flow_amount_l,  # 均圧配管流量 [L/min]
        "diff_press": dP,  # 均圧塔の圧力差 [MPaA]
    }

    return output


def calculate_downstream_flow_after_depressurization(
    sim_conds, stream_conds, state_manager, tower_num, mass_balance_results, downstream_tower_pressure
):
    """減圧計算時に下流塔の圧力・流入量を計算する
        減圧モード

    Args:
        sim_conds (dict): 全体共通パラメータ
        stream_conds (dict): ストリーム内共通パラメータ
        variables (dict): 状態変数
        mass_balance_results (dict): マテバラ計算結果
        downstream_tower_pressure (float): 下流塔の現在全圧

    Returns:
        float: 減圧後の下流塔の全圧 [MPaA]
        float: 下流塔への流出CO2流量 [cm3]
        float: 下流塔への流出N2流量 [cm3]
    """
    tower = state_manager.towers[tower_num]
    ### 下流塔の圧力計算 ----------------------------------------------
    # 容器内平均温度 [℃]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
                for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
            ]
        )
        + 273.15
    )
    # 下流流出量合計（最下流セクションの合計）[L]
    most_down_section = sim_conds["COMMON_COND"]["num_sections"]
    sum_outflow_fr = (
        sum(
            [
                mass_balance_results[stream][most_down_section]["outflow_fr_co2"]
                + mass_balance_results[stream][most_down_section]["outflow_fr_n2"]
                for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
            ]
        )
        / 1e3
    )
    # 下流流出物質量 [mol]
    sum_outflow_mol = sum_outflow_fr / 22.4
    # 均圧下流側空間体積 [m3]
    V_downflow = (
        sim_conds["EQUALIZING_PIPING_COND"]["volume"]
        + sim_conds["PACKED_BED_COND"]["void_volume"]
        + sim_conds["PACKED_BED_COND"]["vessel_internal_void_volume"]
    )
    # 下流容器圧力変化 [MPaA]
    dP = 8.314 * T_K / V_downflow * sum_outflow_mol / 1e6
    # 次時刻の下流容器全圧 [MPaA]
    total_press_after_depressure_downflow = downstream_tower_pressure + dP

    ### 下流容器への流入量 --------------------------------------------
    # 下流塔への合計流出CO2流量 [cm3]
    sum_outflow_fr_co2 = sum(
        [
            mass_balance_results[stream][most_down_section]["outflow_fr_co2"]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
        ]
    )
    # 下流塔への合計流出N2流量 [cm3]
    sum_outflow_fr_n2 = sum(
        [
            mass_balance_results[stream][most_down_section]["outflow_fr_n2"]
            for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
        ]
    )
    # 下流塔への流出CO2, N2流量 [cm3]
    outflow_fr = {}
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"]):
        outflow_fr[stream] = {}
        outflow_fr[stream]["outflow_fr_co2"] = sum_outflow_fr_co2 * stream_conds[stream]["area_fraction"]
        outflow_fr[stream]["outflow_fr_n2"] = sum_outflow_fr_n2 * stream_conds[stream]["area_fraction"]

    # 出力
    output = {
        "total_press_after_depressure_downflow": total_press_after_depressure_downflow,  # 減圧後の下流塔の全圧 [MPaA]
        "outflow_fr": outflow_fr,  # 下流塔への流出CO2, N2流量 [cm3]
    }

    return output


def calculate_pressure_after_desorption(
    sim_conds, state_manager, tower_num, mole_fraction_results, vacuum_pumping_results
):
    """気相放出後の全圧の計算

    Args:
        variables (dict): 状態変数
        mole_fraction_results (dict): マテバラの出力結果（モル分率）

    Returns:
        dict: 気相放出後の全圧
    """
    tower = state_manager.towers[tower_num]
    # 容器内平均温度 [℃]
    T_K = (
        np.mean(
            [
                tower.temp[stream - 1, section - 1]
                for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
                for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
            ]
        )
        + 273.15
    )
    # 気相放出後の全物質量合計 [mol]
    sum_desorp_mw = (
        sum(
            [
                mole_fraction_results[stream][section]["desorp_mw_all_after_vacuum"]
                for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"])
                for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"])
            ]
        )
        / 22.4
        / 1e3
    )
    # 配管上のモル量を加算
    sum_desorp_mw += (
        vacuum_pumping_results["total_press_after_vacuum"]
        * 1e6
        * sim_conds["VACUUM_PIPING_COND"]["volume"]
        / 8.314
        / T_K
    )
    # 気相放出後の全圧 [MPaA]
    pressure_after_desorption = sum_desorp_mw * 8.314 * T_K / sim_conds["VACUUM_PIPING_COND"]["space_volume"] * 1e-6

    return pressure_after_desorption


def calculate_pressure_after_batch_adsorption(
    sim_conds, state_manager: StateVariables, tower_num: int, is_series_operation
):
    """バッチ吸着における圧力変化

    Args:
        sim_conds (dict): 実験条件
        variables (dict): 状態変数

    Return:
        float: バッチ吸着後の全圧
    """
    tower = state_manager.towers[tower_num]

    # 気体定数 [J/K/mol]
    R = 8.314
    # 平均温度 [K]
    temp_mean = []
    for stream in range(1, 1 + sim_conds["COMMON_COND"]["num_streams"]):
        for section in range(1, 1 + sim_conds["COMMON_COND"]["num_sections"]):
            temp_mean.append(tower.temp[stream - 1, section - 1])
    temp_mean = np.mean(temp_mean)
    temp_mean += 273.15
    # 空間体積（配管含む）
    if is_series_operation:
        V = (
            (sim_conds["PACKED_BED_COND"]["void_volume"] + sim_conds["PACKED_BED_COND"]["vessel_internal_void_volume"])
            * 2
            + sim_conds["PACKED_BED_COND"]["upstream_piping_volume"]
            + sim_conds["EQUALIZING_PIPING_COND"]["volume"]
        )
    else:
        V = (
            sim_conds["PACKED_BED_COND"]["void_volume"]
            + sim_conds["PACKED_BED_COND"]["vessel_internal_void_volume"]
            + sim_conds["PACKED_BED_COND"]["upstream_piping_volume"]
        )
    # ノルマル体積流量
    F = sim_conds["FEED_GAS_COND"]["total_flow_rate"]  # バッチ吸着: 導入ガスの流量
    # 圧力変化量
    diff_pressure = R * temp_mean / V * (F / 22.4) * sim_conds["COMMON_COND"]["calculation_step_time"] / 1e6
    # 変化後全圧
    pressure_after_batch_adsorption = tower.total_press + diff_pressure

    return pressure_after_batch_adsorption


def _calculate_equilibrium_adsorption_amount(P, T):
    """平衡吸着量を計算

    Args:
        P (float): 圧力 [kPaA]
        T (float): 温度 [K]
    Returns:
        平衡吸着量
    """
    # 吸着等温式（シンボリック回帰による近似式）
    adsorp_amt_equilibrium = (
        P
        * (252.0724 - 0.50989705 * T)
        / (P - 3554.54819062669 * (1 - 0.0655247236249063 * np.sqrt(T)) ** 3 + 1.7354268)
    )
    return adsorp_amt_equilibrium


def _optimize_bed_temperature(
    temp_reached,
    sim_conds,
    stream_conds,
    gas_cp,
    Mgas,
    temp_now,
    Habs,
    Hwin,
    Hwout,
    Hbb,
    Hroof,
    stream,
):
    """セクション到達温度算出におけるソルバー用関数

    Args:
        temp_reached (float): セクション到達温度
        args (dict): 充填層が受ける熱を計算するためのパラメータ

    Returns:
        float: 充填層が受ける熱の熱収支基準と時間基準の差分
    """
    # 流入ガスが受け取る熱 [J]
    Hgas = gas_cp * Mgas * (temp_reached - temp_now)
    # 充填層が受け取る熱(ΔT基準) [J]
    Hbed_time = (
        sim_conds["PACKED_BED_COND"]["heat_capacity"]
        * stream_conds[stream]["area_fraction"]
        / sim_conds["COMMON_COND"]["num_sections"]
        * (temp_reached - temp_now)
    )
    # 充填層が受け取る熱(熱収支基準) [J]
    Hbed_heat_blc = Habs - Hgas + Hwin - Hwout - Hbb - Hroof
    return Hbed_heat_blc - Hbed_time


def _optimize_wall_temperature(
    temp_reached,
    sim_conds,
    stream_conds,
    temp_now,
    Hwin,
    Hwout,
    Hbb,
    Hroof,
):
    """壁面の到達温度算出におけるソルバー用関数

    Args:
        temp_reached (float): セクション到達温度
        args (dict): 充填層が受ける熱を計算するためのパラメータ

    Returns:
        float: 充填層が受ける熱の熱収支基準と時間基準の差分
    """
    # 壁が受け取る熱(熱収支基準) [J]
    Hwall_heat_blc = Hwin - Hroof - Hwout - Hbb
    # 壁が受け取る熱(ΔT基準) [J]
    Hwall_time = (
        sim_conds["VESSEL_COND"]["wall_specific_heat_capacity"]
        * stream_conds[1 + sim_conds["COMMON_COND"]["num_streams"]]["wall_weight"]
        * (temp_reached - temp_now)
    )

    return Hwall_heat_blc - Hwall_time


def _optimize_lid_temperature(temp_reached, sim_conds, temp_now, Hlidall_heat, position):
    """上下蓋の到達温度算出におけるソルバー用関数

    Args:
        temp_reached (float): セクション到達温度
        args (dict): 充填層が受ける熱を計算するためのパラメータ

    Returns:
        float: 充填層が受ける熱の熱収支基準と時間基準の差分
    """
    # 壁が受け取る熱(ΔT基準) [J]
    Hlid_time = sim_conds["VESSEL_COND"]["wall_specific_heat_capacity"] * (temp_reached - temp_now)
    if position == "up":
        Hlid_time *= sim_conds["LID_COND"]["flange_total_weight"]
    else:
        Hlid_time *= sim_conds["BOTTOM_COND"]["flange_total_weight"]

    return Hlidall_heat - Hlid_time
