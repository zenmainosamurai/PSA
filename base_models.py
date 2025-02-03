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
import japanize_matplotlib
from scipy import optimize
import CoolProp.CoolProp as CP

from utils import const, init_functions, plot_csv, other_utils

import warnings
warnings.simplefilter('error')


def material_balance_adsorp(sim_conds, stream_conds, stream, section, variables, inflow_gas=None):
    """ 任意セルのマテリアルバランスを計算する
        吸着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        inflow_gas (dict): 上部セルの出力値

    Returns:
        dict: 対象セルの計算結果
    """
    ### マテバラ計算開始 ------------------------------------------------

    # セクション吸着材量 [g]
    Mabs = stream_conds[stream]["Mabs"] / sim_conds["CELL_SPLIT"]["num_sec"]
    # 流入CO2流量 [cm3]
    if (section == 1) & (inflow_gas is None): # 最上流かつ流入ガスが導入ガス
        inflow_fr_co2 = (
            sim_conds["INFLOW_GAS_COND"]["fr_co2"] * sim_conds["dt"]
            * stream_conds[stream]["streamratio"] * 1000
        )
    else :
        inflow_fr_co2 = inflow_gas["outflow_fr_co2"]
    # 流入N2流量 [cm3]
    if (section == 1) & (inflow_gas is None):
        inflow_fr_n2 = (
            sim_conds["INFLOW_GAS_COND"]["fr_n2"] * sim_conds["dt"]
            * stream_conds[stream]["streamratio"] * 1000
        )
    else:
        inflow_fr_n2 = inflow_gas["outflow_fr_n2"]
    # 流入CO2分率
    inflow_mf_co2 = inflow_fr_co2 / (inflow_fr_co2 + inflow_fr_n2)
    # 流入N2分率
    inflow_mf_n2 = inflow_fr_n2 / (inflow_fr_co2 + inflow_fr_n2)
    # 全圧 [MPaA]
    total_press = variables["total_press"]
    # CO2分圧 [MPaA]
    p_co2 = total_press * inflow_mf_co2
    # 現在温度 [℃]
    temp = variables["temp"][stream][section]
    # ガス密度 [kg/m3]
    gas_density = (
        sim_conds["INFLOW_GAS_COND"]["dense_co2"] * inflow_mf_co2
        + sim_conds["INFLOW_GAS_COND"]["dense_n2"] * inflow_mf_n2
    )
    # ガス比熱 [kJ/kg/K]
    gas_cp = (
        sim_conds["INFLOW_GAS_COND"]["cp_co2"] * inflow_mf_co2
        + sim_conds["INFLOW_GAS_COND"]["cp_n2"] * inflow_mf_n2
    )
    # 現在雰囲気の平衡吸着量 [cm3/g-abs]
    P_kpa = p_co2 * 1000 # [kPaA]
    T_K = temp + 273.15 # [K]
    adsorp_amt_equilibrium = _equilibrium_adsorp_amt(P_kpa, T_K)
    # 現在の既存吸着量 [cm3/g-abs]
    adsorp_amt_current = variables["adsorp_amt"][stream][section]
    # 理論新規吸着量 [cm3/g-abs]
    if adsorp_amt_equilibrium != 0:
        if adsorp_amt_equilibrium >= adsorp_amt_current:
            adsorp_amt_estimate_abs = (
                sim_conds["PACKED_BED_COND"]["ks_adsorp"] ** (adsorp_amt_current / adsorp_amt_equilibrium)
                / sim_conds["PACKED_BED_COND"]["rho_abs"]
                * 6 * (1 - sim_conds["PACKED_BED_COND"]["epsilon"]) * sim_conds["PACKED_BED_COND"]["phi"]
                / sim_conds["PACKED_BED_COND"]["dp"] * (adsorp_amt_equilibrium - adsorp_amt_current)
                * sim_conds["dt"] / 1e6 * 60
            )
        else:
            adsorp_amt_estimate_abs = (
                sim_conds["PACKED_BED_COND"]["ks_desorp"] ** (adsorp_amt_current / adsorp_amt_equilibrium)
                / sim_conds["PACKED_BED_COND"]["rho_abs"]
                * 6 * (1 - sim_conds["PACKED_BED_COND"]["epsilon"]) * sim_conds["PACKED_BED_COND"]["phi"]
                / sim_conds["PACKED_BED_COND"]["dp"] * (adsorp_amt_equilibrium - adsorp_amt_current)
                * sim_conds["dt"] / 1e6 * 60
            )
    else:
        adsorp_amt_estimate_abs = 0
    # セクション理論新規吸着量 [cm3]
    adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
    # 実際のセクション新規吸着量 [cm3]
    adsorp_amt_estimate = min(adsorp_amt_estimate, inflow_fr_co2)
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
        "adsorp_amt_equilibrium": adsorp_amt_equilibrium, # 平衡吸着量
        "adsorp_amt_estimate": adsorp_amt_estimate,
        "accum_adsorp_amt": accum_adsorp_amt,
        "outflow_fr_co2": outflow_fr_co2,
        "outflow_fr_n2": outflow_fr_n2,
        "outflow_mf_co2": outflow_mf_co2,
        "outflow_mf_n2": outflow_mf_n2,
        "adsorp_amt_estimate_abs": adsorp_amt_estimate_abs,
        "desorp_mw_co2": 0,
        "p_co2": p_co2,
    }

    return output

def material_balance_desorp(sim_conds, stream_conds, stream, section, variables, vaccume_output):
    """ 任意セルのマテリアルバランスを計算する
        脱着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vaccume_output (dict): 排気後圧力計算の出力

    Returns:
        dict: 対象セルの計算結果
    """
    # 現在温度 [℃]
    temp_now = variables["temp"][stream][section]
    # CoolProp用の前準備
    T_K = temp_now + 273.15 # 温度 [K]
    P = vaccume_output["total_press_after_vaccume"] * 1e6 # 圧力 [Pa]        
    # セクション吸着材量 [g]
    Mabs = stream_conds[stream]["Mabs"] / sim_conds["CELL_SPLIT"]["num_sec"]
    # ガス密度 [kg/m3]
    gas_density = (
        CP.PropsSI('D', 'T', T_K, 'P', P, "co2") * variables["mf_co2"][stream][section]
        + CP.PropsSI('D', 'T', T_K, 'P', P, "nitrogen") * variables["mf_n2"][stream][section]
    )
    # ガス比熱 [kJ/kg/K]
    gas_cp = (
        CP.PropsSI('C', 'T', T_K, 'P', P, "co2") * variables["mf_co2"][stream][section]
        + CP.PropsSI('C', 'T', T_K, 'P', P, "nitrogen") * variables["mf_n2"][stream][section]
    )
    # CO2分圧 [MPaA]
    p_co2 = vaccume_output["total_press_after_vaccume"] * variables["mf_co2"][stream][section]
    # 現在雰囲気の平衡吸着量 [cm3/g-abs]
    P_kpa = p_co2 * 1000 # [MPaA] → [kPaA]
    adsorp_amt_equilibrium = _equilibrium_adsorp_amt(P_kpa, T_K)
    # 現在の既存吸着量 [cm3/g-abs]
    adsorp_amt_current = variables["adsorp_amt"][stream][section]
    # 理論新規吸着量 [cm3/g-abs]
    if adsorp_amt_equilibrium >= adsorp_amt_current:
        adsorp_amt_estimate_abs = (
            sim_conds["PACKED_BED_COND"]["ks_adsorp"] ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / sim_conds["PACKED_BED_COND"]["rho_abs"]
            * 6 * (1 - sim_conds["PACKED_BED_COND"]["epsilon"]) * sim_conds["PACKED_BED_COND"]["phi"]
            / sim_conds["PACKED_BED_COND"]["dp"] * (adsorp_amt_equilibrium - adsorp_amt_current)
            * sim_conds["dt"] / 1e6 * 60
        )
    else:
        adsorp_amt_estimate_abs = (
            sim_conds["PACKED_BED_COND"]["ks_desorp"] ** (adsorp_amt_current / adsorp_amt_equilibrium)
            / sim_conds["PACKED_BED_COND"]["rho_abs"]
            * 6 * (1 - sim_conds["PACKED_BED_COND"]["epsilon"]) * sim_conds["PACKED_BED_COND"]["phi"]
            / sim_conds["PACKED_BED_COND"]["dp"] * (adsorp_amt_equilibrium - adsorp_amt_current)
            * sim_conds["dt"] / 1e6 * 60
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

    # 出力
    output = {
        "inflow_fr_co2": 0,
        "inflow_fr_n2": 0,
        "inflow_mf_co2": 0,
        "inflow_mf_n2": 0,
        "gas_density": gas_density,
        "gas_cp": gas_cp,
        "adsorp_amt_equilibrium": adsorp_amt_equilibrium, # 平衡吸着量
        "adsorp_amt_estimate": adsorp_amt_estimate,
        "accum_adsorp_amt": accum_adsorp_amt,
        "outflow_fr_co2": 0,
        "outflow_fr_n2": 0,
        "outflow_mf_co2": 0,
        "outflow_mf_n2": 0,
        "adsorp_amt_estimate_abs": adsorp_amt_estimate_abs,
        "desorp_mw_co2": desorp_mw_co2,
        "p_co2": p_co2,
    }

    return output

def material_balance_valve_stop(stream, section, variables):
    """ 任意セルのマテリアルバランスを計算する
        脱着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vaccume_output (dict): 排気後圧力計算の出力

    Returns:
        dict: 対象セルの計算結果
    """
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
        "accum_adsorp_amt": variables["adsorp_amt"][stream][section],
        "outflow_fr_co2": 0,
        "outflow_fr_n2": 0,
        "outflow_mf_co2": 0,
        "outflow_mf_n2": 0,
        "adsorp_amt_estimate_abs": 0,
        "desorp_mw_co2": 0,
        "p_co2": 0,
    }

    return output

def heat_balance(sim_conds, stream_conds, stream, section, variables, mode,
                    material_output=None, heat_output=None, vaccume_output=None):
    """ 対象セルの熱バランスを計算する

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 状態変数
        mode (int): 吸着・脱着等の運転モード
        material_output (dict): 対象セルのマテバラ出力
        heat_output (dict): 上流セクションの熱バラ出力
        vaccume_output (dict): 排気後圧力計算の出力（脱着時）

    Returns:
        dict: 対象セルの熱バラ出力
    """
    ### 前準備 ------------------------------------------------------

    # セクション現在温度 [℃]
    temp_now = variables["temp"][stream][section]
    # 内側セクション温度 [℃]
    if stream == 1:
        temp_inside_cell = 18
    else:
        temp_inside_cell = variables["temp"][stream-1][section]
    # 外側セクション温度 [℃]
    if stream != sim_conds["CELL_SPLIT"]["num_str"]:
        temp_outside_cell = variables["temp"][stream+1][section]
    else:
        temp_outside_cell = variables["temp_wall"][section]
    # 下流セクション温度 [℃]
    if section != sim_conds["CELL_SPLIT"]["num_sec"]:
        temp_below_cell = variables["temp"][stream][section+1]
    # 発生する吸着熱 [J]
    if mode == 1:
        Habs = 0 # 弁停止モードでは0
    else:
        Habs = (
            material_output["adsorp_amt_estimate"] / 1000 / 22.4
            * sim_conds["INFLOW_GAS_COND"]["mw_co2"]
            * sim_conds["INFLOW_GAS_COND"]["adsorp_heat_co2"]
        )
    # 流入ガス質量 [g]
    if mode in [1, 2]:
        Mgas = 0 # 弁停止・脱着モードでは0
    else:
        Mgas = (
            material_output["inflow_fr_co2"] / 1000 / 22.4
            * sim_conds["INFLOW_GAS_COND"]["mw_co2"]
            + material_output["inflow_fr_n2"] / 1000 / 22.4
            * sim_conds["INFLOW_GAS_COND"]["mw_n2"]
        )
    # 流入ガス比熱 [J/g/K]
    if mode == 1:
        gas_cp = 0 # 弁停止モードでは0
    else:
        gas_cp = material_output["gas_cp"]
    # 内側境界面積 [m2]
    Ain = stream_conds[stream]["Ain"] / sim_conds["CELL_SPLIT"]["num_sec"]
    # 外側境界面積 [m2]
    Aout = stream_conds[stream]["Aout"] / sim_conds["CELL_SPLIT"]["num_sec"]
    # 下流セル境界面積 [m2]
    Abb = stream_conds[stream]["Sstream"]
    # 壁-層伝熱係数、層伝熱係数
    if mode == 0:
        hw1, u1 = _heat_transfer_coef(sim_conds,
                                      stream_conds,
                                      stream,
                                      temp_now,
                                      mode,
                                      variables,
                                      material_output)
    elif mode == 1: # 弁停止モードでは直前値に置換
        hw1 = variables["heat_t_coef"][stream][section]
        u1 = variables["heat_t_coef_wall"][stream][section]
    elif mode == 2: # 脱着モードでは入力に排気後圧力計算の出力を使用
        hw1 = variables["heat_t_coef"][stream][section]
        u1 = variables["heat_t_coef_wall"][stream][section]
        # hw1, u1 = _heat_transfer_coef(stream,
        #                               temp_now,
        #                               mode,
        #                               variables,
        #                               material_output,
        #                               vaccume_output)

    ### 熱流束計算 ---------------------------------------------------

    # 内側境界からの熱流束 [J]
    if stream == 1:
        Hwin = 0
    else:
        Hwin = u1 * Ain * (temp_inside_cell - temp_now) * sim_conds["dt"] * 60
    # 外側境界への熱流束 [J]
    if stream == sim_conds["CELL_SPLIT"]["num_str"]:
        Hwout = hw1 * Aout * (temp_now - temp_outside_cell) * sim_conds["dt"] * 60
    else :
        Hwout = u1 * Aout * (temp_now - temp_outside_cell) * sim_conds["dt"] * 60
    # 下流セルへの熱流束 [J]
    if section == sim_conds["CELL_SPLIT"]["num_sec"]:
        Hbb = ( # 下蓋への熱流束
            hw1 * stream_conds[stream]["Sstream"]
            * (temp_now - variables["temp_lid"]["down"])
            * sim_conds["dt"] * 60
        )
    else:
        Hbb = ( # 下流セルへの熱流束
            u1 * Abb * (temp_now - temp_below_cell)
            * sim_conds["dt"] * 60
        )
    # 上流セルヘの熱流束 [J]
    if section == 1: # 上蓋からの熱流束
        Hroof = (
            hw1 * stream_conds[stream]["Sstream"]
            * (temp_now - variables["temp_lid"]["up"])
            * sim_conds["dt"] * 60
        )
    else: # 上流セルヘの熱流束 = -1 * 上流セルの「下流セルへの熱流束」
        Hroof = - heat_output["Hbb"]

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
    temp_reached = optimize.newton(__optimize_temp_reached, temp_now, args=args.values())

    output = {
        "temp_reached": temp_reached,
        "hw1": hw1, # 壁-層伝熱係数
        "Hroof": Hroof,
        "Hbb": Hbb, # 下流セルへの熱流束 [J]
        ### 以下確認用
        "Habs": Habs, # 発生する吸着熱 [J]
        "Hwin": Hwin, # 内側境界からの熱流束 [J]
        "Hwout": Hwout, # 外側境界への熱流束 [J]
        "u1": u1, # 層伝熱係数 [W/m2/K]
    }

    return output

def heat_balance_wall(sim_conds, stream_conds,
                           section, variables, heat_output, heat_wall_output=None):
    """ 壁面の熱バランス計算

    Args:
        section (int): 対象のセクション番号
        variables (dict): 各セルの変数
        heat_output (dict): 隣接セルの熱バラ出力
        heat_wall_output (dict, optional): 上流セルの壁面熱バラ出力. Defaults to None.

    Returns:
        dict: 壁面熱バラ出力
    """
    # セクション現在温度 [℃]
    temp_now = variables["temp_wall"][section]
    # 内側セクション温度 [℃]
    temp_inside_cell = variables["temp"][sim_conds["CELL_SPLIT"]["num_str"]][section]
    # 外側セクション温度 [℃]
    temp_outside_cell = sim_conds["DRUM_WALL_COND"]["temp_outside"]
    # 下流セクション温度 [℃]
    if section != sim_conds["CELL_SPLIT"]["num_sec"]:
        temp_below_cell = variables["temp_wall"][section+1]
    # 上流壁への熱流束 [J]
    if section == 1:
        Hroof = (
            sim_conds["DRUM_WALL_COND"]["c_drumw"]
            * stream_conds[3]["Sstream"]
            * (temp_now - variables["temp_lid"]["up"])
            * sim_conds["dt"] * 60
        )
    else:
        Hroof = heat_wall_output["Hbb"]
    # 内側境界からの熱流束 [J]
    Hwin = (
        heat_output["hw1"] * stream_conds[3]["Ain"] / sim_conds["CELL_SPLIT"]["num_sec"]
        * (temp_inside_cell - temp_now) * sim_conds["dt"] * 60
    )
    # 外側境界への熱流束 [J]
    Hwout = (
        sim_conds["DRUM_WALL_COND"]["coef_outair_heat"] * stream_conds[3]["Aout"]
        / sim_conds["CELL_SPLIT"]["num_sec"] * (temp_now - temp_outside_cell)
        * sim_conds["dt"] * 60
    )
    # 下流壁への熱流束 [J]
    if section == sim_conds["CELL_SPLIT"]["num_sec"]:
        Hbb = (
            sim_conds["DRUM_WALL_COND"]["c_drumw"]
            * stream_conds[3]["Sstream"]
            * (temp_now - variables["temp_lid"]["down"])
            * sim_conds["dt"] * 60
        )
    else:
        Hbb = (
            sim_conds["DRUM_WALL_COND"]["c_drumw"] * stream_conds[3]["Sstream"]
            * (temp_now - variables["temp_wall"][section+1])
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
    temp_reached = optimize.newton(__optimize_temp_reached_wall, temp_now, args=args.values())

    output = {
        "Hbb": Hbb,
        "temp_reached": temp_reached,
        "Hroof": Hroof, # 上流壁への熱流束 [J]
        ### 以下記録用
        "Hwin": Hwin, # 内側境界からの熱流束 [J]
        "Hwout": Hwout, # 外側境界への熱流束 [J]
        "Hbb": Hbb, # 下流壁への熱流束 [J]
    }

    return output

def heat_balance_lid(sim_conds, position, variables, heat_output, heat_wall_output):
    """ 上下蓋の熱バランス計算

    Args:
        position (str): 上と下のどちらの蓋か
        variables (dict): 各セルの変数
        heat_output (dict): 各セルの熱バラ出力

    Returns:
        dict: 熱バラ出力
    """
    # セクション現在温度 [℃]
    temp_now = variables["temp_lid"][position]
    # 外気への熱流束 [J]
    Hout = (
        sim_conds["DRUM_WALL_COND"]["coef_outair_heat"]
        * (temp_now - sim_conds["DRUM_WALL_COND"]["temp_outside"])
        * sim_conds["dt"] * 60
    )
    if position == "up":
        Hout *= sim_conds["LID_COND"]["DOWN"]["Sflange_up"]
    elif position == "down":
        Hout *= sim_conds["LID_COND"]["UP"]["Sflange_up"]
    # 底が受け取る熱(熱収支基準)
    if position == "up":
        Hlidall_heat = (
            heat_output[2][1]["Hroof"] - heat_output[1][1]["Hroof"]
            - Hout - heat_wall_output[1]["Hroof"]
        )
    else:
        Hlidall_heat = (
            heat_output[2][sim_conds["CELL_SPLIT"]["num_sec"]]["Hroof"] - heat_output[1][sim_conds["CELL_SPLIT"]["num_sec"]]["Hroof"]
            - Hout - heat_wall_output[sim_conds["CELL_SPLIT"]["num_sec"]]["Hbb"]
        )
    # セクション到達温度 [℃]
    args = {
        "sim_conds": sim_conds,
        "temp_now": temp_now,
        "Hlidall_heat": Hlidall_heat,
        "position": position,
    }
    temp_reached = optimize.newton(__optimize_temp_reached_lid, temp_now, args=args.values())

    output = {
        "temp_reached": temp_reached,
        # "Hlidall_heat": Hlidall_heat,
        # "Hlid_time": Hlid_time,
    }

    return output

def total_press_after_vacuuming(sim_conds, variables, stop_mode=False):
    """ 排気後圧力の計算
        真空脱着モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 排気後圧力と排気後CO2・N2モル量
    """
    # 容器内平均温度 [℃]
    temp_now_mean = np.mean([
        variables["temp"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                           for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    # 容器内部空間体積 [m3]
    case_inner_volume = (
        sim_conds["PACKED_BED_COND"]["Vbed"] * sim_conds["PACKED_BED_COND"]["epsilon"]
        # / sim_conds["CELL_SPLIT"]["num_str"] / sim_conds["CELL_SPLIT"]["num_sec"]
    )
    # 容器内部空間物質量 [mol]
    case_inner_amount = (
        variables["total_press"]*1e6 * case_inner_volume / 8.314 /
        (temp_now_mean + 273.15)
    )
    if stop_mode: # 排気をしない場合(停止モード)
        # 真空ポンプ排気速度 [mol]
        vacuum_rate = 0
        # 排気後容器内部空間物質量 [mol]
        case_inner_amount_after_vaccume = case_inner_amount
        # 排気後容器内部圧力 [MPaA]
        total_press_after_vaccume = variables["total_press"]
    else: # 排気をする場合(真空引きモード)
        # 真空ポンプ排気速度 [m3/min]
        if variables["total_press"]*1e6 > 900:
            vacuum_rate = 0.57
        elif variables["total_press"]*1e6 > 100:
            vacuum_rate = 0.085 * np.log(variables["total_press"]*1e6 - 99)
        else:
            vacuum_rate = 0
        # 真空ポンプ排気速度 [mol]
        vacuum_rate = (
            variables["total_press"]*1e6 * vacuum_rate / 8.314 / (temp_now_mean + 273.15)
        )
        # 排気量 [mol]
        vaccume_amount = vacuum_rate * sim_conds["dt"]
        # 排気後容器内部空間物質量 [mol]
        case_inner_amount_after_vaccume = case_inner_amount - vaccume_amount
        case_inner_amount_after_vaccume = max(0, case_inner_amount_after_vaccume)
        # 排気後容器内部圧力 [MPaA]
        total_press_after_vaccume = (
            case_inner_amount_after_vaccume * 8.314 * (temp_now_mean + 273.15)
            / case_inner_volume / 1e6
        )
    # 排気後CO2モル量 [mol]
    mf_co2 = np.mean([
        variables["mf_co2"][stream][section] for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"])
                                             for section in range(1, 1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    mw_co2_after_vaccume = mf_co2 * case_inner_amount_after_vaccume
    # 排気後N2モル量 [mol]
    mf_n2 = np.mean([
        variables["mf_n2"][stream][section] for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"])
                                             for section in range(1, 1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    mw_n2_after_vaccume = mf_n2 * case_inner_amount_after_vaccume

    # 出力
    output = {
        "case_inner_amount_after_vaccume": case_inner_amount_after_vaccume, # 排気後容器内部空間物質量
        "total_press_after_vaccume": total_press_after_vaccume, # 排気後圧力
        "mw_co2_after_vaccume": mw_co2_after_vaccume, # 排気後CO2モル量
        "mw_n2_after_vaccume": mw_n2_after_vaccume, # 排気後N2モル量
        "vacuum_rate": vacuum_rate, # 真空ポンプの排気速度
    }

    return output

def total_press_after_decompression(sim_conds, variables, downflow_total_press):
    """ バッチ減圧後の圧力計算
        バッチ均圧(上流側)モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 排気後圧力と排気後CO2・N2モル量
    """
    # 容器内平均温度 [℃]
    T_K = np.mean([
        variables["temp"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                           for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ]) + 273.15
    #　ガス粘度 [Pa・s]
    P = variables["total_press"] * 1e6 # [MPaA]→[Pa]
    _mean_mf_co2 = np.mean([ # 平均co2分率
        variables["mf_co2"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                             for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    _mean_mf_n2 = np.mean([ # 平均n2分率
        variables["mf_n2"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                            for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    mu = (
        CP.PropsSI('V', 'T', T_K, 'P', P, "co2") * _mean_mf_co2
        + CP.PropsSI('V', 'T', T_K, 'P', P, "nitrogen") * _mean_mf_n2
    ) / 1e6
    # ?
    rho = (
        CP.PropsSI('D', 'T', T_K, 'P', P, "co2") * _mean_mf_co2
        + CP.PropsSI('D', 'T', T_K, 'P', P, "nitrogen") * _mean_mf_n2
    ) / 1e6
    # 均圧配管圧力損失 [MPaA]
    _max_iteration = 100
    P_resist = 0
    tolerance = 1e-6
    L = 1.0
    for iter in range(_max_iteration):
        P_resist_old = P_resist
        # 塔間の圧力差 [MPaA]
        dP = variables["total_press"] - downflow_total_press - P_resist
        if dP <= 1e-10:
            dP = 0
        # 配管流速 [m/s]
        flow_rate = (
            dP * 1e6 * sim_conds["PRESS_EQUAL_PIPE_COND"]["Dpipe"] ** 2
            / (32 * mu * sim_conds["PRESS_EQUAL_PIPE_COND"]["Lpipe"])
        )
        # ?
        Re = (
            rho * abs(flow_rate) * sim_conds["PRESS_EQUAL_PIPE_COND"]["Dpipe"] / mu
        )
        # ?
        lambda_f = 64 / Re if Re != 0 else 0
        # 均圧配管圧力損失 [MPaA]
        P_resist = (
            lambda_f * L / sim_conds["PRESS_EQUAL_PIPE_COND"]["Dpipe"]
            * flow_rate ** 2 / (2*9.81) * 1e-6
        )
        # 収束判定
        if np.abs(P_resist - P_resist_old) < tolerance:
            break
    if iter == _max_iteration:
        print("収束せず")
    # 均圧配管流量 [m3/min]
    flow_amount_m3 = (
        sim_conds["PRESS_EQUAL_PIPE_COND"]["Spipe"] * flow_rate * 60
        * sim_conds["PRESS_EQUAL_PIPE_COND"]["coef_fr"]
    )
    # 均圧配管流量 [L/min] (下流塔への入力)
    flow_amount_l = flow_amount_m3 * 1e3

    # 出力
    output = {
        "flow_amount_m3": flow_amount_m3, # 均圧配管流量 [m3/min]
        "flow_amount_l": flow_amount_l, # 均圧配管流量 [L/min]
        "diff_press": dP, # 均圧塔の圧力差 [MPaA]
    }

    return output

def mf_after_vaccume_decompression(sim_conds, stream_conds, stream, section, variables, vaccume_output, mb_dict):
    """ バッチ減圧後の圧力計算
        バッチ均圧(上流側)モード

    Args:
        variables (dict): 状態変数
        vaccume_output (dict): total_press_after_decompressionの出力
        mb_dict (dict): 各セクションのマテバラ計算結果

    Returns:
        dict: 排気後圧力と排気後CO2・N2モル量
    """
    # 容器内平均温度 [℃]
    T_K = np.mean([
        variables["temp"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                           for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ]) + 273.15
    # 容器内部空間体積 [m3]
    case_inner_volume = (
        sim_conds["PACKED_BED_COND"]["Vbed"] * sim_conds["PACKED_BED_COND"]["epsilon"]
        * stream_conds[stream]["streamratio"] / sim_conds["CELL_SPLIT"]["num_sec"]
    )
    # 容器内部空間物質量 [mol]
    case_inner_amount = (
        variables["total_press"]*1e6 * case_inner_volume / 8.314 / T_K
    )
    # 物質移動量・流出量 [mol]
    flow_amount_mol = vaccume_output["flow_amount_m3"] * 1e3 * sim_conds["dt_eq"] / 22.4
    flow_amount_mol = min(flow_amount_mol, case_inner_amount) # 内部空間物質量は超過しない
    # セクション流出量 [mol]
    flow_amount_mol *= (stream_conds[stream]["streamratio"] / sim_conds["CELL_SPLIT"]["num_sec"])
    # 脱着による気相放出CO2モル量 [mol]
    desorp_mw = mb_dict[stream][section]["desorp_mw_co2"]
    # 現在CO2モル量 [mol]
    now_amt_co2 = case_inner_amount * variables["mf_co2"][stream][section]
    # 現在N2モル量 [mol]
    now_amt_n2 = case_inner_amount * variables["mf_n2"][stream][section]
    # N2容器外流出量 [mol]
    outflow_amt_n2 = min(now_amt_n2, flow_amount_mol)
    # CO2容器外流出量 [mol]
    outflow_amt_co2 = flow_amount_mol - outflow_amt_n2
    # 次時刻内部空間物質量 [mol]
    next_case_inner_amount = case_inner_amount + desorp_mw - flow_amount_mol
    # 次時刻容器内部圧力(全圧) [MPaA]
    total_press_after_decompression = (
        next_case_inner_amount * 8.314 * T_K / case_inner_volume / 1e6
    )
    # 減圧後CO2モル量 [mol]
    mw_co2_after_decompression = now_amt_co2 + desorp_mw - outflow_amt_co2
    # 減圧後N2モル量 [mol]
    mw_n2_after_decompression = now_amt_n2 - outflow_amt_n2
    # 減圧後CO2モル分率
    mf_co2_after_decompression = mw_co2_after_decompression / (mw_co2_after_decompression + mw_n2_after_decompression)
    # 減圧後N2モル分率
    mf_n2_after_decompression = mw_n2_after_decompression / (mw_co2_after_decompression + mw_n2_after_decompression)

    # 出力
    output = {
        # "total_press_after_decompression": total_press_after_decompression, # 減圧後圧力 [MPaA]
        "mf_co2_after_decompression": mf_co2_after_decompression, # 減圧後CO2モル分率
        "mf_n2_after_decompression": mf_n2_after_decompression, # 減圧後N2モル分率
    }

    return output

def mf_after_vaccume_vaccume(sim_conds, vaccume_output, mb_dict):
    """ 排気後モル分率の計算

    Args:
        vaccume_output (dict): 排気後圧力計算の出力
        mb_dict (dict): マテバラ計算の出力

    Returns:
        dict: 排気後モル分率
    """
    # 脱着した気相放出CO2モル量 [mol]
    sum_desorp_mw = sum([
        mb_dict[stream][section]["desorp_mw_co2"] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"]) \
                                                    for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    # 脱着後気相CO2モル量 [mol]
    mw_co2_after_vaccume = sum_desorp_mw + vaccume_output["mw_co2_after_vaccume"]
    mw_co2_after_vaccume = max(0, mw_co2_after_vaccume)
    # 脱着後気相N2モル量 [mol]
    mw_n2_after_vaccume = vaccume_output["mw_n2_after_vaccume"]
    mw_n2_after_vaccume = max(0, mw_n2_after_vaccume)
    # 脱着後気相CO2モル分率
    try:
        mf_co2_after_vaccume = mw_co2_after_vaccume / (mw_co2_after_vaccume + mw_n2_after_vaccume)
    except ZeroDivisionError:
        mf_co2_after_vaccume = 1
    # 脱着後気相N2モル分率
    try:
        mf_n2_after_vaccume = mw_n2_after_vaccume / (mw_co2_after_vaccume + mw_n2_after_vaccume)
    except ZeroDivisionError:
        mf_n2_after_vaccume = 0

    output = {
        "sum_desorp_mw": sum_desorp_mw, # 脱着した気相放出CO2モル量
        "mf_co2_after_vaccume": mf_co2_after_vaccume, # 脱着後気相CO2モル分率
        "mf_n2_after_vaccume": mf_n2_after_vaccume, # 脱着後気相N2モル分率
    }

    return output

def total_press_after_batch_adsorp(sim_conds, variables, equalization_mode=False, upstream_flow_amount=None):
    """ バッチ吸着における圧力変化

    Args:
        sim_conds (dict): 実験条件
        variables (dict): 状態変数

    Return:
        float: バッチ吸着後の全圧
    """
    # 気体定数 [J/K/mol]
    R = 8.314
    # 平均温度 [K]
    temp_mean = []
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        for section in range(1, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            temp_mean.append(variables["temp"][stream][section])
    temp_mean = np.mean(temp_mean)
    temp_mean += 273.15
    # 空間体積（配管含む）
    V = sim_conds["PACKED_BED_COND"]["v_space"] + sim_conds["PACKED_BED_COND"]["v_upstream"]
    # ノルマル体積流量
    if equalization_mode: # バッチ均圧(下流): 上流側の均圧菅流量
        F = upstream_flow_amount
    else:
        F = sim_conds["INFLOW_GAS_COND"]["fr_all"] # バッチ吸着: 導入ガスの流量
    # 圧力変化量
    if equalization_mode: # バッチ均圧(下流): 計算ステップを小さくする
        diff_pressure = (
            R * temp_mean / V * (F / 22.4) * sim_conds["dt_eq"] / 1e6
        )
    else:
        diff_pressure = (
            R * temp_mean / V * (F / 22.4) * sim_conds["dt"] / 1e6
        )
    # 変化後全圧
    total_press_after_batch_adsorp = variables["total_press"] + diff_pressure

    return total_press_after_batch_adsorp

def _heat_transfer_coef(sim_conds, stream_conds, stream, temp_now, mode, variables,
                                material_output, vaccume_output=None):
    """ 層伝熱係数、壁-層伝熱係数を算出する

    Args:
        stream (int): 対象のストリーム番号
        temp_now (float): 対象セルの現在温度
        mode (str): 吸着・脱着等の運転モード
        material_output (dict): 対象セルのマテバラ計算結果
        vaccume_output (dict): 排気後圧力計算の出力（脱着時）

    Returns:
        float: 層伝熱係数、壁-層伝熱係数
    """
    # CoolProp用の前準備
    T_K = temp_now + 273.15 # 温度 [K]
    if mode == 0: # 吸着
        P = variables["total_press"] * 1e6 # 圧力 [Pa]
        mf_co2 = material_output["inflow_mf_co2"] # 流入ガスのCO2モル分率
        mf_n2 = material_output["inflow_mf_n2"] # 流入ガスのN2モル分率
    elif mode == 2: # 脱着
        P = vaccume_output["total_press_after_vaccume"] * 1e6 # 圧力 [Pa]
        mf_co2 = variables["mf_co2"] # 気相中のCO2モル分率
        mf_n2 = variables["mf_n2"] # 気相中のN2モル分率
    # 導入気体の熱伝導率 [W/m/K]
    kf = (
        CP.PropsSI('L', 'T', T_K, 'P', P, "co2") * mf_co2
        + CP.PropsSI('L', 'T', T_K, 'P', P, "nitrogen") * mf_n2
    ) / 1000
    # 充填剤の熱伝導率 [W/m/K]
    kp = sim_conds["PACKED_BED_COND"]["lambda_col"]
    # Yagi-Kunii式 1
    Phi_1 = 0.15
    # Yagi-Kunii式 2
    Phi_2 = 0.07
    # Yagi-Kunii式 3
    Phi = (
        Phi_2 + (Phi_1 - Phi_2)
        * (sim_conds["PACKED_BED_COND"]["epsilon"] - 0.26)
        / 0.26
    )
    # Yagi-Kunii式 4
    hrv = (
        (0.227 / (1 + sim_conds["PACKED_BED_COND"]["epsilon"] / 2
                    / (1 - sim_conds["PACKED_BED_COND"]["epsilon"])
                    * (1 - sim_conds["PACKED_BED_COND"]["epsilon_p"])
                    / sim_conds["PACKED_BED_COND"]["epsilon_p"]))
        * ((temp_now + 273.15) / 100)**3
    )
    # Yagi-Kunii式 5
    hrp = (
        0.227 * sim_conds["PACKED_BED_COND"]["epsilon_p"]
        / (2 - sim_conds["PACKED_BED_COND"]["epsilon_p"])
        * ((temp_now + 273.15) / 100)**3
    )
    # Yagi-Kunii式 6
    ksi = (
        1 / Phi + hrp * sim_conds["PACKED_BED_COND"]["dp"] / kf
    )
    # Yagi-Kunii式 7
    ke0_kf = (
        sim_conds["PACKED_BED_COND"]["epsilon"]
        * (1 + hrv * sim_conds["PACKED_BED_COND"]["dp"] / kf)
        + (1 - sim_conds["PACKED_BED_COND"]["epsilon"])
        / (1 / ksi + 2 * kf / 3 / kp)
    )
    # 静止充填層有効熱伝導率 [W/m/K]
    ke0 = kf * ke0_kf        
    # ストリーム換算直径 [m]
    d1 = 2 * (stream_conds[stream]["Sstream"] / math.pi)**0.5
    # 気体粘度 [Pas]
    mu = (
        CP.PropsSI('V', 'T', T_K, 'P', P, "co2") * mf_co2
        + CP.PropsSI('V', 'T', T_K, 'P', P, "nitrogen") * mf_n2
    )
    # プラントル数
    Pr = mu * 1000 * material_output["gas_cp"] / kf
    # 流入ガス体積流量 [m3/s]
    if mode == 0:
        f0 = (
            (material_output["inflow_fr_co2"] + material_output["inflow_fr_n2"])
            / 1e6 / (sim_conds["dt"] * 60)
        )
    elif mode == 2: # 脱着時は排気ガス体積流量 [m3/s]
        f0 = (
            vaccume_output["vacuum_rate"] / 60 * stream_conds[stream]["streamratio"]
        )
    # ストリーム空塔速度 [m/s]
    vcol = f0 / stream_conds[stream]["Sstream"]
    # 気体動粘度 [m2/s]
    nu = mu / material_output["gas_density"]
    # 粒子レイノルズ数
    Rep = vcol * sim_conds["PACKED_BED_COND"]["dp"] / nu
    # 充填層有効熱伝導率 1
    psi_beta = (
        1.0985 * (sim_conds["PACKED_BED_COND"]["dp"] / d1)**2
        - 0.5192 * (sim_conds["PACKED_BED_COND"]["dp"] / d1) + 0.1324
    )
    # 充填層有効熱伝導率 2
    ke_kf = ke0 / kf + psi_beta * Pr * Rep
    # 充填層有効熱伝導率 3 [W/m/K]
    ke = ke_kf * kf
    # ヌッセルト数
    Nup = 0.84 * Rep
    # 粒子‐流体間熱伝達率 [W/m2/K]
    habs = Nup / sim_conds["PACKED_BED_COND"]["dp"] * kf
    # 隙間係数
    a = 2
    # 格子長さ [m]
    l0 = sim_conds["PACKED_BED_COND"]["dp"] * a * 2 / 2**0.5
    # 粒子代表長さ [m]
    dlat = l0 * (1 - sim_conds["PACKED_BED_COND"]["epsilon"])
    # 代表長さ（セクション全長）1
    c0 = 4
    # 代表長さ（セクション全長）2
    Lambda_2 = c0 * Pr ** (1/3) * Rep ** (1/2)
    # 代表長さ（セクション全長）3
    knew = ke + 1 / (1 / (0.02 * Pr * Rep) + 2 / Lambda_2)
    # 代表長さ（セクション全長）4
    Lambda_1 = 2 / (kf / ke - kf / knew)
    # 代表長さ（セクション全長）5
    b0 = (
        0.5 * Lambda_1 * d1 / sim_conds["PACKED_BED_COND"]["dp"]
        * kf / ke
    )
    # 代表長さ（セクション全長）6
    Phi_b = 0.0775 * np.log(b0)+0.028
    # 代表長さ（セクション全長）7
    a12 = 0.9107 * np.log(b0) + 2.2395
    # 代表長さ（セクション全長）8 [m]
    Lp = sim_conds["PACKED_BED_COND"]["Lbed"] / sim_conds["CELL_SPLIT"]["num_sec"]
    # 粒子層-壁面伝熱ヌッセルト数 1
    y0 = (
        4 * sim_conds["PACKED_BED_COND"]["dp"]
        / d1 * Lp / d1 * ke_kf / (Pr * Rep)
    )
    # 粒子層-壁面伝熱ヌッセルト数 1
    Nupw = (
        sim_conds["PACKED_BED_COND"]["dp"]
        / d1 * ke_kf * (a12 + Phi_b / y0)
    )
    # 壁-層伝熱係数 [W/m2/K]
    hw1 = Nupw / sim_conds["PACKED_BED_COND"]["dp"] * kf
    hw1 *= sim_conds["DRUM_WALL_COND"]["coef_hw1"]
    # 層伝熱係数 [W/m2/K]
    u1 = 1 / (dlat / ke + 1 / habs)

    return hw1, u1

def _equilibrium_adsorp_amt(P, T):
    """ 平衡吸着量を計算

    Args:
        P (float): 圧力 [kPaA]
        T (float): 温度 [K]
    Returns:
        平衡吸着量
    """
    # 吸着等温式（シンボリック回帰による近似式）
    adsorp_amt_equilibrium = (
        P * (252.0724 - 0.50989705 * T)
        / (P - 3554.54819062669*(1 - 0.0655247236249063 * np.sqrt(T))**3 + 1.7354268)
    )
    return adsorp_amt_equilibrium

def __optimize_temp_reached(temp_reached, sim_conds, stream_conds,
                            gas_cp, Mgas, temp_now,
                            Habs, Hwin, Hwout, Hbb, Hroof,
                            stream):
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
        sim_conds["PACKED_BED_COND"]["Cbed"] * stream_conds[stream]["streamratio"]
        / sim_conds["CELL_SPLIT"]["num_sec"] * (temp_reached - temp_now)
    )
    # 充填層が受け取る熱(熱収支基準) [J]
    Hbed_heat_blc = Habs - Hgas + Hwin - Hwout - Hbb - Hroof
    return Hbed_heat_blc - Hbed_time

def __optimize_temp_reached_wall(temp_reached, sim_conds, stream_conds,
                                 temp_now, Hwin, Hwout, Hbb, Hroof,
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
        sim_conds["DRUM_WALL_COND"]["sh_drumw"]
        * stream_conds[1+sim_conds["CELL_SPLIT"]["num_str"]]["Mwall"]
        * (temp_reached - temp_now)
    )

    return Hwall_heat_blc - Hwall_time

def __optimize_temp_reached_lid(temp_reached, sim_conds, 
                                temp_now, Hlidall_heat, position
                                ):
    """上下蓋の到達温度算出におけるソルバー用関数

    Args:
        temp_reached (float): セクション到達温度
        args (dict): 充填層が受ける熱を計算するためのパラメータ

    Returns:
        float: 充填層が受ける熱の熱収支基準と時間基準の差分
    """
    # 壁が受け取る熱(ΔT基準) [J]
    Hlid_time = (
        sim_conds["DRUM_WALL_COND"]["sh_drumw"] * (temp_reached - temp_now)
    )
    if position == "up":
        Hlid_time *= sim_conds["LID_COND"]["UP"]["Mflange"]
    else:
        Hlid_time *= sim_conds["LID_COND"]["DOWN"]["Mflange"]

    return Hlidall_heat - Hlid_time