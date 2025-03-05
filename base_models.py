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


def material_balance_adsorp(sim_conds, stream_conds, stream, section, variables,
                            inflow_gas=None, flow_amt_depress=None):
    """ 任意セルのマテリアルバランスを計算する
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

    # セクション吸着材量 [g]
    Mabs = stream_conds[stream]["Mabs"] / sim_conds["CELL_SPLIT"]["num_sec"]
    # 流入CO2, N2流量 [cm3]
    if (section == 1) & (inflow_gas is None) & (flow_amt_depress is None): # 最上流セルの流入ガスによる吸着など
        inflow_fr_co2 = (
            sim_conds["INFLOW_GAS_COND"]["fr_co2"] * sim_conds["dt"]
            * stream_conds[stream]["streamratio"] * 1000
        )
        inflow_fr_n2 = (
            sim_conds["INFLOW_GAS_COND"]["fr_n2"] * sim_conds["dt"]
            * stream_conds[stream]["streamratio"] * 1000
        )
    elif (section == 1) & (flow_amt_depress is not None): # 減圧時の最上流セルのみ対象
        inflow_fr_co2 = (
            sim_conds["INFLOW_GAS_COND"]["mf_co2"] * flow_amt_depress * sim_conds["dt"]
            * stream_conds[stream]["streamratio"] * 1000
        )
        inflow_fr_n2 = (
            sim_conds["INFLOW_GAS_COND"]["mf_n2"] * flow_amt_depress * sim_conds["dt"]
            * stream_conds[stream]["streamratio"] * 1000
        )
    elif inflow_gas is not None : # 下流セクションや下流塔での吸着など
        inflow_fr_co2 = inflow_gas["outflow_fr_co2"]
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
            # セクション理論新規吸着量 [cm3]
            adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
            # 実際のセクション新規吸着量 [cm3]
            adsorp_amt_estimate = min(adsorp_amt_estimate, inflow_fr_co2)
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
    else:
        adsorp_amt_estimate = 0
        adsorp_amt_estimate_abs = 0
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
        "p_co2": p_co2,
    }

    return output

def material_balance_desorp(sim_conds, stream_conds, stream, section, variables, vacuum_output):
    """ 任意セルのマテリアルバランスを計算する
        脱着モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vacuum_output (dict): 排気後圧力計算の出力

    Returns:
        dict: 対象セルの計算結果
    """
    ### 現在気相モル量 = 流入モル量の計算 -------------------
    # セクション空間割合
    space_ratio_section = (
        stream_conds[stream]["streamratio"] / sim_conds["CELL_SPLIT"]["num_sec"]
        * sim_conds["PACKED_BED_COND"]["v_space"]
        / sim_conds["VACUUMING_PIPE_COND"]["Vspace"]
    )
    # セクション空間現在物質量 [mol]
    mol_amt_section = (
        vacuum_output["case_inner_mol_amt_after_vacuum"] * space_ratio_section
    )
    # 現在気相モル量 [mol]
    inflow_fr_co2 = mol_amt_section * variables["mf_co2"][stream][section]
    inflow_fr_n2 = mol_amt_section * variables["mf_n2"][stream][section]
    # 現在気相ノルマル体積(=流入量) [cm3]
    inflow_fr_co2 *= 22.4 * 1000
    inflow_fr_n2 *= 22.4 * 1000

    ### 気相放出後モル量の計算 -----------------------------    
    # 現在温度 [℃]
    T_K = variables["temp"][stream][section] + 273.15
    # モル分率
    mf_co2 = variables["mf_co2"][stream][section]
    mf_n2 = variables["mf_n2"][stream][section]
    # CO2分圧 [MPaA]
    p_co2 = vacuum_output["total_press_after_vacuum"] * mf_co2
    # セクション吸着材量 [g]
    Mabs = stream_conds[stream]["Mabs"] / sim_conds["CELL_SPLIT"]["num_sec"]
    # 現在雰囲気の平衡吸着量 [cm3/g-abs]
    P_KPA = p_co2 * 1000 # [MPaA] → [kPaA]
    adsorp_amt_equilibrium = _equilibrium_adsorp_amt(P_KPA, T_K)
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
        # セクション理論新規吸着量 [cm3]
        adsorp_amt_estimate = adsorp_amt_estimate_abs * Mabs
        # 実際のセクション新規吸着量 [cm3]
        adsorp_amt_estimate = min(adsorp_amt_estimate, inflow_fr_co2)
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
    # 気相放出後モル量 [mol]
    desorp_mw_co2_after_vacuum = inflow_fr_co2 + desorp_mw_co2
    desorp_mw_n2_after_vacuum = inflow_fr_n2
    # 気相放出後全物質量 [mol]
    desorp_mw_all_after_vacuum = desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum
    # 気相放出後モル分率
    desorp_mf_co2_after_vacuum = (
        desorp_mw_co2_after_vacuum
        / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)
    )
    desorp_mf_n2_after_vacuum = (
        desorp_mw_n2_after_vacuum
        / (desorp_mw_co2_after_vacuum + desorp_mw_n2_after_vacuum)
    )

    ### その他（熱バラ渡す用） ---------------------------------------
    P = vacuum_output["total_press_after_vacuum"] * 1e6
    # ガス密度 [kg/m3]
    gas_density = (
        CP.PropsSI('D', 'T', T_K, 'P', P, "co2") * mf_co2
        + CP.PropsSI('D', 'T', T_K, 'P', P, "nitrogen") * mf_n2
    )
    # ガス比熱 [kJ/kg/K]
    gas_cp = (
        CP.PropsSI('CPMASS', 'T', T_K, 'P', P, "co2") * mf_co2
        + CP.PropsSI('CPMASS', 'T', T_K, 'P', P, "nitrogen") * mf_n2
    ) * 1e-3

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
        "p_co2": p_co2,
    }

    # 出力２（モル分率）
    output2 = {
        "mf_co2_after_vacuum": desorp_mf_co2_after_vacuum, # 気相放出後CO2モル分率
        "mf_n2_after_vacuum": desorp_mf_n2_after_vacuum, # 気相放出後N2モル分率
        "desorp_mw_all_after_vacuum": desorp_mw_all_after_vacuum, # 気相放出後物質量 [mol]
    }

    return output, output2

def material_balance_valve_stop(stream, section, variables):
    """ 任意セルのマテリアルバランスを計算する
        弁停止モード

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 温度等の状態変数
        vacuum_output (dict): 排気後圧力計算の出力

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
        "p_co2": 0,
    }

    return output

def heat_balance(sim_conds, stream_conds, stream, section, variables, mode,
                    material_output=None, heat_output=None, vacuum_output=None):
    """ 対象セルの熱バランスを計算する

    Args:
        stream (int): 対象セルのstream番号
        section (int): 対象セルのsection番号
        variables (dict): 状態変数
        mode (int): 吸着・脱着等の運転モード
        material_output (dict): 対象セルのマテバラ出力
        heat_output (dict): 上流セクションの熱バラ出力
        vacuum_output (dict): 排気後圧力計算の出力（脱着時）

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
                                      section,
                                      temp_now,
                                      mode,
                                      variables,
                                      material_output)
    elif mode == 1: # 弁停止モードでは直前値に置換
        hw1 = variables["heat_t_coef"][stream][section]
        u1 = variables["heat_t_coef_wall"][stream][section]
    elif mode == 2: # 脱着モードでは入力に排気後圧力計算の出力を使用
        # hw1 = variables["heat_t_coef"][stream][section]
        # u1 = variables["heat_t_coef_wall"][stream][section]
        hw1, u1 = _heat_transfer_coef(sim_conds,
                                      stream_conds,
                                      stream,
                                      section,
                                      temp_now,
                                      mode,
                                      variables,
                                      material_output,
                                      vacuum_output)

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

def total_press_after_vacuum(sim_conds, variables):
    """ 排気後圧力とCO2回収濃度の計算
        真空脱着モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 圧損, 積算CO2・N2回収量, CO2回収濃度, 排気後圧力, 
    """
    ### 前準備 --------------------------------------
    # 容器内平均温度 [K]
    T_K = np.mean([
        variables["temp"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                           for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ]) + 273.15
    # 全圧 [PaA]
    P = variables["total_press"] * 1e6 # [MPaA]→[Pa]
    # 平均co2分率
    _mean_mf_co2 = np.mean([
        variables["mf_co2"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                             for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    # 平均n2分率
    _mean_mf_n2 = np.mean([
        variables["mf_n2"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                            for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    # 真空ポンプ排気ガス粘度 [Pa・s]
    mu = (
        CP.PropsSI('V', 'T', T_K, 'P', P, "co2") * _mean_mf_co2
        + CP.PropsSI('V', 'T', T_K, 'P', P, "nitrogen") * _mean_mf_n2
    )
    # 真空ポンプ排気ガス密度 [kg/m3]
    rho = (
        CP.PropsSI('D', 'T', T_K, 'P', P, "co2") * _mean_mf_co2
        + CP.PropsSI('D', 'T', T_K, 'P', P, "nitrogen") * _mean_mf_n2
    )

    ### 圧損計算 --------------------------------------
    _max_iteration = 1000
    P_resist = 0
    tolerance = 1e-6
    for iter in range(_max_iteration):
        P_resist_old = P_resist
        # ポンプ見せかけの全圧 [PaA]
        P_PUMP= (variables["total_press"] - P_resist) * 1e6
        # 真空ポンプ排気速度 [m3/min]
        if P_PUMP < 100:
            vacuum_rate = 0
        elif (P_PUMP >= 100) & (P_PUMP < 900):
            vacuum_rate = 0.085 * np.log(P_PUMP)
        else:
            vacuum_rate = 0.57
        # 真空ポンプ排気ノルマル流量 [m3/min]
        vacuum_rate_N = vacuum_rate / 0.1013 * P_PUMP * 1e-6
        # 真空ポンプ排気線流速 [m/3]
        linear_velocity = vacuum_rate / sim_conds["VACUUMING_PIPE_COND"]["Spipe"]
        # 真空ポンプ排気レイノルズ数
        Re = (
            rho * linear_velocity * sim_conds["VACUUMING_PIPE_COND"]["Dpipe"] / mu
        )
        # 真空ポンプ排気管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        # 均圧配管圧力損失 [MPaA]
        P_resist = (
            lambda_f * sim_conds["VACUUMING_PIPE_COND"]["Lpipe"]
            / sim_conds["VACUUMING_PIPE_COND"]["Dpipe"] * linear_velocity ** 2 / (2*9.81)
        ) * 1e-6
        # 収束判定
        if np.abs(P_resist - P_resist_old) < tolerance:
            break
        if pd.isna(P_resist):
            break
    if iter == _max_iteration-1:
        print("収束せず: 見せかけの全圧 =", np.abs(P_resist - P_resist_old))
    
    ### CO2回収濃度計算 --------------------------------------
    # 排気速度 [mol/min]
    vacuum_rate_mol = P_PUMP * vacuum_rate_N / 8.314 / T_K
    # 排気量 [mol]
    vacuum_amt = vacuum_rate_mol * sim_conds["dt"]
    # 排気CO2量 [mol]
    vacuum_amt_co2 = vacuum_amt * _mean_mf_co2
    # 排気N2量 [mol]
    vacuum_amt_n2 = vacuum_amt * _mean_mf_n2
    # 積算排気CO2量 [mol]
    accum_vacuum_amt_co2 = variables["vacuum_amt_co2"] + vacuum_amt_co2
    # 積算排気N2量 [mol]
    accum_vacuum_amt_n2 = variables["vacuum_amt_n2"] + vacuum_amt_n2
    # CO2回収濃度 [%]
    vacuum_co2_mf = (
        accum_vacuum_amt_co2 / (accum_vacuum_amt_co2 + accum_vacuum_amt_n2)
    ) * 100

    ### 排気後圧力計算 --------------------------------
    # 排気"前"の真空排気空間の現在物質量 [mol]
    case_inner_mol_amt = (
        P_PUMP * sim_conds["VACUUMING_PIPE_COND"]["Vspace"]
        / 8.314 / T_K
    )
    # 排気"後"の現在物質量 [mol]
    case_inner_mol_amt_after_vacuum = (
        case_inner_mol_amt - vacuum_amt
    )
    # 排気"後"の容器内部圧力 [MPaA]
    total_press_after_vacuum = (
        case_inner_mol_amt_after_vacuum * 8.314 * T_K
        / sim_conds["VACUUMING_PIPE_COND"]["Vspace"] * 1e-6
    )

    # 出力
    output = {
        "P_resist": P_resist, # 圧損 [MPaA]
        "accum_vacuum_amt_co2": accum_vacuum_amt_co2, # 積算排気CO2量 [mol]
        "accum_vacuum_amt_n2": accum_vacuum_amt_n2, # 積算排気N2量 [mol]
        "vacuum_co2_mf": vacuum_co2_mf, # CO2回収濃度 [%]
        "vacuum_rate_N": vacuum_rate_N, # 真空ポンプ排気速度 [m3/min]
        "case_inner_mol_amt_after_vacuum": case_inner_mol_amt_after_vacuum, # 排気内部物質量 [mol]
        "total_press_after_vacuum": total_press_after_vacuum, # 排気後圧力 [MPaA]
    }

    return output

def total_press_after_depressure(sim_conds, variables, downflow_total_press):
    """ 減圧時の上流からの均圧配管流量計算
        バッチ均圧(上流側)モード

    Args:
        variables (dict): 状態変数

    Returns:
        dict: 排気後圧力と排気後CO2・N2モル量
    """
    ### 上流均圧配管流量の計算 ----------------------------------------
    # 容器内平均温度 [℃]
    T_K = np.mean([
        variables["temp"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                           for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ]) + 273.15
    # 全圧 [PaA]
    P = variables["total_press"] * 1e6 # [MPaA]→[Pa]
    # 平均co2分率
    _mean_mf_co2 = np.mean([
        variables["mf_co2"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                             for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    # 平均n2分率
    _mean_mf_n2 = np.mean([
        variables["mf_n2"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                            for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ])
    # 上流均圧管ガス粘度 [Pa・s]
    mu = (
        CP.PropsSI('V', 'T', T_K, 'P', P, "co2") * _mean_mf_co2
        + CP.PropsSI('V', 'T', T_K, 'P', P, "nitrogen") * _mean_mf_n2
    )
    # 上流均圧管ガス密度 [kg/m3]
    rho = (
        CP.PropsSI('D', 'T', T_K, 'P', P, "co2") * _mean_mf_co2
        + CP.PropsSI('D', 'T', T_K, 'P', P, "nitrogen") * _mean_mf_n2
    )
    # 上流均圧配管圧力損失 [MPaA]
    _max_iteration = 1000
    P_resist = 0
    tolerance = 1e-6
    for iter in range(_max_iteration):
        P_resist_old = P_resist
        # 塔間の圧力差 [PaA]
        dP = (variables["total_press"] - downflow_total_press - P_resist) * 1e6
        if np.abs(dP) < 1:
            dP = 0
        # 配管流速 [m/s]
        flow_rate = (
            dP * sim_conds["PRESS_EQUAL_PIPE_COND"]["Dpipe"] ** 2
            / (32 * mu * sim_conds["PRESS_EQUAL_PIPE_COND"]["Lpipe"])
        )
        flow_rate = max(1e-8, flow_rate)
        # 均圧管レイノルズ数
        Re = (
            rho * abs(flow_rate) * sim_conds["PRESS_EQUAL_PIPE_COND"]["Dpipe"] / mu
        )
        # 管摩擦係数
        lambda_f = 64 / Re if Re != 0 else 0
        # 均圧配管圧力損失 [MPaA]
        P_resist = (
            lambda_f * sim_conds["PRESS_EQUAL_PIPE_COND"]["Lpipe"]
            / sim_conds["PRESS_EQUAL_PIPE_COND"]["Dpipe"] * flow_rate ** 2 / (2*9.81)
        ) * 1e-6
        # 収束判定
        if np.abs(P_resist - P_resist_old) < tolerance:
            break
        if pd.isna(P_resist):
            break
    if iter == _max_iteration-1:
        print("収束せず: 圧力差 =", np.abs(P_resist - P_resist_old))
    # 均圧配管流量 [m3/min]
    flow_amount_m3 = (
        sim_conds["PRESS_EQUAL_PIPE_COND"]["Spipe"] * flow_rate / 60
        * sim_conds["PRESS_EQUAL_PIPE_COND"]["coef_fr"]
    )
    # 均圧配管ノルマル流量 [m3/min]
    flow_amount_m3_N = (
        flow_amount_m3 * variables["total_press"] / 0.1013
    )
    # 均圧配管流量 [L/min] (下流塔への入力)
    flow_amount_l = flow_amount_m3_N * 1e3

    ### 次時刻の圧力計算 ----------------------------------------
    # 容器上流空間を移動する物質量 [mol]
    mw_upper_space = (
        flow_amount_m3_N * 1000 * sim_conds["dt"] / 22.4
    )
    # 上流側の合計体積 [m3]
    V_upper_tower = sim_conds["PACKED_BED_COND"]["v_upstream"] + sim_conds["PACKED_BED_COND"]["v_space"]
    # 上流容器圧力変化 [MPaA]
    dP_upper = (
        8.314 * T_K / V_upper_tower * mw_upper_space * 1e-6
    )
    # 次時刻の容器圧力 [MPaA]
    total_press_after_depressure = variables["total_press"] - dP_upper

    # 出力
    output = {
        "total_press_after_depressure": total_press_after_depressure, # 減圧後の全圧 [MPaA]
        "flow_amount_l": flow_amount_l, # 均圧配管流量 [L/min]
        "diff_press": dP, # 均圧塔の圧力差 [MPaA]
    }

    return output

def downflow_fr_after_depressure(sim_conds, stream_conds, variables, mb_dict, downflow_total_press):
    """ 減圧計算時に下流塔の圧力・流入量を計算する
        減圧モード

    Args:
        sim_conds (dict): 全体共通パラメータ
        stream_conds (dict): ストリーム内共通パラメータ
        variables (dict): 状態変数
        mb_dict (dict): マテバラ計算結果
        downflow_total_press (float): 下流塔の現在全圧

    Returns:
        float: 減圧後の下流塔の全圧 [MPaA]
        float: 下流塔への流出CO2流量 [cm3]
        float: 下流塔への流出N2流量 [cm3]
    """
    ### 下流塔の圧力計算 ----------------------------------------------
    # 容器内平均温度 [℃]
    T_K = np.mean([
        variables["temp"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                           for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ]) + 273.15
    # 下流流出量合計（最下流セクションの合計）[L]
    most_down_section = sim_conds["CELL_SPLIT"]["num_sec"]
    sum_outflow_fr = sum([
        mb_dict[stream][most_down_section]["outflow_fr_co2"] + mb_dict[stream][most_down_section]["outflow_fr_n2"]
        for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"])
    ]) / 1e3
    # 下流流出物質量 [mol]
    sum_outflow_mol = sum_outflow_fr / 22.4
    # 均圧下流側空間体積 [m3]
    V_downflow = sim_conds["PRESS_EQUAL_PIPE_COND"]["Vpipe"] + sim_conds["PACKED_BED_COND"]["v_space"]
    # 下流容器圧力変化 [MPaA]
    dP = (
        8.314 * T_K / V_downflow * sum_outflow_mol / 1e6
    )
    # 次時刻の下流容器全圧 [MPaA]
    total_press_after_depressure_downflow = downflow_total_press + dP

    ### 下流容器への流入量 --------------------------------------------
    # 下流塔への合計流出CO2流量 [cm3]
    sum_outflow_fr_co2 = sum([
        mb_dict[stream][most_down_section]["outflow_fr_co2"] for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"])
    ])
    # 下流塔への合計流出N2流量 [cm3]
    sum_outflow_fr_n2 = sum([
        mb_dict[stream][most_down_section]["outflow_fr_n2"] for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"])
    ])
    # 下流塔への流出CO2, N2流量 [cm3]
    outflow_fr = {}
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        outflow_fr[stream] = {}
        outflow_fr[stream]["outflow_fr_co2"] = sum_outflow_fr_co2 * stream_conds[stream]["streamratio"]
        outflow_fr[stream]["outflow_fr_n2"] = sum_outflow_fr_n2 * stream_conds[stream]["streamratio"]

    # 出力
    output = {
        "total_press_after_depressure_downflow": total_press_after_depressure_downflow, # 減圧後の下流塔の全圧 [MPaA]
        "outflow_fr": outflow_fr, # 下流塔への流出CO2, N2流量 [cm3]
    }

    return output

def total_press_after_desorp(sim_conds, variables, mf_dict):
    """ 気相放出後の全圧の計算

    Args:
        variables (dict): 状態変数
        mf_dict (dict): マテバラの出力結果（モル分率）

    Returns:
        dict: 気相放出後の全圧
    """
    # 容器内平均温度 [℃]
    T_K = np.mean([
        variables["temp"][stream][section] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"])\
                                           for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ]) + 273.15
    # 気相放出後の全物質量合計 [mol]
    sum_desorp_mw = sum([
        mf_dict[stream][section]["desorp_mw_all_after_vacuum"] for stream in range(1,1+sim_conds["CELL_SPLIT"]["num_str"]) 
                                                                for section in range(1,1+sim_conds["CELL_SPLIT"]["num_sec"])
    ]) / 22.4 / 1e3
    # 気相放出後の全圧 [MPaA]
    total_press_after_desorp = (
        sum_desorp_mw * 8.314 * T_K / sim_conds["VACUUMING_PIPE_COND"]["Vspace"] * 1e-6
    )

    return total_press_after_desorp

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
    # if equalization_mode: # バッチ均圧(下流): 計算ステップを小さくする
    #     diff_pressure = (
    #         R * temp_mean / V * (F / 22.4) * sim_conds["dt_eq"] / 1e6
    #     )
    # else:
    diff_pressure = (
        R * temp_mean / V * (F / 22.4) * sim_conds["dt"] / 1e6
    )
    # 変化後全圧
    total_press_after_batch_adsorp = variables["total_press"] + diff_pressure

    return total_press_after_batch_adsorp

def _heat_transfer_coef(sim_conds, stream_conds, stream, section, temp_now, mode, variables,
                        material_output, vacuum_output=None):
    """ 層伝熱係数、壁-層伝熱係数を算出する

    Args:
        stream (int): 対象のストリーム番号
        temp_now (float): 対象セルの現在温度
        mode (str): 吸着・脱着等の運転モード
        material_output (dict): 対象セルのマテバラ計算結果
        vacuum_output (dict): 排気後圧力計算の出力（脱着時）

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
        P = vacuum_output["total_press_after_vacuum"] * 1e6 # 圧力 [Pa]
        mf_co2 = variables["mf_co2"][stream][section] # 気相中のCO2モル分率
        mf_n2 = variables["mf_n2"][stream][section] # 気相中のN2モル分率
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
            vacuum_output["vacuum_rate_N"] / 60 * stream_conds[stream]["streamratio"]
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