import numpy as np
import math
import CoolProp.CoolProp as CP


def add_sim_conds(sim_conds):
    """ 全体共通パラメータの追加定義

        Args:
            sim_conds (dict): 全体共通パラメータ群
        Return:
            sim_conds (dict): 追加後の全体共通パラメータ群
    """
    # 触媒充填層条件
    packed_bed = sim_conds["PACKED_BED_COND"]
    packed_bed["Rbed"] = packed_bed["Dbed"] / 2
    packed_bed["Sbed"] = math.pi*packed_bed["Rbed"] ** 2
    packed_bed["Vbed"] = packed_bed["Sbed"] * packed_bed["Lbed"]
    packed_bed["rho_abs"] = packed_bed["Mabs"] / packed_bed["Vbed"] / 1e6
    packed_bed["Cbed"] = packed_bed["cabs"] * packed_bed["Mabs"]
    packed_bed["v_space"] = packed_bed["Lbed"] * packed_bed["Sbed"] * packed_bed["epsilon"]

    # 容器壁条件
    drum_wall = sim_conds["DRUM_WALL_COND"]
    drum_wall["Rdrum"] = drum_wall["Ddrum"] / 2
    drum_wall["Ldrum"] = packed_bed["Lbed"]
    drum_wall["Tdrumw"] = drum_wall["Rdrum"] - packed_bed["Rbed"]
    drum_wall["Sdrumw"] = math.pi * (drum_wall["Rdrum"]**2 - packed_bed["Rbed"]**2)
    drum_wall["Vdrumw"] = drum_wall["Sdrumw"] * drum_wall["Ldrum"]
    drum_wall["Wdrumw"] = drum_wall["Dense_drumw"] * drum_wall["Vdrumw"] * 1e6
    drum_wall["la_drum"] = math.pi * drum_wall["Ddrum"] * drum_wall["Ldrum"]

    # 上下蓋条件
    lid_cond = sim_conds["LID_COND"]
    for x in ["UP", "DOWN"]:
        lid_cond[x]["Sflange_up"] = (
            ((lid_cond[x]["Dflange"] / 2) ** 2 - (lid_cond[x]["PDflange_up"] / 2) ** 2)
            * math.pi / 1000000
        )
        lid_cond[x]["Vflange_up"] = (
            ((lid_cond[x]["Dflange"] / 2) ** 2 - (lid_cond[x]["PDflange_up"] / 2) ** 2)
            * math.pi * lid_cond[x]["Tflange"] / 1000
        )
        lid_cond[x]["Vflange_dw"] = (
            ((lid_cond[x]["Dflange"] / 2) ** 2 - (lid_cond[x]["PDflange_dw"] / 2) ** 2)
            * math.pi * lid_cond[x]["Tflange"] / 1000
        )
        lid_cond[x]["Mflange"] = (
            (lid_cond[x]["Vflange_up"] + lid_cond[x]["Vflange_dw"])
            * drum_wall["Dense_drumw"]
        )

    # 導入ガス条件
    input_gass = sim_conds["INFLOW_GAS_COND"]
    input_gass["fr_all"] = input_gass["fr_co2"] + input_gass["fr_n2"]
    input_gass["mf_co2"] = input_gass["fr_co2"] / input_gass["fr_all"]
    input_gass["mf_n2"] = input_gass["fr_n2"] / input_gass["fr_all"]
    input_gass["dense_mean"] = (
        input_gass["dense_co2"] * input_gass["mf_co2"]
        + input_gass["dense_n2"] * input_gass["mf_n2"]
    )
    input_gass["c_mean"] = (
        input_gass["c_co2"] * input_gass["mf_co2"]
        + input_gass["c_n2"] * input_gass["mf_n2"]
    )
    input_gass["vi_mean"] = (
        input_gass["vi_co2"] * input_gass["mf_co2"]
        + input_gass["vi_n2"] * input_gass["mf_n2"]
    )
    input_gass["cp_mean"] = (
        input_gass["cp_co2"] * input_gass["mf_co2"]
        + input_gass["cp_n2"] * input_gass["mf_n2"]
    )
    a1 = input_gass["fr_co2"] / 22.4 * input_gass["mw_co2"] * 60 / 1000
    a2 = input_gass["fr_n2"] / 22.4 * input_gass["mw_n2"] * 60 / 1000
    a3 = a1 + a2
    input_gass["C_per_hour"] = input_gass["cp_mean"] * a3

    # 均圧配管条件
    press_equal = sim_conds["PRESS_EQUAL_PIPE_COND"]
    press_equal["Spipe"] = np.pi * press_equal["Dpipe"]**2 / 4
    press_equal["Vpipe"] = press_equal["Spipe"] * press_equal["Dpipe"]

    return sim_conds


def update_params_by_obs(sim_conds):
    """ 観測値によって更新されたパラメータに紐づくパラメータの更新

    Args:
        sim_conds (dict): 実験パラメータ

    Returns:
        dict: 一部更新後の実験パラメータ
    """
    # 導入ガス条件
    input_gass = sim_conds["INFLOW_GAS_COND"]
    input_gass["fr_all"] = input_gass["fr_co2"] + input_gass["fr_n2"]
    if input_gass["fr_all"] != 0:
        input_gass["mf_co2"] = input_gass["fr_co2"] / input_gass["fr_all"]
        input_gass["mf_n2"] = input_gass["fr_n2"] / input_gass["fr_all"]
    else:
        input_gass["mf_co2"] = 0
        input_gass["mf_n2"] = 0

    # CropProp関連の物性
    T_K = input_gass["temp"] + 273.15 # 温度 [K]
    P = input_gass["total_press"] * 1e6 # 圧力 [Pa]
    # CO2密度 [kg/m3]
    input_gass["dense_co2"] = CP.PropsSI('D', 'T', T_K, 'P', P, "co2")
    # N2密度 [kg/m3]
    input_gass["dense_n2"] = CP.PropsSI('D', 'T', T_K, 'P', P, "nitrogen")
    # CO2熱伝導率 [W/m/K]
    input_gass["c_co2"] = CP.PropsSI('L', 'T', T_K, 'P', P, "co2")
    # N2熱伝導率 [W/m/K]
    input_gass["c_n2"] = CP.PropsSI('L', 'T', T_K, 'P', P, "nitrogen")
    # CO2粘度 [Pa s]
    input_gass["vi_co2"] = CP.PropsSI('V', 'T', T_K, 'P', P, "co2") * 1e6
    # N2粘度 [Pa s]
    input_gass["vi_n2"] = CP.PropsSI('V', 'T', T_K, 'P', P, "nitrogen") * 1e6
    # CO2比熱容量 [kJ/kg/K]
    input_gass["cp_co2"] = CP.PropsSI('C', 'T', T_K, 'P', P, "co2") / 1000
    # N2比熱容量 [kJ/kg/K]
    input_gass["cp_n2"] = CP.PropsSI('C', 'T', T_K, 'P', P, "nitrogen") / 1000

    input_gass["dense_mean"] = (
        input_gass["dense_co2"] * input_gass["mf_co2"]
        + input_gass["dense_n2"] * input_gass["mf_n2"]
    )
    input_gass["c_mean"] = (
        input_gass["c_co2"] * input_gass["mf_co2"]
        + input_gass["c_n2"] * input_gass["mf_n2"]
    )
    input_gass["vi_mean"] = (
        input_gass["vi_co2"] * input_gass["mf_co2"]
        + input_gass["vi_n2"] * input_gass["mf_n2"]
    )
    input_gass["cp_mean"] = (
        input_gass["cp_co2"] * input_gass["mf_co2"]
        + input_gass["cp_n2"] * input_gass["mf_n2"]
    )
    a1 = input_gass["fr_co2"] / 22.4 * input_gass["mw_co2"] * 60 / 1000
    a2 = input_gass["fr_n2"] / 22.4 * input_gass["mw_n2"] * 60 / 1000
    a3 = a1 + a2
    input_gass["C_per_hour"] = input_gass["cp_mean"] * a3

    return sim_conds


def init_stream_conds(sim_conds, stream, stream_conds):
    """ 全体条件から各ストリーム条件を算出する

        Args:
            sim_conds (dict): 全体共通パラメータ群
            streams (int): 対象のストリーム番号
            stream_conds (dict): 各ストリーム条件
        Return:
            sim_conds (dict): 追加後の全体共通パラメータ群
    """
    tgt_stream_conds = {}
    # 内側境界半径座標(軸中心0)
    if stream == 1:
        tgt_stream_conds["r_in"] = 0
    else:
        tgt_stream_conds["r_in"] = stream_conds[stream-1]["r_out"]
    # 外側境界半径座標
    if stream == 1:
        tgt_stream_conds["r_out"] = (
            sim_conds["PACKED_BED_COND"]["Rbed"] / sim_conds["CELL_SPLIT"]["num_str"]
            * stream + tgt_stream_conds["r_in"]
        )
    else:
        tgt_stream_conds["r_out"] = (
            sim_conds["PACKED_BED_COND"]["Rbed"] / sim_conds["CELL_SPLIT"]["num_str"]
            * stream + stream_conds[stream-1]["r_in"]
        )
    # ストリーム断面積
    tgt_stream_conds["Sstream"] = math.pi * (tgt_stream_conds["r_out"]**2
                                             - tgt_stream_conds["r_in"]**2)
    # ストリーム分配割合
    tgt_stream_conds["streamratio"] = (
        tgt_stream_conds["Sstream"] / sim_conds["PACKED_BED_COND"]["Sbed"]
    )
    # 内側境界周長
    tgt_stream_conds["Circ_in"] = 2 * math.pi * tgt_stream_conds["r_in"]
    # 内側境界面積
    tgt_stream_conds["Ain"] = tgt_stream_conds["Circ_in"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    # 外側境界周長
    tgt_stream_conds["Circ_out"] = 2 * math.pi * tgt_stream_conds["r_out"]
    # 外側境界面積
    tgt_stream_conds["Aout"] = tgt_stream_conds["Circ_out"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    # ストリーム吸着材量
    tgt_stream_conds["Mabs"] = sim_conds["PACKED_BED_COND"]["Mabs"] * tgt_stream_conds["streamratio"]

    return tgt_stream_conds


def init_drum_wall_conds(sim_conds, stream_conds):
    """ 全体条件から各ストリーム条件を算出する

        Args:
            sim_conds (dict): 全体共通パラメータ群
            stream_conds (dict): 各ストリーム条件
        Return:
            sim_conds (dict): 追加後の全体共通パラメータ群
    """
    tgt_stream_conds = {}
    # 内側境界半径座標(軸中心0)
    tgt_stream_conds["r_in"] = stream_conds[sim_conds["CELL_SPLIT"]["num_str"]]["r_out"]
    # 外側境界半径座標
    tgt_stream_conds["r_out"] = sim_conds["DRUM_WALL_COND"]["Rdrum"]
    # ストリーム断面積
    tgt_stream_conds["Sstream"] = math.pi * (tgt_stream_conds["r_out"]**2
                                             - tgt_stream_conds["r_in"]**2)
    # 内側境界周長
    tgt_stream_conds["Circ_in"] = 2 * math.pi * tgt_stream_conds["r_in"]
    # 内側境界面積
    tgt_stream_conds["Ain"] = tgt_stream_conds["Circ_in"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    # 外側境界周長
    tgt_stream_conds["Circ_out"] = 2 * math.pi * tgt_stream_conds["r_out"]
    # 外側境界面積
    tgt_stream_conds["Aout"] = tgt_stream_conds["Circ_out"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    # 容器壁質量
    tgt_stream_conds["Mwall"] = sim_conds["DRUM_WALL_COND"]["Wdrumw"]

    return tgt_stream_conds