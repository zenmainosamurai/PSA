import numpy as np
import math
import CoolProp.CoolProp as CP

def read_sim_conds(df_sim_conds):
    """ sim_conds.xlsxを辞書形式に変換する

    Args:
        df_sim_conds (pd.DataFrame): 実験条件のxlsxファイル
    """
    sim_conds = {1: {}, 2: {}, 3: {}}
    for tower_num in [1,2,3]:
        # 共通パラメータ
        sim_conds[tower_num]["dt"] = df_sim_conds["共通"].iloc[4,3]
        sim_conds[tower_num]["NUM_STR"] = int(df_sim_conds["共通"].iloc[5,3])
        sim_conds[tower_num]["NUM_SEC"] = int(df_sim_conds["共通"].iloc[6,3])
        # 触媒充填層条件
        sim_conds[tower_num]["PACKED_BED_COND"] = {}
        for idx in range(3,24):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,2]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,3]
            sim_conds[tower_num]["PACKED_BED_COND"][key] = val
        # 容器壁条件
        sim_conds[tower_num]["DRUM_WALL_COND"] = {}
        for idx in range(3,17):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,6]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,7]
            sim_conds[tower_num]["DRUM_WALL_COND"][key] = val
        # 上蓋の条件
        sim_conds[tower_num]["LID_COND"] = {}
        sim_conds[tower_num]["LID_COND"]["UP"] = {}
        for idx in range(3,11):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,10]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,11]
            sim_conds[tower_num]["LID_COND"]["UP"][key] = val
        # 下蓋の条件
        sim_conds[tower_num]["LID_COND"]["DOWN"] = {}
        for idx in range(13,21):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,10]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,11]
            sim_conds[tower_num]["LID_COND"]["DOWN"][key] = val
        # 導入ガス条件
        sim_conds[tower_num]["INFLOW_GAS_COND"] = {}
        for idx in range(3,28):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,14]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,15]
            sim_conds[tower_num]["INFLOW_GAS_COND"][key] = val
        # 均圧配管条件
        sim_conds[tower_num]["PRESS_EQUAL_PIPE_COND"] = {}
        for idx in range(3,9):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,18]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,19]
            sim_conds[tower_num]["PRESS_EQUAL_PIPE_COND"][key] = val
        # 真空引き配管条件
        sim_conds[tower_num]["VACUUMING_PIPE_COND"] = {}
        for idx in range(11,16):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,18]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,19]
            sim_conds[tower_num]["VACUUMING_PIPE_COND"][key] = val
        # 熱電対条件
        sim_conds[tower_num]["THERMOCOUPLE_COND"] = {}
        for idx in range(18,21):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,18]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,19]
            sim_conds[tower_num]["THERMOCOUPLE_COND"][key] = val

    return sim_conds

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
    packed_bed["v_space"] = packed_bed["Vbed"] * packed_bed["epsilon"]

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

    # 真空引き配管条件
    vacuum_pipe = sim_conds["VACUUMING_PIPE_COND"]
    vacuum_pipe["Spipe"] = np.pi * vacuum_pipe["Dpipe"] ** 2 / 4
    vacuum_pipe["Vspace"] = vacuum_pipe["Vpipe"] + packed_bed["v_space"]

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
            sim_conds["PACKED_BED_COND"]["Rbed"] / sim_conds["NUM_STR"]
            * stream + tgt_stream_conds["r_in"]
        )
    else:
        tgt_stream_conds["r_out"] = (
            sim_conds["PACKED_BED_COND"]["Rbed"] / sim_conds["NUM_STR"]
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
    """ 全体条件から壁面条件を算出する

        Args:
            sim_conds (dict): 全体共通パラメータ群
            stream_conds (dict): 各ストリーム条件
        Return:
            sim_conds (dict): 追加後の全体共通パラメータ群
    """
    tgt_stream_conds = {}
    # 内側境界半径座標(軸中心0)
    tgt_stream_conds["r_in"] = stream_conds[sim_conds["NUM_STR"]]["r_out"]
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