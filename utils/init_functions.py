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
        for idx in range(3,10):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,18]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,19]
            sim_conds[tower_num]["PRESS_EQUAL_PIPE_COND"][key] = val
        # 真空引き配管条件
        sim_conds[tower_num]["VACUUMING_PIPE_COND"] = {}
        for idx in range(12,17):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,18]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,19]
            sim_conds[tower_num]["VACUUMING_PIPE_COND"][key] = val
        # 熱電対条件
        sim_conds[tower_num]["THERMOCOUPLE_COND"] = {}
        for idx in range(19,22):
            key = df_sim_conds[f"塔{tower_num}"].iloc[idx,18]
            val = df_sim_conds[f"塔{tower_num}"].iloc[idx,19]
            sim_conds[tower_num]["THERMOCOUPLE_COND"][key] = val

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