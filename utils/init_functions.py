import math


def read_sim_conds(df_sim_conds):
    """sim_conds.xlsxを辞書形式に変換する

    Args:
        df_sim_conds (pd.DataFrame): 実験条件のxlsxファイル
    """
    correspondence_dict = {
        "共通": "COMMON_COND",
        "触媒充填層条件": "PACKED_BED_COND",
        "導入ガス条件": "INFLOW_GAS_COND",
        "容器壁条件": "DRUM_WALL_COND",
        "上蓋の条件": "LID_COND_UP",
        "下蓋の条件": "LID_COND_DOWN",
        "均圧配管条件": "PRESS_EQUAL_PIPE_COND",
        "真空引き配管条件": "VACUUMING_PIPE_COND",
        "熱電対条件": "THERMOCOUPLE_COND",
    }
    # パラメータ読み込み
    sim_conds = {1: {}, 2: {}, 3: {}}
    for cond_name, df in df_sim_conds.items():
        # 日本語→英語に変換
        key = correspondence_dict[cond_name]
        for tower_num in range(1, 4):
            sim_conds[tower_num][key] = {}
            # 各パラメータを収録
            for param in df.index:
                sim_conds[tower_num][key][param] = df.loc[param, f"塔{tower_num}"]
    # 前処理
    for tower_num in range(1, 4):
        sim_conds[tower_num]["COMMON_COND"]["NUM_STR"] = int(
            sim_conds[tower_num]["COMMON_COND"]["NUM_STR"]
        )
        sim_conds[tower_num]["COMMON_COND"]["NUM_SEC"] = int(
            sim_conds[tower_num]["COMMON_COND"]["NUM_SEC"]
        )

    return sim_conds


def init_stream_conds(sim_conds, stream, stream_conds):
    """全体条件から各ストリーム条件を算出する

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
        tgt_stream_conds["r_in"] = stream_conds[stream - 1]["r_out"]
    # 外側境界半径座標
    if stream == 1:
        tgt_stream_conds["r_out"] = (
            sim_conds["PACKED_BED_COND"]["Rbed"]
            / sim_conds["COMMON_COND"]["NUM_STR"]
            * stream
            + tgt_stream_conds["r_in"]
        )
    else:
        tgt_stream_conds["r_out"] = (
            sim_conds["PACKED_BED_COND"]["Rbed"]
            / sim_conds["COMMON_COND"]["NUM_STR"]
            * stream
            + stream_conds[stream - 1]["r_in"]
        )
    # ストリーム断面積
    tgt_stream_conds["Sstream"] = math.pi * (
        tgt_stream_conds["r_out"] ** 2 - tgt_stream_conds["r_in"] ** 2
    )
    # ストリーム分配割合
    tgt_stream_conds["streamratio"] = (
        tgt_stream_conds["Sstream"] / sim_conds["PACKED_BED_COND"]["Sbed"]
    )
    # 内側境界周長
    tgt_stream_conds["Circ_in"] = 2 * math.pi * tgt_stream_conds["r_in"]
    # 内側境界面積
    tgt_stream_conds["Ain"] = (
        tgt_stream_conds["Circ_in"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    )
    # 外側境界周長
    tgt_stream_conds["Circ_out"] = 2 * math.pi * tgt_stream_conds["r_out"]
    # 外側境界面積
    tgt_stream_conds["Aout"] = (
        tgt_stream_conds["Circ_out"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    )
    # ストリーム吸着材量
    tgt_stream_conds["Mabs"] = (
        sim_conds["PACKED_BED_COND"]["Mabs"] * tgt_stream_conds["streamratio"]
    )

    return tgt_stream_conds


def init_drum_wall_conds(sim_conds, stream_conds):
    """全体条件から壁面条件を算出する

    Args:
        sim_conds (dict): 全体共通パラメータ群
        stream_conds (dict): 各ストリーム条件
    Return:
        sim_conds (dict): 追加後の全体共通パラメータ群
    """
    tgt_stream_conds = {}
    # 内側境界半径座標(軸中心0)
    tgt_stream_conds["r_in"] = stream_conds[sim_conds["COMMON_COND"]["NUM_STR"]][
        "r_out"
    ]
    # 外側境界半径座標
    tgt_stream_conds["r_out"] = sim_conds["DRUM_WALL_COND"]["Rdrum"]
    # ストリーム断面積
    tgt_stream_conds["Sstream"] = math.pi * (
        tgt_stream_conds["r_out"] ** 2 - tgt_stream_conds["r_in"] ** 2
    )
    # 内側境界周長
    tgt_stream_conds["Circ_in"] = 2 * math.pi * tgt_stream_conds["r_in"]
    # 内側境界面積
    tgt_stream_conds["Ain"] = (
        tgt_stream_conds["Circ_in"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    )
    # 外側境界周長
    tgt_stream_conds["Circ_out"] = 2 * math.pi * tgt_stream_conds["r_out"]
    # 外側境界面積
    tgt_stream_conds["Aout"] = (
        tgt_stream_conds["Circ_out"] * sim_conds["PACKED_BED_COND"]["Lbed"]
    )
    # 容器壁質量
    tgt_stream_conds["Mwall"] = sim_conds["DRUM_WALL_COND"]["Wdrumw"]

    return tgt_stream_conds
