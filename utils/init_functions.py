import math


def read_sim_conds(df_sim_conds):
    """sim_conds.xlsxを辞書形式に変換する

    Args:
        df_sim_conds (pd.DataFrame): 実験条件のxlsxファイル
    """
    correspondence_dict = {
        "共通": "COMMON_COND",
        "触媒充填層条件": "PACKED_BED_COND",
        "導入ガス条件": "FEED_GAS_COND",
        "容器壁条件": "VESSEL_COND",
        "蓋条件": "LID_COND",
        "底条件": "BOTTOM_COND",
        "均圧配管条件": "EQUALIZING_PIPING_COND",
        "真空引き配管条件": "VACUUM_PIPING_COND",
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
        sim_conds[tower_num]["COMMON_COND"]["num_streams"] = int(sim_conds[tower_num]["COMMON_COND"]["num_streams"])
        sim_conds[tower_num]["COMMON_COND"]["num_sections"] = int(sim_conds[tower_num]["COMMON_COND"]["num_sections"])

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
    dr = sim_conds["PACKED_BED_COND"]["radius"] / sim_conds["COMMON_COND"]["num_streams"]
    # 内側境界半径座標(軸中心0)
    tgt_stream_conds["inner_radius"] = (stream - 1) * dr
    # 外側境界半径座標
    tgt_stream_conds["outer_radius"] = stream * dr
    # ストリーム断面積
    tgt_stream_conds["cross_section"] = math.pi * (
        tgt_stream_conds["outer_radius"] ** 2 - tgt_stream_conds["inner_radius"] ** 2
    )
    # ストリーム分配割合
    tgt_stream_conds["area_fraction"] = (
        tgt_stream_conds["cross_section"] / sim_conds["PACKED_BED_COND"]["cross_section"]
    )
    # 内側境界周長
    tgt_stream_conds["innter_perimeter"] = 2 * math.pi * tgt_stream_conds["inner_radius"]
    # 内側境界面積
    tgt_stream_conds["inner_boundary_area"] = (
        tgt_stream_conds["innter_perimeter"] * sim_conds["PACKED_BED_COND"]["height"]
    )
    # 外側境界周長
    tgt_stream_conds["outer_perimeter"] = 2 * math.pi * tgt_stream_conds["outer_radius"]
    # 外側境界面積
    tgt_stream_conds["outer_boundary_area"] = (
        tgt_stream_conds["outer_perimeter"] * sim_conds["PACKED_BED_COND"]["height"]
    )
    # ストリーム吸着材量
    tgt_stream_conds["adsorbent_mass"] = (
        sim_conds["PACKED_BED_COND"]["adsorbent_mass"] * tgt_stream_conds["area_fraction"]
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
    tgt_stream_conds["inner_radius"] = stream_conds[sim_conds["COMMON_COND"]["num_streams"]]["outer_radius"]
    # 外側境界半径座標
    tgt_stream_conds["outer_radius"] = sim_conds["VESSEL_COND"]["radius"]
    # ストリーム断面積
    tgt_stream_conds["cross_section"] = math.pi * (
        tgt_stream_conds["outer_radius"] ** 2 - tgt_stream_conds["inner_radius"] ** 2
    )
    # 内側境界周長
    tgt_stream_conds["innter_perimeter"] = 2 * math.pi * tgt_stream_conds["inner_radius"]
    # 内側境界面積
    tgt_stream_conds["inner_boundary_area"] = (
        tgt_stream_conds["innter_perimeter"] * sim_conds["PACKED_BED_COND"]["height"]
    )
    # 外側境界周長
    tgt_stream_conds["outer_perimeter"] = 2 * math.pi * tgt_stream_conds["outer_radius"]
    # 外側境界面積
    tgt_stream_conds["outer_boundary_area"] = (
        tgt_stream_conds["outer_perimeter"] * sim_conds["PACKED_BED_COND"]["height"]
    )
    # 容器壁質量
    tgt_stream_conds["wall_weight"] = sim_conds["VESSEL_COND"]["wall_total_weight"]

    return tgt_stream_conds
