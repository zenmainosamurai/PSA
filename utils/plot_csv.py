import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from utils import const

from config.sim_conditions import CommonConditions


def plot_csv_outputs(tgt_foldapath, df_obs, tgt_sections, tower_num, timestamp, df_p_end):
    """熱バラ計算結果の可視化

    Args:
        tgt_foldapath (str): 出力先フォルダパス
        df_obs (pd.DataFrame): 観測値のデータフレーム
        tgt_sections (list): 可視化対象のセクション
        tower_num (int): 塔番号
        df_p_end (pd.DataFrame): プロセス終了時刻を含むデータフレーム
    """
    ### パラメータ設定 --------------------------------------

    linestyle_dict = {  # section
        tgt_sections[0]: "-",
        tgt_sections[1]: "--",
        tgt_sections[2]: ":",
    }
    color_dict = {  # stream
        1: "tab:red",
        2: "tab:blue",
        3: "tab:green",
    }
    color_dict_obs = {  # stream (観測値)
        1: "black",
        2: "dimgrey",
    }
    output_foldapath = tgt_foldapath + f"/png/tower_{tower_num}/"
    os.makedirs(output_foldapath, exist_ok=True)

    ### 可視化（熱バラ） -------------------------------------

    # csv読み込み
    filename_list = glob.glob(tgt_foldapath + f"/csv/tower_{tower_num}/heat/*.csv")
    df_dict = {}
    for filename in filename_list:
        df_dict[filename.split("/")[-1][:-4].split("\\")[1]] = pd.read_csv(
            filename, index_col="timestamp", encoding="shift_jis"
        )

    num_row = math.ceil((len(df_dict)) / 2)
    fig = plt.figure(figsize=(16 * 2, 5.5 * num_row), tight_layout=True)
    fig.patch.set_facecolor("white")

    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.family"] = "Meiryo"
        plt.subplot(num_row, 2, i + 1)
        # 可視化対象のcolumnsを抽出
        plt_tgt_cols = [col for col in df.columns if int(col.split("-")[-1]) in tgt_sections]
        # 各項目のプロット
        for col in plt_tgt_cols:
            stream = int(col.split("-")[-2])
            section = int(col.split("-")[-1])
            plt.plot(
                df[col],
                label=f"(str,sec) = ({stream}, {section})",
                linestyle=linestyle_dict[section],
                c=color_dict[stream],
            )
        # タイトルに単位を付ける
        unit = const.UNIT.get(key, "")
        plt.title(f"{key} {unit}" if unit else key)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("timestamp")
        # セクション到達温度のみ観測値をプロット
        if "セクション到達温度" in key or "熱電対温度" in key:
            for section in range(1, 4):
                plt.plot(
                    df_obs.loc[:timestamp, f"T{tower_num}_temp_{section}"],
                    label=f"(str,sec) = (1, {tgt_sections[section-1]})",
                    linestyle=linestyle_dict[tgt_sections[section - 1]],
                    c="black",
                )
            plt.legend(fontsize=12)
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        p_name_bfr = ""
        for idx in df_p_end.index:
            p_name = df_p_end.loc[idx, f"塔{tower_num}"]
            if p_name == p_name_bfr:
                continue
            tgt_timestamp = df_p_end.loc[idx, "終了時刻(min)"]
            plt.vlines(tgt_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            p_name_bfr = p_name
        plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(output_foldapath + "heat.png", dpi=100)
    plt.close()

    ### 可視化（マテバラ） -------------------------------------

    # csv読み込み
    filename_list = glob.glob(tgt_foldapath + f"/csv/tower_{tower_num}/material/*.csv")
    df_dict = {}
    for filename in filename_list:
        df_dict[filename.split("/")[-1][:-4].split("\\")[1]] = pd.read_csv(
            filename, index_col="timestamp", encoding="shift_jis"
        )

    num_row = math.ceil((len(df_dict)) / 2)
    fig = plt.figure(figsize=(16 * 2, 5.5 * num_row), tight_layout=True)
    fig.patch.set_facecolor("white")

    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.family"] = "Meiryo"
        plt.subplot(num_row, 2, i + 1)
        # 可視化対象のcolumnsを抽出
        plt_tgt_cols = [col for col in df.columns if int(col.split("-")[-1]) in tgt_sections]
        # 各項目のプロット
        for col in plt_tgt_cols:
            stream = int(col.split("-")[-2])
            section = int(col.split("-")[-1])
            plt.plot(
                df[col],
                label=f"(str,sec) = ({stream}, {section})",
                linestyle=linestyle_dict[section],
                c=color_dict[stream],
            )
        # タイトルに単位を付ける
        unit = const.UNIT.get(key, "")
        plt.title(f"{key} {unit}" if unit else key)
        plt.grid()
        plt.legend(fontsize=16)
        plt.xlabel("timestamp")
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        p_name_bfr = ""
        for idx in df_p_end.index:
            p_name = df_p_end.loc[idx, f"塔{tower_num}"]
            if p_name == p_name_bfr:
                continue
            tgt_timestamp = df_p_end.loc[idx, "終了時刻(min)"]
            plt.vlines(tgt_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            p_name_bfr = p_name
    plt.savefig(output_foldapath + "material.png", dpi=100)
    plt.close()

    ### 可視化（熱バラ(上下蓋)） -------------------------------------

    # csv読み込み
    filename_list = glob.glob(tgt_foldapath + f"/csv/tower_{tower_num}/heat_lid/蓋温度.csv")
    df = pd.read_csv(filename_list[0], index_col="timestamp", encoding="shift_jis")

    num_row = math.ceil((len(df.columns)) / 2)
    fig = plt.figure(figsize=(16 * 2, 5.5 * num_row), tight_layout=True)
    fig.patch.set_facecolor("white")

    for i, col in enumerate(df.columns):
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.family"] = "Meiryo"
        plt.subplot(num_row, 2, i + 1)
        plt.plot(df[col])
        # カラム名から上蓋/下蓋を判定してタイトルを設定
        if "up" in col:
            title = "セクション到達温度 [℃]_上蓋"
        else:
            title = "セクション到達温度 [℃]_下蓋"
        plt.title(title)
        plt.grid()
        plt.xlabel("timestamp")
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        p_name_bfr = ""
        for idx in df_p_end.index:
            p_name = df_p_end.loc[idx, f"塔{tower_num}"]
            if p_name == p_name_bfr:
                continue
            tgt_timestamp = df_p_end.loc[idx, "終了時刻(min)"]
            plt.vlines(tgt_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            p_name_bfr = p_name
    plt.savefig(output_foldapath + "heat_lid.png", dpi=100)
    plt.close()

    ### 可視化（others） -------------------------------------

    fig = plt.figure(figsize=(16 * 2, 5.5 * 3), tight_layout=True)
    fig.patch.set_facecolor("white")
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "Meiryo"
    plt_cell = 1

    # 1. 全圧
    filename = tgt_foldapath + f"/csv/tower_{tower_num}/others/全圧.csv"
    df = pd.read_csv(filename, index_col="timestamp", encoding="shift_jis")
    plt.subplot(3, 2, plt_cell)
    # CSVに保存されている実際のカラム名を使用
    column_name = df.columns[0]  # 最初のカラム名を取得
    plt.plot(df[column_name], label="計算値")
    plt.plot(df_obs.loc[:timestamp, f"T{tower_num}_press"], label="観測値", c="black")  # 観測値もプロット
    plt.title("全圧 [MPaA]")
    plt.legend()
    plt.grid()
    plt.xlabel("timestamp")
    # プロセス終了時刻の縦線をプロット
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    p_name_bfr = ""
    for idx in df_p_end.index:
        p_name = df_p_end.loc[idx, f"塔{tower_num}"]
        if p_name == p_name_bfr:
            continue
        tgt_timestamp = df_p_end.loc[idx, "終了時刻(min)"]
        plt.vlines(tgt_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
        p_name_bfr = p_name
    plt_cell += 1

    # 2. モル分率
    for _tgt_name in ["co2_mole_fraction", "n2_mole_fraction"]:
        japanese_name = const.TRANSLATION[_tgt_name] if _tgt_name in const.TRANSLATION else _tgt_name
        filename = tgt_foldapath + f"/csv/tower_{tower_num}/others/{japanese_name}.csv"
        df = pd.read_csv(filename, index_col="timestamp", encoding="shift_jis")
        plt.subplot(3, 2, plt_cell)
        # 可視化対象のcolumnsを抽出
        plt_tgt_cols = [col for col in df.columns if int(col.split("-")[-1]) in tgt_sections]
        # 各項目のプロット
        for col in plt_tgt_cols:
            stream = int(col.split("-")[-2])
            section = int(col.split("-")[-1])
            plt.plot(
                df[col],
                label=f"(str,sec) = ({stream}, {section})",
                linestyle=linestyle_dict[section],
                c=color_dict[stream],
            )
        plt.title(_tgt_name.split("_")[0] + "モル分率 [-]")
        plt.legend()
        plt.grid()
        plt.xlabel("timestamp")
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        p_name_bfr = ""
        for idx in df_p_end.index:
            p_name = df_p_end.loc[idx, f"塔{tower_num}"]
            if p_name == p_name_bfr:
                continue
            tgt_timestamp = df_p_end.loc[idx, "終了時刻(min)"]
            plt.vlines(tgt_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            p_name_bfr = p_name
        plt_cell += 1

    # 3. CO2,N2回収量
    title_list = ["CO2回収率 [%]", "累積CO2回収量 [mol]", "累積N2回収量 [mol]"]
    filename = tgt_foldapath + f"/csv/tower_{tower_num}/others/回収量.csv"
    df = pd.read_csv(filename, index_col="timestamp", encoding="shift_jis")

    for i, column_name in enumerate(df.columns):
        plt.subplot(3, 2, plt_cell)
        plt.plot(df[column_name], label="計算値")
        plt.title(title_list[i])
        plt.legend()
        plt.grid()
        plt.xlabel("timestamp")
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        p_name_bfr = ""
        for idx in df_p_end.index:
            p_name = df_p_end.loc[idx, f"塔{tower_num}"]
            if p_name == p_name_bfr:
                continue
            tgt_timestamp = df_p_end.loc[idx, "終了時刻(min)"]
            plt.vlines(tgt_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            p_name_bfr = p_name
        plt_cell += 1

    plt.savefig(output_foldapath + f"others.png", dpi=100)
    plt.close()


def _add_units_to_columns(columns):
    """カラム名に単位を追加する"""
    columns_with_units = []
    for col in columns:
        # カラム名から基本名を抽出（-XXX-YYY形式の場合）
        base_name = col.split("-")[0]

        # 翻訳辞書から日本語名を取得
        if base_name in const.TRANSLATION:
            japanese_name = const.TRANSLATION[base_name]
            # 単位辞書から単位を取得
            if japanese_name in const.UNIT:
                unit = const.UNIT[japanese_name]
                # カラム名に単位を追加
                if "-" in col:
                    suffix = col[len(base_name) :]
                    columns_with_units.append(f"{japanese_name}{unit}{suffix}")
                else:
                    columns_with_units.append(f"{japanese_name}{unit}")
            else:
                columns_with_units.append(col)
        else:
            columns_with_units.append(col)

    return columns_with_units


def _create_dataframe_and_save(values, columns, timestamp_index, file_path, add_units=True):
    """データフレームを作成してCSVファイルに保存する"""
    # 単位を追加するかどうかを制御
    if add_units:
        columns = _add_units_to_columns(columns)

    df = pd.DataFrame(values, columns=columns, index=timestamp_index)
    df.index.name = "timestamp"
    df.to_csv(file_path, encoding="shift_jis")


def _generate_stream_section_columns(base_name, num_streams, num_sections):
    """ストリーム・セクション用のカラム名を生成する"""
    columns = []
    for stream in range(1, 1 + num_streams):
        for section in range(1, 1 + num_sections):
            columns.append(f"{base_name}-{stream:03d}-{section:03d}")
    return columns


def _extract_heat_material_values(record_data, common_conds):
    """heat/materialデータから値を抽出する"""
    values = []
    for record in record_data:
        values_tmp = []
        for stream in range(1, 1 + common_conds.num_streams):
            for section in range(1, 1 + common_conds.num_sections):
                result_data = record.get_result(stream, section).to_dict()
                values_tmp.extend(result_data.values())
        values.append(values_tmp)
    return values


def _generate_heat_material_columns(record_data, common_conds):
    """heat/materialデータ用のカラム名を生成する"""
    columns = []
    for stream in range(1, 1 + common_conds.num_streams):
        for section in range(1, 1 + common_conds.num_sections):
            result_data = record_data[0].get_result(stream, section).to_dict()
            for key in result_data.keys():
                columns.append(f"{key}-{stream:03d}-{section:03d}")
    return columns


def _save_heat_material_data(tgt_foldapath, tower_results, common_conds, data_type):
    """heat/materialデータをCSVに保存する"""
    folder_path = os.path.join(tgt_foldapath, data_type)
    os.makedirs(folder_path, exist_ok=True)

    record_data = getattr(tower_results.time_series_data, data_type)
    values = _extract_heat_material_values(record_data, common_conds)
    columns = _generate_heat_material_columns(record_data, common_conds)

    # キーごとにCSVファイルを作成
    sample_result_data = record_data[0].get_result(1, 1).to_dict()
    for key in sample_result_data.keys():
        key_indices = [i for i, col in enumerate(columns) if key in col]
        key_values = np.array(values)[:, key_indices]
        key_columns = [columns[i] for i in key_indices]

        file_path = os.path.join(folder_path, f"{const.TRANSLATION[key]}.csv")
        _create_dataframe_and_save(
            key_values, key_columns, tower_results.time_series_data.timestamps, file_path, add_units=True
        )


def _save_heat_lid_data(tgt_foldapath, tower_results):
    """heat_lidデータをCSVに保存する"""
    folder_path = os.path.join(tgt_foldapath, "heat_lid")
    os.makedirs(folder_path, exist_ok=True)

    values = []
    for record in tower_results.time_series_data.heat_lid:
        values.append(
            [
                record["up"].temperature,
                record["down"].temperature,
            ]
        )

    columns = ["temp_reached-up", "temp_reached-down"]
    file_path = os.path.join(folder_path, "蓋温度.csv")
    _create_dataframe_and_save(values, columns, tower_results.time_series_data.timestamps, file_path, add_units=True)


def _calculate_vacuum_rate_co2(cumulative_co2, cumulative_n2):
    """CO2回収率を計算する"""
    try:
        total = cumulative_co2 + cumulative_n2
        return (cumulative_co2 / total) * 100 if total != 0 else 0
    except (ZeroDivisionError, TypeError):
        return 0


def _save_total_pressure_data(folder_path, tower_results):
    """全圧データをCSVに保存する"""
    values = [record["total_pressure"] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "全圧.csv")
    _create_dataframe_and_save(
        values, ["total_pressure"], tower_results.time_series_data.timestamps, file_path, add_units=True
    )


def _save_vacuum_amount_data(folder_path, tower_results):
    """CO2, N2回収量データをCSVに保存する"""
    values = []
    for record in tower_results.time_series_data.others:
        vacuum_rate_co2 = _calculate_vacuum_rate_co2(
            record["cumulative_co2_recovered"], record["cumulative_n2_recovered"]
        )
        values.append(
            [
                vacuum_rate_co2,
                record["cumulative_co2_recovered"],
                record["cumulative_n2_recovered"],
            ]
        )

    columns = ["vacuum_rate_co2", "cumulative_co2_recovered", "cumulative_n2_recovered"]
    file_path = os.path.join(folder_path, "回収量.csv")
    _create_dataframe_and_save(values, columns, tower_results.time_series_data.timestamps, file_path, add_units=True)


def _save_mole_fraction_data(folder_path, tower_results, common_conds, fraction_type):
    """モル分率データをCSVに保存する"""
    values = []
    for record in tower_results.time_series_data.others:
        values_tmp = []
        for stream in range(common_conds.num_streams):
            for section in range(common_conds.num_sections):
                values_tmp.append(record[fraction_type][stream, section])
        values.append(values_tmp)

    columns = _generate_stream_section_columns(fraction_type, common_conds.num_streams, common_conds.num_sections)
    japanese_name = const.TRANSLATION[fraction_type] if fraction_type in const.TRANSLATION else fraction_type
    file_path = os.path.join(folder_path, f"{japanese_name}.csv")
    _create_dataframe_and_save(values, columns, tower_results.time_series_data.timestamps, file_path, add_units=True)


def _save_others_data(tgt_foldapath, tower_results, common_conds):
    """othersデータをCSVに保存する"""
    folder_path = os.path.join(tgt_foldapath, "others")
    os.makedirs(folder_path, exist_ok=True)

    # 全圧
    _save_total_pressure_data(folder_path, tower_results)

    # CO2, N2回収量
    _save_vacuum_amount_data(folder_path, tower_results)

    # モル分率
    for fraction_type in ["co2_mole_fraction", "n2_mole_fraction"]:
        _save_mole_fraction_data(folder_path, tower_results, common_conds, fraction_type)


def outputs_to_csv(tgt_foldapath, tower_results, common_conds: CommonConditions):
    """計算結果をcsv出力する

    Args:
        tgt_foldapath (str): 出力先フォルダパス
        tower_results (TowerSimulationResults): 計算結果
        common_conds (CommonConditions): 実験パラメータ
    """
    # heat, material データの保存
    for data_type in ["heat", "material"]:
        _save_heat_material_data(tgt_foldapath, tower_results, common_conds, data_type)

    # heat_lid データの保存
    _save_heat_lid_data(tgt_foldapath, tower_results)

    # others データの保存
    _save_others_data(tgt_foldapath, tower_results, common_conds)

    # heat_wall
