"""CSV形式での結果出力モジュール

シミュレーション結果をCSVファイルとグラフ画像として出力します。

出力先: output/{cond_id}/tower{N}/
出力内容:
    - 温度分布（temp/）
    - 吸着量分布（loading/）
    - モル分率（mole_fraction/）
    - 圧力履歴（pressure/）
    - 熱収支（heat_balance/）
    - CO2回収量（vacuum/）

使用例:
    from utils.plot_csv import save_results
    save_results(output_dir, tower_results, common_conditions)
"""
import os
import warnings
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import math
import glob
from utils import const

from config.sim_conditions import CommonConditions
from common.enums import LidPosition

# matplotlibのフォント関連警告を抑制
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='Glyph .* missing from font')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def _get_japanese_font():
    """日本語フォントを取得（クロスプラットフォーム対応）
    
    Returns:
        str: 利用可能な日本語フォント名
    """
    # 優先順位でフォントを試行
    font_candidates = [
        "Meiryo",           # Windows
        "Yu Gothic",        # Windows
        "MS Gothic",        # Windows
        "Hiragino Sans",    # macOS
        "Noto Sans CJK JP", # Linux (noto-fonts-cjk)
        "IPAGothic",        # Linux (ipa-fonts)
        "IPAPGothic",       # Linux
        "TakaoPGothic",     # Linux (Ubuntuなど)
        "VL Gothic",        # Linux
        "DejaVu Sans",      # フォールバック（日本語非対応）
    ]
    
    # 利用可能なフォント一覧を取得
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    for font in font_candidates:
        if font in available_fonts:
            return font
    
    # 見つからない場合はデフォルト
    return "sans-serif"


# モジュール読み込み時にフォントを決定
JAPANESE_FONT = _get_japanese_font()


def plot_csv_outputs(output_dir, df_obs, target_sections, tower_num, timestamp, df_schedule):
    """熱バラ計算結果の可視化

    Args:
        output_dir (str): 出力先フォルダパス
        df_obs (pd.DataFrame): 観測値のデータフレーム
        target_sections (list): 可視化対象のセクション
        tower_num (int): 塔番号
        df_schedule (pd.DataFrame): プロセス終了時刻を含むデータフレーム
    """
    ### パラメータ設定 --------------------------------------

    linestyle_dict = {  # section
        target_sections[0]: "-",
        target_sections[1]: "--",
        target_sections[2]: ":",
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
    png_output_dir = output_dir + f"/png/tower_{tower_num}/"
    os.makedirs(png_output_dir, exist_ok=True)

    ### 可視化（熱バラ） -------------------------------------

    # csv読み込み
    filename_list = glob.glob(output_dir + f"/csv/tower_{tower_num}/heat/*.csv")
    df_dict = {}
    for filename in filename_list:
        # ファイル名（拡張子なし）を取得（クロスプラットフォーム対応）
        basename = os.path.splitext(os.path.basename(filename))[0]
        df_dict[basename] = pd.read_csv(
            filename, index_col="timestamp", encoding="shift_jis"
        )

    num_row = math.ceil((len(df_dict)) / 2)
    fig = plt.figure(figsize=(16 * 2, 5.5 * num_row), tight_layout=True)
    fig.patch.set_facecolor("white")

    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.family"] = JAPANESE_FONT
        plt.subplot(num_row, 2, i + 1)
        # 可視化対象のcolumnsを抽出
        plot_columns = [col for col in df.columns if int(col.split("-")[-1]) in target_sections]
        # 各項目のプロット
        for col in plot_columns:
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
            if df_obs is not None and not df_obs.empty:
                for section in range(1, 4):
                    col_name = f"T{tower_num}_temp_{section}"
                    if col_name in df_obs.columns:
                        plt.plot(
                            df_obs.loc[:timestamp, col_name],
                            label=f"(str,sec) = (1, {target_sections[section-1]})",
                            linestyle=linestyle_dict[target_sections[section - 1]],
                            c="black",
                        )
                plt.legend(fontsize=12)
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        prev_process_name = ""
        for idx in df_schedule.index:
            p_name = df_schedule.loc[idx, f"塔{tower_num}"]
            if p_name == prev_process_name:
                continue
            end_timestamp = df_schedule.loc[idx, "終了時刻(min)"]
            plt.vlines(end_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            prev_process_name = p_name
        plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(png_output_dir, f"heat_tower{tower_num}.png"), dpi=100)
    plt.close()

    ### 可視化（マテバラ） -------------------------------------

    # csv読み込み
    filename_list = glob.glob(output_dir + f"/csv/tower_{tower_num}/material/*.csv")
    df_dict = {}
    for filename in filename_list:
        # ファイル名（拡張子なし）を取得（クロスプラットフォーム対応）
        basename = os.path.splitext(os.path.basename(filename))[0]
        df_dict[basename] = pd.read_csv(
            filename, index_col="timestamp", encoding="shift_jis"
        )

    num_row = math.ceil((len(df_dict)) / 2)
    fig = plt.figure(figsize=(16 * 2, 5.5 * num_row), tight_layout=True)
    fig.patch.set_facecolor("white")

    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.family"] = JAPANESE_FONT
        plt.subplot(num_row, 2, i + 1)
        # 可視化対象のcolumnsを抽出
        plot_columns = [col for col in df.columns if int(col.split("-")[-1]) in target_sections]
        # 各項目のプロット
        for col in plot_columns:
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
        prev_process_name = ""
        for idx in df_schedule.index:
            p_name = df_schedule.loc[idx, f"塔{tower_num}"]
            if p_name == prev_process_name:
                continue
            end_timestamp = df_schedule.loc[idx, "終了時刻(min)"]
            plt.vlines(end_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            prev_process_name = p_name
    plt.savefig(os.path.join(png_output_dir, f"material_tower{tower_num}.png"), dpi=100)
    plt.close()

    ### 可視化（熱バラ(上下蓋)） -------------------------------------

    # csv読み込み
    filename_list = glob.glob(output_dir + f"/csv/tower_{tower_num}/heat_lid/蓋温度.csv")
    df = pd.read_csv(filename_list[0], index_col="timestamp", encoding="shift_jis")

    num_row = math.ceil((len(df.columns)) / 2)
    fig = plt.figure(figsize=(16 * 2, 5.5 * num_row), tight_layout=True)
    fig.patch.set_facecolor("white")

    for i, col in enumerate(df.columns):
        plt.rcParams["font.size"] = 20
        plt.rcParams["font.family"] = JAPANESE_FONT
        plt.subplot(num_row, 2, i + 1)
        plt.plot(df[col])
        # カラム名から上蓋/下蓋を判定してタイトルを設定
        if "top" in col:
            title = "セクション到達温度 [℃]_上蓋"
        else:
            title = "セクション到達温度 [℃]_下蓋"
        plt.title(title)
        plt.grid()
        plt.xlabel("timestamp")
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        prev_process_name = ""
        for idx in df_schedule.index:
            p_name = df_schedule.loc[idx, f"塔{tower_num}"]
            if p_name == prev_process_name:
                continue
            end_timestamp = df_schedule.loc[idx, "終了時刻(min)"]
            plt.vlines(end_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            prev_process_name = p_name
    plt.savefig(os.path.join(png_output_dir, f"heat_lid_tower{tower_num}.png"), dpi=100)
    plt.close()

    ### 可視化（others） -------------------------------------

    fig = plt.figure(figsize=(16 * 2, 5.5 * 3), tight_layout=True)
    fig.patch.set_facecolor("white")
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = JAPANESE_FONT
    plt_cell = 1

    # 1. 全圧
    filename = output_dir + f"/csv/tower_{tower_num}/others/全圧.csv"
    df = pd.read_csv(filename, index_col="timestamp", encoding="shift_jis")
    plt.subplot(3, 2, plt_cell)
    # CSVに保存されている実際のカラム名を使用
    column_name = df.columns[0]  # 最初のカラム名を取得
    plt.plot(df[column_name], label="計算値")
    if df_obs is not None and not df_obs.empty:
        press_col = f"T{tower_num}_press"
        if press_col in df_obs.columns:
            plt.plot(df_obs.loc[:timestamp, press_col], label="観測値", c="black")
    plt.title("全圧 [MPaA]")
    plt.legend()
    plt.grid()
    plt.xlabel("timestamp")
    # プロセス終了時刻の縦線をプロット
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    prev_process_name = ""
    for idx in df_schedule.index:
        p_name = df_schedule.loc[idx, f"塔{tower_num}"]
        if p_name == prev_process_name:
            continue
        end_timestamp = df_schedule.loc[idx, "終了時刻(min)"]
        plt.vlines(end_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
        prev_process_name = p_name
    plt_cell += 1

    # 2. モル分率
    for metric_name in ["co2_mole_fraction", "n2_mole_fraction"]:
        japanese_name = const.TRANSLATION[metric_name] if metric_name in const.TRANSLATION else metric_name
        filename = output_dir + f"/csv/tower_{tower_num}/others/{japanese_name}.csv"
        df = pd.read_csv(filename, index_col="timestamp", encoding="shift_jis")
        plt.subplot(3, 2, plt_cell)
        # 可視化対象のcolumnsを抽出
        plot_columns = [col for col in df.columns if int(col.split("-")[-1]) in target_sections]
        # 各項目のプロット
        for col in plot_columns:
            stream = int(col.split("-")[-2])
            section = int(col.split("-")[-1])
            plt.plot(
                df[col],
                label=f"(str,sec) = ({stream}, {section})",
                linestyle=linestyle_dict[section],
                c=color_dict[stream],
            )
        plt.title(metric_name.split("_")[0] + "モル分率 [-]")
        plt.legend()
        plt.grid()
        plt.xlabel("timestamp")
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        prev_process_name = ""
        for idx in df_schedule.index:
            p_name = df_schedule.loc[idx, f"塔{tower_num}"]
            if p_name == prev_process_name:
                continue
            end_timestamp = df_schedule.loc[idx, "終了時刻(min)"]
            plt.vlines(end_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            prev_process_name = p_name
        plt_cell += 1

    # 3. CO2,N2回収量
    file_info_list = [
        ("CO2回収率.csv", "CO2回収率 [%]"),
        ("累積CO2回収量.csv", "累積CO2回収量 [Nm3]"),
        ("累積N2回収量.csv", "累積N2回収量 [Nm3]"),
    ]

    for filename, title in file_info_list:
        filepath = output_dir + f"/csv/tower_{tower_num}/others/{filename}"
        df = pd.read_csv(filepath, index_col="timestamp", encoding="shift_jis")

        plt.subplot(3, 2, plt_cell)
        plt.plot(df[df.columns[0]], label="計算値")  # 最初のカラムをプロット
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.xlabel("timestamp")
        # プロセス終了時刻の縦線をプロット
        ax = plt.gca()
        ymin, ymax = ax.get_ylim()
        prev_process_name = ""
        for idx in df_schedule.index:
            p_name = df_schedule.loc[idx, f"塔{tower_num}"]
            if p_name == prev_process_name:
                continue
            end_timestamp = df_schedule.loc[idx, "終了時刻(min)"]
            plt.vlines(end_timestamp, ymin=ymin, ymax=ymax, colors="tab:orange", alpha=1)
            prev_process_name = p_name
        plt_cell += 1

    plt.savefig(os.path.join(png_output_dir, f"others_tower{tower_num}.png"), dpi=100)
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
                result = record.get_result(stream, section)
                if result is None:
                    continue
                result_data = result.to_dict()
                values_tmp.extend(result_data.values())
        values.append(values_tmp)
    return values


def _generate_heat_material_columns(record_data, common_conds):
    """heat/materialデータ用のカラム名を生成する"""
    columns = []
    for stream in range(1, 1 + common_conds.num_streams):
        for section in range(1, 1 + common_conds.num_sections):
            result = record_data[0].get_result(stream, section)
            if result is None:
                continue
            result_data = result.to_dict()
            for key in result_data.keys():
                columns.append(f"{key}-{stream:03d}-{section:03d}")
    return columns


def _save_heat_material_data(output_dir, tower_results, common_conds, data_type):
    """heat/materialデータをCSVに保存する"""
    folder_path = os.path.join(output_dir, data_type)
    os.makedirs(folder_path, exist_ok=True)

    record_data = getattr(tower_results.time_series_data, data_type)
    values = _extract_heat_material_values(record_data, common_conds)
    columns = _generate_heat_material_columns(record_data, common_conds)

    # キーごとにCSVファイルを作成
    sample_result_data = record_data[0].get_result(1, 1).to_dict()
    for key in sample_result_data.keys():
        key_indices = [i for i, col in enumerate(columns) if (col.split("-")[0] if "-" in col else col) == key]
        key_values = np.array(values)[:, key_indices]
        key_columns = [columns[i] for i in key_indices]

        file_path = os.path.join(folder_path, f"{const.TRANSLATION[key]}.csv")
        _create_dataframe_and_save(
            key_values, key_columns, tower_results.time_series_data.timestamps, file_path, add_units=True
        )


def _save_heat_lid_data(output_dir, tower_results):
    """heat_lidデータをCSVに保存する"""
    folder_path = os.path.join(output_dir, "heat_lid")
    os.makedirs(folder_path, exist_ok=True)

    values = []
    for record in tower_results.time_series_data.heat_lid:
        values.append(
            [
                record[LidPosition.TOP].temperature,
                record[LidPosition.BOTTOM].temperature,
            ]
        )

    columns = ["temp_reached-top", "temp_reached-bottom"]
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
    """CO2, N2回収量データを個別CSVファイルに保存する"""
    # CO2回収率を個別ファイルに保存
    vacuum_rate_co2_values = []
    for record in tower_results.time_series_data.others:
        vacuum_rate_co2 = _calculate_vacuum_rate_co2(
            record["cumulative_co2_recovered"], record["cumulative_n2_recovered"]
        )
        vacuum_rate_co2_values.append(vacuum_rate_co2)

    file_path = os.path.join(folder_path, "CO2回収率.csv")
    _create_dataframe_and_save(
        vacuum_rate_co2_values,
        ["vacuum_rate_co2"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
    )

    # 累積CO2回収量を個別ファイルに保存
    cumulative_co2_values = [record["cumulative_co2_recovered"] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "累積CO2回収量.csv")
    _create_dataframe_and_save(
        cumulative_co2_values,
        ["cumulative_co2_recovered"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
    )

    # 累積N2回収量を個別ファイルに保存
    cumulative_n2_values = [record["cumulative_n2_recovered"] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "累積N2回収量.csv")
    _create_dataframe_and_save(
        cumulative_n2_values,
        ["cumulative_n2_recovered"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
    )


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


def _save_others_data(output_dir, tower_results, common_conds):
    """othersデータをCSVに保存する"""
    folder_path = os.path.join(output_dir, "others")
    os.makedirs(folder_path, exist_ok=True)

    # 全圧
    _save_total_pressure_data(folder_path, tower_results)

    # CO2, N2回収量
    _save_vacuum_amount_data(folder_path, tower_results)

    # モル分率
    for fraction_type in ["co2_mole_fraction", "n2_mole_fraction"]:
        _save_mole_fraction_data(folder_path, tower_results, common_conds, fraction_type)


def outputs_to_csv(output_dir, tower_results, common_conds: CommonConditions):
    """計算結果をcsv出力する

    Args:
        output_dir (str): 出力先フォルダパス
        tower_results (TowerSimulationResults): 計算結果
        common_conds (CommonConditions): 実験パラメータ
    """
    # heat, material データの保存
    for data_type in ["heat", "material"]:
        _save_heat_material_data(output_dir, tower_results, common_conds, data_type)

    # heat_lid データの保存
    _save_heat_lid_data(output_dir, tower_results)

    # others データの保存
    _save_others_data(output_dir, tower_results, common_conds)

    # heat_wall
