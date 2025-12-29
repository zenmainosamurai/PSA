"""Excel形式での結果出力モジュール（新版）

plot_xlsx.pyの改良版です。
シミュレーション結果をExcelファイル（グラフ付き）として出力します。

出力先: output/{cond_id}/tower{N}/
出力内容:
    - 温度分布（temp/）
    - 吸着量分布（loading/）
    - モル分率（mole_fraction/）
    - 圧力履歴（pressure/）
    - 熱収支（heat_balance/）

使用例:
    from utils.plot_xlsx_new import save_results
    save_results(output_dir, tower_results, common_conditions)
"""
import os
import pandas as pd
import numpy as np
import glob
from utils import const
import xlsxwriter

from config.sim_conditions import CommonConditions
from common.enums import LidPosition


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


def _create_dataframe_and_save_xlsx(
    values, columns, timestamp_index, file_path, add_units=True, chart_title="", target_sections=None
):
    """データフレームを作成してXLSXファイルに保存し、グラフシートも追加する"""
    # 単位を追加するかどうかを制御
    if add_units:
        columns = _add_units_to_columns(columns)

    df = pd.DataFrame(values, columns=columns, index=timestamp_index)
    df.index.name = "timestamp"

    # Excelファイルとして保存
    workbook = xlsxwriter.Workbook(file_path)

    # データシートの作成
    data_worksheet = workbook.add_worksheet("データ")

    # データフレームをExcelに書き込み
    data_worksheet.write(0, 0, "timestamp")
    for col_idx, col_name in enumerate(df.columns):
        data_worksheet.write(0, col_idx + 1, col_name)

    for row_idx, (index, row) in enumerate(df.iterrows()):
        data_worksheet.write(row_idx + 1, 0, index)
        for col_idx, value in enumerate(row):
            data_worksheet.write(row_idx + 1, col_idx + 1, value)

    # グラフシートの作成
    chart_worksheet = workbook.add_worksheet("グラフ")

    # Excelネイティブグラフの作成
    if len(df.columns) > 0:
        chart = workbook.add_chart({"type": "line"})

        # 可視化対象のカラムを抽出（target_sectionsが指定されている場合）
        if target_sections is not None:
            target_columns = []
            for col in df.columns:
                try:
                    if "-" in col:
                        section = int(col.split("-")[-1])
                        if section in target_sections:
                            target_columns.append(col)
                    else:
                        target_columns.append(col)
                except (ValueError, IndexError):
                    target_columns.append(col)
        else:
            target_columns = df.columns[: min(10, len(df.columns))]  # 最大10系列まで

        # 各系列をグラフに追加
        for col_idx, col_name in enumerate(target_columns):
            chart.add_series(
                {
                    "name": col_name,
                    "categories": ["データ", 1, 0, len(df), 0],  # timestamp列
                    "values": [
                        "データ",
                        1,
                        list(df.columns).index(col_name) + 1,
                        len(df),
                        list(df.columns).index(col_name) + 1,
                    ],
                    "line": {"width": 2},
                }
            )

        chart.set_title({"name": chart_title})
        chart.set_x_axis({"name": "timestamp"})
        chart.set_size({"width": 800, "height": 480})

        # グラフをシートに挿入
        chart_worksheet.insert_chart("A1", chart)

    workbook.close()


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


def _save_heat_material_data_xlsx(output_dir, tower_results, common_conds, data_type, target_sections=None):
    """heat/materialデータをXLSXに保存する"""
    folder_path = os.path.join(output_dir, data_type)
    os.makedirs(folder_path, exist_ok=True)

    record_data = getattr(tower_results.time_series_data, data_type)
    values = _extract_heat_material_values(record_data, common_conds)
    columns = _generate_heat_material_columns(record_data, common_conds)

    # キーごとにXLSXファイルを作成
    sample_result_data = record_data[0].get_result(1, 1).to_dict()
    for key in sample_result_data.keys():
        key_indices = [i for i, col in enumerate(columns) if key in col]
        key_values = np.array(values)[:, key_indices]
        key_columns = [columns[i] for i in key_indices]

        japanese_name = const.TRANSLATION.get(key, key)
        file_path = os.path.join(folder_path, f"{japanese_name}.xlsx")
        chart_title = f"{japanese_name} ({const.UNIT.get(japanese_name, '')})"

        _create_dataframe_and_save_xlsx(
            key_values,
            key_columns,
            tower_results.time_series_data.timestamps,
            file_path,
            add_units=True,
            chart_title=chart_title,
            target_sections=target_sections,
        )


def _save_heat_lid_data_xlsx(output_dir, tower_results):
    """heat_lidデータをXLSXに保存する"""
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
    file_path = os.path.join(folder_path, "蓋温度.xlsx")
    _create_dataframe_and_save_xlsx(
        values, columns, tower_results.time_series_data.timestamps, file_path, add_units=True, chart_title="蓋温度 [℃]"
    )


def _calculate_vacuum_rate_co2(cumulative_co2, cumulative_n2):
    """CO2回収率を計算する"""
    try:
        total = cumulative_co2 + cumulative_n2
        return (cumulative_co2 / total) * 100 if total != 0 else 0
    except (ZeroDivisionError, TypeError):
        return 0


def _save_total_pressure_data_xlsx(folder_path, tower_results):
    """全圧データをXLSXに保存する"""
    values = [[record["total_pressure"]] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "全圧.xlsx")
    _create_dataframe_and_save_xlsx(
        values,
        ["total_pressure"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title="全圧 [MPaA]",
    )


def _save_vacuum_amount_data_xlsx(folder_path, tower_results):
    """CO2, N2回収量データを個別XLSXファイルに保存する"""
    # CO2回収率を個別ファイルに保存
    vacuum_rate_co2_values = []
    for record in tower_results.time_series_data.others:
        vacuum_rate_co2 = _calculate_vacuum_rate_co2(
            record["cumulative_co2_recovered"], record["cumulative_n2_recovered"]
        )
        vacuum_rate_co2_values.append([vacuum_rate_co2])

    file_path = os.path.join(folder_path, "CO2回収率.xlsx")
    _create_dataframe_and_save_xlsx(
        vacuum_rate_co2_values,
        ["vacuum_rate_co2"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title="CO2回収率 [%]",
    )

    # 累積CO2回収量を個別ファイルに保存
    cumulative_co2_values = [[record["cumulative_co2_recovered"]] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "累積CO2回収量.xlsx")
    _create_dataframe_and_save_xlsx(
        cumulative_co2_values,
        ["cumulative_co2_recovered"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title="累積CO2回収量 [Nm3]",
    )

    # 累積N2回収量を個別ファイルに保存
    cumulative_n2_values = [[record["cumulative_n2_recovered"]] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "累積N2回収量.xlsx")
    _create_dataframe_and_save_xlsx(
        cumulative_n2_values,
        ["cumulative_n2_recovered"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title="累積N2回収量 [Nm3]",
    )


def _save_mole_fraction_data_xlsx(folder_path, tower_results, common_conds, fraction_type, target_sections=None):
    """モル分率データをXLSXに保存する"""
    values = []
    for record in tower_results.time_series_data.others:
        values_tmp = []
        for stream in range(common_conds.num_streams):
            for section in range(common_conds.num_sections):
                values_tmp.append(record[fraction_type][stream, section])
        values.append(values_tmp)

    columns = _generate_stream_section_columns(fraction_type, common_conds.num_streams, common_conds.num_sections)
    japanese_name = const.TRANSLATION[fraction_type] if fraction_type in const.TRANSLATION else fraction_type
    file_path = os.path.join(folder_path, f"{japanese_name}.xlsx")
    chart_title = f"{fraction_type.split('_')[0]}モル分率 [-]"

    _create_dataframe_and_save_xlsx(
        values,
        columns,
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title=chart_title,
        target_sections=target_sections,
    )


def _save_others_data_xlsx(output_dir, tower_results, common_conds, target_sections=None):
    """othersデータをXLSXに保存する"""
    folder_path = os.path.join(output_dir, "others")
    os.makedirs(folder_path, exist_ok=True)

    # 全圧
    _save_total_pressure_data_xlsx(folder_path, tower_results)

    # CO2, N2回収量
    _save_vacuum_amount_data_xlsx(folder_path, tower_results)

    # モル分率
    for fraction_type in ["co2_mole_fraction", "n2_mole_fraction"]:
        _save_mole_fraction_data_xlsx(folder_path, tower_results, common_conds, fraction_type, target_sections)


def outputs_to_xlsx(output_dir, tower_results, common_conds: CommonConditions, target_sections=None):
    """計算結果をxlsx出力する

    Args:
        output_dir (str): 出力先フォルダパス
        tower_results (TowerSimulationResults): 計算結果
        common_conds (CommonConditions): 実験パラメータ
        target_sections (list): 可視化対象のセクション
    """
    # heat, material データの保存
    for data_type in ["heat", "material"]:
        _save_heat_material_data_xlsx(output_dir, tower_results, common_conds, data_type, target_sections)

    # heat_lid データの保存
    _save_heat_lid_data_xlsx(output_dir, tower_results)

    # others データの保存
    _save_others_data_xlsx(output_dir, tower_results, common_conds, target_sections)


def plot_xlsx_outputs(output_dir, df_obs, target_sections, tower_num, timestamp, df_schedule):
    """熱バラ計算結果のxlsx可視化（統合されたファイルを作成）

    Args:
        output_dir (str): 出力先フォルダパス
        df_obs (pd.DataFrame): 観測値のデータフレーム
        target_sections (list): 可視化対象のセクション
        tower_num (int): 塔番号
        df_schedule (pd.DataFrame): プロセス終了時刻を含むデータフレーム
    """
    output_foldapath = output_dir + f"/xlsx/tower_{tower_num}/"
    os.makedirs(output_foldapath, exist_ok=True)

    # 統合されたExcelファイルを作成
    _create_integrated_xlsx_files(output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath)


def _create_integrated_xlsx_files(
    output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath
):
    """統合されたXLSXファイルを作成する"""

    # heat.xlsx
    _create_integrated_heat_xlsx(output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath)

    # material.xlsx
    _create_integrated_material_xlsx(
        output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath
    )

    # heat_lid.xlsx
    _create_integrated_heat_lid_xlsx(output_dir, tower_num, df_obs, timestamp, df_schedule, output_foldapath)

    # others.xlsx
    _create_integrated_others_xlsx(
        output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath
    )


def _create_integrated_heat_xlsx(output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath):
    """統合されたheat.xlsxファイルを作成する"""
    # CSVファイルからデータを読み込み
    filename_list = glob.glob(output_dir + f"/csv/tower_{tower_num}/heat/*.csv")

    if not filename_list:
        return

    # 統合データフレームを作成
    combined_df = pd.DataFrame()

    for filename in filename_list:
        df = pd.read_csv(filename, index_col="timestamp", encoding="shift-jis")
        if combined_df.empty:
            combined_df = df.copy()
        else:
            combined_df = pd.concat([combined_df, df], axis=1)

    file_path = output_foldapath + "heat.xlsx"
    _create_xlsx_with_multiple_charts(file_path, combined_df, "熱バランス", target_sections, filename_list)


def _create_integrated_material_xlsx(
    output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath
):
    """統合されたmaterial.xlsxファイルを作成する"""
    # CSVファイルからデータを読み込み
    filename_list = glob.glob(output_dir + f"/csv/tower_{tower_num}/material/*.csv")

    if not filename_list:
        return

    # 統合データフレームを作成
    combined_df = pd.DataFrame()

    for filename in filename_list:
        df = pd.read_csv(filename, index_col="timestamp", encoding="shift-jis")
        if combined_df.empty:
            combined_df = df.copy()
        else:
            combined_df = pd.concat([combined_df, df], axis=1)

    file_path = output_foldapath + "material.xlsx"
    _create_xlsx_with_multiple_charts(file_path, combined_df, "マテリアルバランス", target_sections, filename_list)


def _create_integrated_heat_lid_xlsx(output_dir, tower_num, df_obs, timestamp, df_schedule, output_foldapath):
    """統合されたheat_lid.xlsxファイルを作成する"""
    filename_list = glob.glob(output_dir + f"/csv/tower_{tower_num}/heat_lid/蓋温度.csv")

    if not filename_list:
        return

    df = pd.read_csv(filename_list[0], index_col="timestamp", encoding="shift-jis")
    file_path = output_foldapath + "heat_lid.xlsx"

    # 単一チャートファイルを作成
    workbook = xlsxwriter.Workbook(file_path)

    # データシート
    data_worksheet = workbook.add_worksheet("データ")
    data_worksheet.write(0, 0, "timestamp")
    for col_idx, col_name in enumerate(df.columns):
        data_worksheet.write(0, col_idx + 1, col_name)

    for row_idx, (index, row) in enumerate(df.iterrows()):
        data_worksheet.write(row_idx + 1, 0, index)
        for col_idx, value in enumerate(row):
            data_worksheet.write(row_idx + 1, col_idx + 1, value)

    # グラフシート
    chart_worksheet = workbook.add_worksheet("グラフ")
    chart = workbook.add_chart({"type": "line"})

    for col_idx, col_name in enumerate(df.columns):
        chart.add_series(
            {
                "name": col_name,
                "categories": ["データ", 1, 0, len(df), 0],
                "values": ["データ", 1, col_idx + 1, len(df), col_idx + 1],
                "line": {"width": 2},
            }
        )

    chart.set_title({"name": "蓋温度 [℃]"})
    chart.set_x_axis({"name": "timestamp"})
    chart.set_size({"width": 800, "height": 480})
    chart_worksheet.insert_chart("A1", chart)

    workbook.close()


def _create_integrated_others_xlsx(
    output_dir, tower_num, df_obs, timestamp, df_schedule, target_sections, output_foldapath
):
    """統合されたothers.xlsxファイルを作成する"""
    # 各othersファイルからデータを収集
    others_files = [
        "全圧.csv",
        "CO2モル分率.csv",
        "N2モル分率.csv",
        "CO2回収率.csv",
        "累積CO2回収量.csv",
        "累積N2回収量.csv",
    ]

    combined_df = pd.DataFrame()

    for filename in others_files:
        filepath = output_dir + f"/csv/tower_{tower_num}/others/{filename}"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col="timestamp", encoding="shift-jis")
            if combined_df.empty:
                combined_df = df.copy()
            else:
                combined_df = pd.concat([combined_df, df], axis=1)

    if not combined_df.empty:
        file_path = output_foldapath + "others.xlsx"
        _create_xlsx_with_multiple_charts(file_path, combined_df, "その他データ", target_sections, others_files)


def _create_xlsx_with_multiple_charts(file_path, df, title_prefix, target_sections, source_files):
    """複数のチャートを含むXLSXファイルを作成する"""
    workbook = xlsxwriter.Workbook(file_path)

    # データシート
    data_worksheet = workbook.add_worksheet("データ")
    data_worksheet.write(0, 0, "timestamp")
    for col_idx, col_name in enumerate(df.columns):
        data_worksheet.write(0, col_idx + 1, col_name)

    for row_idx, (index, row) in enumerate(df.iterrows()):
        data_worksheet.write(row_idx + 1, 0, index)
        for col_idx, value in enumerate(row):
            data_worksheet.write(row_idx + 1, col_idx + 1, value)

    # グラフシート
    chart_worksheet = workbook.add_worksheet("グラフ")

    # ソースファイルごとにチャートを作成
    chart_row = 0
    for file_idx, source_file in enumerate(source_files):
        file_key = os.path.basename(source_file).replace(".csv", "")

        # このファイルに関連するカラムを抽出
        related_columns = [col for col in df.columns if file_key in col or file_key.replace(".csv", "") in col]

        if not related_columns:
            continue

        # 可視化対象のカラムを絞り込み
        if target_sections is not None:
            target_columns = []
            for col in related_columns:
                try:
                    if "-" in col:
                        section = int(col.split("-")[-1])
                        if section in target_sections:
                            target_columns.append(col)
                    else:
                        target_columns.append(col)
                except (ValueError, IndexError):
                    target_columns.append(col)
        else:
            target_columns = related_columns[: min(5, len(related_columns))]  # 最大5系列まで

        if not target_columns:
            continue

        # チャートを作成
        chart = workbook.add_chart({"type": "line"})

        for col in target_columns:
            col_idx = list(df.columns).index(col)
            chart.add_series(
                {
                    "name": col,
                    "categories": ["データ", 1, 0, len(df), 0],
                    "values": ["データ", 1, col_idx + 1, len(df), col_idx + 1],
                    "line": {"width": 2},
                }
            )

        chart.set_title({"name": f"{title_prefix} - {file_key}"})
        chart.set_x_axis({"name": "timestamp"})
        chart.set_size({"width": 600, "height": 400})

        # チャートを配置（2列レイアウト）
        col_position = "A" if file_idx % 2 == 0 else "J"
        row_position = chart_row + 1
        chart_worksheet.insert_chart(f"{col_position}{row_position}", chart)

        if file_idx % 2 == 1:  # 2個ごとに行を移動
            chart_row += 25  # チャートの高さ分だけ行を移動

    workbook.close()
