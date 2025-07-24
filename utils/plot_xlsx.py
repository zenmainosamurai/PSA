import os
import pandas as pd
import numpy as np
import glob
from utils import const
import xlsxwriter

from config.sim_conditions import CommonConditions


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
    values,
    columns,
    timestamp_index,
    file_path,
    add_units=True,
    chart_title="",
    tgt_sections=None,
    observed_data=None,
    tower_num=None,
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
        # タイムスタンプを数値として書き込み（Excelで時系列として認識させるため）
        data_worksheet.write(row_idx + 1, 0, float(index))
        for col_idx, value in enumerate(row):
            data_worksheet.write(row_idx + 1, col_idx + 1, value)

    # 実測値シートの作成（observed_dataが提供されている場合）
    observed_worksheet = None
    if observed_data is not None and not observed_data.empty:
        # 計算値のタイムスタンプより後の実測値データを削除
        if len(df) > 0:
            calc_max_time = df.index.max()
            observed_data = observed_data[observed_data.index <= calc_max_time]

        # 実測値データを計算値のタイムスタンプに線形補間
        if len(df) > 0:
            observed_data = _interpolate_observed_data_to_calc_timestamps(observed_data, df.index)

        # 削除後にデータが残っている場合のみシートを作成
        if observed_data is not None and not observed_data.empty:
            observed_worksheet = workbook.add_worksheet("実測値")

            # 実測値データをExcelに書き込み
            observed_worksheet.write(0, 0, "timestamp")
            for col_idx, col_name in enumerate(observed_data.columns):
                observed_worksheet.write(0, col_idx + 1, col_name)

            for row_idx, (index, row) in enumerate(observed_data.iterrows()):
                # タイムスタンプを数値として書き込み（Excelで時系列として認識させるため）
                observed_worksheet.write(row_idx + 1, 0, float(index))
                for col_idx, value in enumerate(row):
                    observed_worksheet.write(row_idx + 1, col_idx + 1, value)
        else:
            # データが全て削除された場合は observed_data を None に設定
            observed_data = None

    # グラフシートの作成
    chart_worksheet = workbook.add_worksheet("グラフ")

    # Excelネイティブグラフの作成
    if len(df.columns) > 0:
        chart = workbook.add_chart({"type": "line"})

        # 可視化対象のカラムを抽出（tgt_sectionsが指定されている場合）
        if tgt_sections is not None:
            target_columns = []
            for col in df.columns:
                try:
                    if "-" in col:
                        section = int(col.split("-")[-1])
                        if section in tgt_sections:
                            target_columns.append(col)
                    else:
                        target_columns.append(col)
                except (ValueError, IndexError):
                    target_columns.append(col)
        else:
            target_columns = df.columns[: min(10, len(df.columns))]  # 最大10系列まで

        # 各系列をグラフに追加（計算値）
        for col_idx, col_name in enumerate(target_columns):
            chart.add_series(
                {
                    "name": col_name,  # 実測値と命名を統一
                    "categories": ["データ", 1, 0, len(df), 0],  # timestamp列
                    "values": [
                        "データ",
                        1,
                        list(df.columns).index(col_name) + 1,
                        len(df),
                        list(df.columns).index(col_name) + 1,
                    ],
                    "line": {"width": 0.5},
                }
            )

        # 実測値をグラフに追加（observed_dataが提供されている場合）
        if observed_data is not None and not observed_data.empty:
            for col_idx, col_name in enumerate(observed_data.columns):
                chart.add_series(
                    {
                        "name": col_name,  # カラム名に既に「実測」が含まれているため
                        "categories": ["実測値", 1, 0, len(observed_data), 0],  # timestamp列
                        "values": [
                            "実測値",
                            1,
                            col_idx + 1,
                            len(observed_data),
                            col_idx + 1,
                        ],
                        "line": {"width": 0.5, "color": "black"},
                    }
                )

        chart.set_title({"name": chart_title})
        # 横軸の範囲を計算値データに合わせて設定
        if len(df) > 0:
            x_min = df.index.min()
            x_max = df.index.max()
            chart.set_x_axis({"name": "timestamp", "min": x_min, "max": x_max})
        else:
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


def _get_observed_data_for_key(df_obs, key, tower_num, tgt_sections=None):
    """指定されたキーと塔番号に対応する実測値データを取得"""
    if df_obs is None or df_obs.empty:
        return None

    observed_data = pd.DataFrame()
    observed_data.index = df_obs.index

    # 温度データの場合
    if "セクション到達温度" in key or "熱電対温度" in key:
        # plot_csvと同じ仕様：tgt_sectionsの値を使用
        if tgt_sections is not None:
            for section in range(1, 4):  # section 1, 2, 3
                target_section = tgt_sections[section - 1]  # 実際のセクション番号
                temp_col = f"T{tower_num}_temp_{section}"
                if temp_col in df_obs.columns:
                    # シミュレーション値と同じ形式で「実測」を先頭に付ける
                    sim_column_name = f"{key}[℃]-001-{target_section:03d}"
                    observed_data[f"実測{sim_column_name}"] = df_obs[temp_col]

    # 圧力データの場合
    elif "全圧" in key:
        press_col = f"T{tower_num}_press"
        if press_col in df_obs.columns:
            # シミュレーション値と同じ形式で「実測」を先頭に付ける
            sim_column_name = f"{key}[MPaA]"
            observed_data[f"実測{sim_column_name}"] = df_obs[press_col]

    return observed_data if not observed_data.empty else None


def _interpolate_observed_data_to_calc_timestamps(observed_data, calc_timestamps):
    """実測値データを計算値のタイムスタンプに線形補間する"""
    if observed_data is None or observed_data.empty:
        return None

    # 線形補間を実行
    interpolated_data = pd.DataFrame(index=calc_timestamps)

    for col in observed_data.columns:
        # NaNでない値のみを使用して補間
        valid_data = observed_data[col].dropna()
        if len(valid_data) > 1:
            # 計算値のタイムスタンプ範囲内のデータのみに限定
            calc_min, calc_max = calc_timestamps.min(), calc_timestamps.max()

            # 補間実行（計算値の範囲内のみ）
            interpolated_values = np.interp(
                calc_timestamps,
                valid_data.index,
                valid_data.values,
                left=np.nan,  # 範囲外は NaN
                right=np.nan,  # 範囲外は NaN
            )

            # 計算値の範囲外は NaN にする
            mask = (calc_timestamps >= valid_data.index.min()) & (calc_timestamps <= valid_data.index.max())
            interpolated_values[~mask] = np.nan

            interpolated_data[col] = interpolated_values

    # すべての列がNaNの行を削除
    interpolated_data = interpolated_data.dropna(how="all")

    return interpolated_data if not interpolated_data.empty else None


def _save_heat_material_data_xlsx(
    tgt_foldapath, tower_results, common_conds, data_type, tgt_sections=None, df_obs=None, tower_num=1
):
    """heat/materialデータをXLSXに保存する"""
    folder_path = os.path.join(tgt_foldapath, data_type)
    os.makedirs(folder_path, exist_ok=True)

    record_data = getattr(tower_results.time_series_data, data_type)
    values = _extract_heat_material_values(record_data, common_conds)
    columns = _generate_heat_material_columns(record_data, common_conds)

    # キーごとにXLSXファイルを作成
    sample_result_data = record_data[0].get_result(1, 1).to_dict()
    for key in sample_result_data.keys():
        # より厳密な条件でkey_indicesを抽出（完全一致）
        key_indices = [i for i, col in enumerate(columns) if col.startswith(f"{key}-")]
        key_values = np.array(values)[:, key_indices]
        key_columns = [columns[i] for i in key_indices]

        japanese_name = const.TRANSLATION.get(key, key)
        file_path = os.path.join(folder_path, f"{japanese_name}.xlsx")
        chart_title = f"{japanese_name} {const.UNIT.get(japanese_name, '')}"

        # 実測値データを取得
        observed_data = _get_observed_data_for_key(df_obs, japanese_name, tower_num, tgt_sections)

        _create_dataframe_and_save_xlsx(
            key_values,
            key_columns,
            tower_results.time_series_data.timestamps,
            file_path,
            add_units=True,
            chart_title=chart_title,
            tgt_sections=tgt_sections,
            observed_data=observed_data,
            tower_num=tower_num,
        )


def _save_heat_lid_data_xlsx(tgt_foldapath, tower_results):
    """heat_lidデータをXLSXに保存する"""
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
    file_path = os.path.join(folder_path, "蓋温度.xlsx")
    chart_title = f"蓋温度 {const.UNIT.get('蓋温度', '[℃]')}"
    _create_dataframe_and_save_xlsx(
        values, columns, tower_results.time_series_data.timestamps, file_path, add_units=True, chart_title=chart_title
    )


def _calculate_vacuum_rate_co2(cumulative_co2, cumulative_n2):
    """CO2回収率を計算する"""
    try:
        total = cumulative_co2 + cumulative_n2
        return (cumulative_co2 / total) * 100 if total != 0 else 0
    except (ZeroDivisionError, TypeError):
        return 0


def _save_total_pressure_data_xlsx(folder_path, tower_results, df_obs=None, tower_num=1, tgt_sections=None):
    """全圧データをXLSXに保存する"""
    values = [[record["total_pressure"]] for record in tower_results.time_series_data.others]

    # 実測値データを取得
    observed_data = _get_observed_data_for_key(df_obs, "全圧", tower_num, tgt_sections)

    file_path = os.path.join(folder_path, "全圧.xlsx")
    chart_title = f"全圧 {const.UNIT.get('全圧', '[MPaA]')}"
    _create_dataframe_and_save_xlsx(
        values,
        ["total_pressure"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title=chart_title,
        observed_data=observed_data,
        tower_num=tower_num,
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
    chart_title = f"CO2回収率 {const.UNIT.get('CO2回収率', '[%]')}"
    _create_dataframe_and_save_xlsx(
        vacuum_rate_co2_values,
        ["vacuum_rate_co2"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title=chart_title,
    )

    # 累積CO2回収量を個別ファイルに保存
    cumulative_co2_values = [[record["cumulative_co2_recovered"]] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "累積CO2回収量.xlsx")
    chart_title = f"累積CO2回収量 {const.UNIT.get('累積CO2回収量', '[Nm3]')}"
    _create_dataframe_and_save_xlsx(
        cumulative_co2_values,
        ["cumulative_co2_recovered"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title=chart_title,
    )

    # 累積N2回収量を個別ファイルに保存
    cumulative_n2_values = [[record["cumulative_n2_recovered"]] for record in tower_results.time_series_data.others]
    file_path = os.path.join(folder_path, "累積N2回収量.xlsx")
    chart_title = f"累積N2回収量 {const.UNIT.get('累積N2回収量', '[Nm3]')}"
    _create_dataframe_and_save_xlsx(
        cumulative_n2_values,
        ["cumulative_n2_recovered"],
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title=chart_title,
    )


def _save_mole_fraction_data_xlsx(folder_path, tower_results, common_conds, fraction_type, tgt_sections=None):
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
    chart_title = f"{japanese_name} {const.UNIT.get(japanese_name, '[-]')}"

    _create_dataframe_and_save_xlsx(
        values,
        columns,
        tower_results.time_series_data.timestamps,
        file_path,
        add_units=True,
        chart_title=chart_title,
        tgt_sections=tgt_sections,
    )


def _save_others_data_xlsx(tgt_foldapath, tower_results, common_conds, tgt_sections=None, df_obs=None, tower_num=1):
    """othersデータをXLSXに保存する"""
    folder_path = os.path.join(tgt_foldapath, "others")
    os.makedirs(folder_path, exist_ok=True)

    # 全圧
    _save_total_pressure_data_xlsx(folder_path, tower_results, df_obs, tower_num, tgt_sections)

    # CO2, N2回収量
    _save_vacuum_amount_data_xlsx(folder_path, tower_results)

    # モル分率
    for fraction_type in ["co2_mole_fraction", "n2_mole_fraction"]:
        _save_mole_fraction_data_xlsx(folder_path, tower_results, common_conds, fraction_type, tgt_sections)


def outputs_to_xlsx(
    tgt_foldapath, tower_results, common_conds: CommonConditions, tgt_sections=None, df_obs=None, tower_num=1
):
    """計算結果をxlsx出力する

    Args:
        tgt_foldapath (str): 出力先フォルダパス
        tower_results (TowerSimulationResults): 計算結果
        common_conds (CommonConditions): 実験パラメータ
        tgt_sections (list): 可視化対象のセクション
        df_obs (pd.DataFrame): 観測値のデータフレーム
        tower_num (int): 塔番号
    """
    # heat, material データの保存
    for data_type in ["heat", "material"]:
        _save_heat_material_data_xlsx(
            tgt_foldapath, tower_results, common_conds, data_type, tgt_sections, df_obs, tower_num
        )

    # heat_lid データの保存
    _save_heat_lid_data_xlsx(tgt_foldapath, tower_results)

    # others データの保存
    _save_others_data_xlsx(tgt_foldapath, tower_results, common_conds, tgt_sections, df_obs, tower_num)


def plot_xlsx_outputs(tgt_foldapath, df_obs, tgt_sections, tower_num, timestamp, df_p_end):
    """熱バラ計算結果のxlsx可視化（統合ファイルは作成しない）

    Args:
        tgt_foldapath (str): 出力先フォルダパス
        df_obs (pd.DataFrame): 観測値のデータフレーム
        tgt_sections (list): 可視化対象のセクション
        tower_num (int): 塔番号
        df_p_end (pd.DataFrame): プロセス終了時刻を含むデータフレーム
    """
    # 統合ファイルは作成しない
    pass
