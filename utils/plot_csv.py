import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
import glob
from utils import const

def plot_csv_outputs(tgt_foldapath, df_obs, tgt_sections, tower_num, timestamp, df_p_end):
    """ 熱バラ計算結果の可視化

    Args:
        tgt_foldapath (str): 出力先フォルダパス
        df_obs (pd.DataFrame): 観測値のデータフレーム
        tgt_sections (list): 可視化対象のセクション
        tower_num (int): 塔番号
        df_p_end (pd.DataFrame): プロセス終了時刻を含むデータフレーム
    """
    ### パラメータ設定 --------------------------------------

    linestyle_dict = { # section
        tgt_sections[0]: "-",
        tgt_sections[1]: "--",
        tgt_sections[2]: ":",
    }
    color_dict = { # stream
        1: "tab:red",
        2: "tab:blue",
        3: "tab:green",
    }
    color_dict_obs = { # stream (観測値)
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
        df_dict[filename.split("/")[-1][:-4].split("\\")[1]] = pd.read_csv(filename, index_col="timestamp")

    num_row = math.ceil((len(df_dict))/2)
    fig = plt.figure(figsize=(16*2, 5.5*num_row), tight_layout=True)
    fig.patch.set_facecolor('white')

    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 20
        plt.subplot(num_row, 2, i+1)
        # 可視化対象のcolumnsを抽出
        plt_tgt_cols = [col for col in df.columns if int(col.split("-")[-1]) in tgt_sections]
        # 各項目のプロット
        for col in plt_tgt_cols:
            stream = int(col.split("-")[-2])
            section = int(col.split("-")[-1])
            plt.plot(df[col],
                    label = f"(str,sec) = ({stream}, {section})",
                    linestyle = linestyle_dict[section],
                    c = color_dict[stream],
                    )
        plt.title(key + " " + const.UNIT[key])
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("timestamp")
        # セクション到達温度のみ観測値をプロット
        if key == "セクション到達温度":
            for section in range(1,4):
                plt.plot(df_obs.loc[:timestamp, f"T{tower_num}_temp_{section}"],
                        label=f"(str,sec) = (1, {tgt_sections[section-1]})",
                        linestyle = linestyle_dict[tgt_sections[section-1]],
                        c = "black"
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
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(output_foldapath + "heat.png", dpi=100)
    plt.close()

    ### 可視化（マテバラ） -------------------------------------

    # csv読み込み
    filename_list = glob.glob(tgt_foldapath + f"/csv/tower_{tower_num}/material/*.csv")
    df_dict = {}
    for filename in filename_list:
        df_dict[filename.split("/")[-1][:-4].split("\\")[1]] = pd.read_csv(filename, index_col="timestamp")

    num_row = math.ceil((len(df_dict))/2)
    fig = plt.figure(figsize=(16*2, 5.5*num_row), tight_layout=True)
    fig.patch.set_facecolor('white')

    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 20
        plt.subplot(num_row, 2, i+1)
        # 可視化対象のcolumnsを抽出
        plt_tgt_cols = [col for col in df.columns if int(col.split("-")[-1]) in tgt_sections]
        # 各項目のプロット
        for col in plt_tgt_cols:
            stream = int(col.split("-")[-2])
            section = int(col.split("-")[-1])
            plt.plot(df[col],
                    label = f"(str,sec) = ({stream}, {section})",
                    linestyle = linestyle_dict[section],
                    c = color_dict[stream],
                    )
        plt.title(key + " " + const.UNIT[key])
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
    filename_list = glob.glob(tgt_foldapath + f"/csv/tower_{tower_num}/heat_lid/heat_lid.csv")
    df = pd.read_csv(filename_list[0], index_col="timestamp")

    num_row = math.ceil((len(df.columns))/2)
    fig = plt.figure(figsize=(16*2, 5.5*num_row), tight_layout=True)
    fig.patch.set_facecolor('white')

    for i, col in enumerate(df.columns):
        plt.rcParams["font.size"] = 20
        plt.subplot(num_row, 2, i+1)
        plt.plot(df[col])
        title = const.TRANSLATION[col.split("-")[0]]
        unit = const.UNIT[title]
        if col.split("-")[1] == "up":
            title += "_上蓋"
        else:
            title += "_下蓋"
        plt.title(title + " " + unit)
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

    fig = plt.figure(figsize=(16*2, 5.5*3), tight_layout=True)
    fig.patch.set_facecolor('white')
    plt.rcParams["font.size"] = 20
    plt_cell = 1

    # 1. 全圧
    filename = tgt_foldapath + f"/csv/tower_{tower_num}/others/total_press.csv"
    df = pd.read_csv(filename, index_col="timestamp")
    plt.subplot(3, 2, plt_cell)
    plt.plot(df["total_press"], label="計算値")
    plt.plot(df_obs.loc[:timestamp, f"T{tower_num}_press"], label="観測値", c="black") # 観測値もプロット
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
    for _tgt_name in ["mf_co2", "mf_n2"]:
        filename = tgt_foldapath + f"/csv/tower_{tower_num}/others/{_tgt_name}.csv"
        df = pd.read_csv(filename, index_col="timestamp")
        plt.subplot(3, 2, plt_cell)
        # 可視化対象のcolumnsを抽出
        plt_tgt_cols = [col for col in df.columns if int(col.split("-")[-1]) in tgt_sections]
        # 各項目のプロット
        for col in plt_tgt_cols:
            stream = int(col.split("-")[-2])
            section = int(col.split("-")[-1])
            plt.plot(df[col],
                    label = f"(str,sec) = ({stream}, {section})",
                    linestyle = linestyle_dict[section],
                    c = color_dict[stream],
                    )
        plt.title(_tgt_name.split("_")[-1] + "モル分率")
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
    title_list = ["CO2回収量 [mol]", "N2回収量 [mol]", "CO2回収率 [%]"]
    for i, _tgt_name in enumerate(["vacuum_amt_co2", "vacuum_amt_n2", "vacuum_rate_co2"]):
        filename = tgt_foldapath + f"/csv/tower_{tower_num}/others/vacuum_amount.csv"
        df = pd.read_csv(filename, index_col="timestamp")
        plt.subplot(3, 2, plt_cell)
        plt.plot(df[_tgt_name], label="計算値")
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


def outputs_to_csv(tgt_foldapath, record_dict, common_conds):
    """ 計算結果をcsv出力する

    Args:
        tgt_foldapath (str): 出力先フォルダパス
        record_dict (dict): 計算結果
        common_conds (dict): 実験パラメータ
    """
    ### heat, material ----------------------------------------------
    for _tgt_name in ["heat", "material"]:
        _foldapath = tgt_foldapath + f"/{_tgt_name}/"
        os.makedirs(_foldapath, exist_ok=True)
        # 値の抽出
        values = []
        for i in range(len(record_dict[_tgt_name])):
            values_tmp = []
            for stream in range(1, 1+common_conds["CELL_SPLIT"]["num_str"]):
                for section in range(1, 1+common_conds["CELL_SPLIT"]["num_sec"]):
                    for value in record_dict[_tgt_name][i][stream][section].values():
                        values_tmp.append(value)
            values.append(values_tmp)
        # カラム名の抽出
        columns = []
        for stream in range(1, 1+common_conds["CELL_SPLIT"]["num_str"]):
            for section in range(1, 1+common_conds["CELL_SPLIT"]["num_sec"]):
                for key in record_dict[_tgt_name][i][stream][section].keys():
                    columns.append(key+"-"+str(stream).zfill(3)+"-"+str(section).zfill(3))
        # df化
        for key in record_dict[_tgt_name][i][stream][section].keys():
            idx = [columns.index(col) for col in columns if key in col]
            df = pd.DataFrame(np.array(values)[:, idx],
                                columns=np.array(columns)[idx],
                                index=record_dict["timestamp"])
            df.index.name = "timestamp"
            df.to_csv(_foldapath + const.TRANSLATION[key] + ".csv")

    # heat_lid
    foldapath = tgt_foldapath + f"/heat_lid/"
    os.makedirs(foldapath, exist_ok=True)
    values = []
    for i in range(len(record_dict["heat_lid"])):
        values.append([
            record_dict["heat_lid"][i]["up"]["temp_reached"],
            record_dict["heat_lid"][i]["down"]["temp_reached"]
        ])
    columns = ["temp_reached-up", "temp_reached-down"]
    df = pd.DataFrame(values,
                        columns=columns,
                        index=record_dict["timestamp"])
    df.index.name = "timestamp"
    df.to_csv(foldapath + "heat_lid.csv")

    ### others ----------------------------------------------
    tgt_name = "others"
    _foldapath = tgt_foldapath + f"/{tgt_name}/"
    os.makedirs(_foldapath, exist_ok=True)
    # 全圧
    values = []
    _tgt_col = "total_press"
    for i in range(len(record_dict[tgt_name])):
        values.append(record_dict[tgt_name][i][_tgt_col])
    df = pd.DataFrame(values,
                      columns=[_tgt_col],
                      index=record_dict["timestamp"])
    df.index.name = "timestamp"
    df.to_csv(_foldapath + f"{_tgt_col}.csv")
    # CO2, N2回収量
    values = []
    _tgt_col = "vacuum_amount"
    for i in range(len(record_dict[tgt_name])):
        try:
            _vacuum_rate_co2 = (
                record_dict[tgt_name][i]["vacuum_amt_co2"] / (
                    record_dict[tgt_name][i]["vacuum_amt_co2"] + record_dict[tgt_name][i]["vacuum_amt_n2"]
                )) * 100
        except ZeroDivisionError:
            _vacuum_rate_co2 = 0
        values.append([_vacuum_rate_co2,
                       record_dict[tgt_name][i]["vacuum_amt_co2"],
                       record_dict[tgt_name][i]["vacuum_amt_n2"]])
    df = pd.DataFrame(values,
                      columns=["vacuum_rate_co2", "vacuum_amt_co2", "vacuum_amt_n2"],
                      index=record_dict["timestamp"])
    df.index.name = "timestamp"
    df.to_csv(_foldapath + f"{_tgt_col}.csv")
    # モル分率
    for _tgt_col in ["mf_co2", "mf_n2"]:
        values = []
        for i in range(len(record_dict[tgt_name])):
            values_tmp = []
            for stream in range(1, 1+common_conds["CELL_SPLIT"]["num_str"]):
                for section in range(1, 1+common_conds["CELL_SPLIT"]["num_sec"]):
                    values_tmp.append(record_dict[tgt_name][i][_tgt_col][stream][section])
            values.append(values_tmp)
        # カラム名の抽出
        columns = []
        for stream in range(1, 1+common_conds["CELL_SPLIT"]["num_str"]):
            for section in range(1, 1+common_conds["CELL_SPLIT"]["num_sec"]):
                columns.append(_tgt_col+"-"+str(stream).zfill(3)+"-"+str(section).zfill(3))
        df = pd.DataFrame(values,
                        columns=columns,
                        index=record_dict["timestamp"])
        df.index.name = "timestamp"
        df.to_csv(_foldapath + f"{_tgt_col}.csv")

    # heat_wall