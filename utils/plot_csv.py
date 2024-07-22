import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
import glob

def plot_csv_files(tgt_foldapath, unit_dict, data_dir):
    # csv読み込み
    filename_list = glob.glob(tgt_foldapath + "csv/*")
    df_dict = {}
    for filename in filename_list:
        df_dict[filename[len(tgt_foldapath)+4:-4]] = pd.read_csv(filename, index_col="timestamp")

    ### 可視化パラメータ
    linestyle_dict = {
        "1": "-",
        "2": "--",
        "3": ":",
    }
    color_dict = {
        "1": "tab:red",
        "2": "tab:blue",
        "3": "tab:green",
    }
    color_dict_obs = {
        "1": "black",
        "2": "dimgrey",
    }
    linewidth_dict = {
        "1": 3,
        "2": 2,
        "3": 1.5,
    }
    output_foldapath = tgt_foldapath + "png/"
    os.makedirs(output_foldapath, exist_ok=True)

    ### 可視化(all) ----------------------------------------------------------------------
    num_row = math.ceil(len(df_dict)/3) + 1
    fig = plt.figure(figsize=(8*3, 5.5*num_row), tight_layout=True)
    fig.patch.set_facecolor('white')

    # セクション到達温度
    plt.rcParams["font.size"] = 16
    plt.subplot(num_row, 1, 1)
    df = df_dict["セクション到達温度"]
    df_obs = pd.read_excel(data_dir + "20240624_ICTへの提供データ_PSA実験_編集_メイン.xlsx",
                           sheet_name="python実装用_吸着のみ", index_col="time")
    for stream in range(1,4):
        for section in range(1,4):
            plt.plot(df[f"temp_reached_{str(stream)}{str(section)}"],
                    label = f"(str,sec) = ({str(stream)}, {str(section)})",
                    linestyle = linestyle_dict[str(section)],
                    c = color_dict[str(stream)],
                    )
    for stream in range(1,3):
        for section in range(1,4):
            plt.plot(df_obs[f"temp_{str(stream)}{str(section)}"],
                     label=f"(str,sec) = ({str(stream)}, {str(section)})",
                     linestyle = linestyle_dict[str(section)],
                     c = color_dict_obs[str(stream)]
                     )
    plt.title("セクション到達温度")
    plt.grid()
    plt.legend(fontsize=12)

    # その他
    tgt_keys = [key for key in df_dict.keys() if key != "セクション到達温度"]
    for i, key in enumerate(tgt_keys):
        plt.rcParams["font.size"] = 16
        plt.subplot(num_row,3,i+4)
        for col in df_dict[key].columns:
            plt.plot(df_dict[key][col],
                    label = f"(str,sec) = ({col[-2]}, {col[-1]})",
                    linestyle = linestyle_dict[col[-1]],
                    c = color_dict[col[-2]],
                    #  linewidth = linewidth_dict[col[-1]]
                    )
        plt.title(key + " " + unit_dict[key])
        plt.grid()
        plt.legend(fontsize=12)
    plt.savefig(output_foldapath + "all.png", dpi=100)
    plt.close()

    ### 可視化(all_2) ----------------------------------------------------------------------
    num_row = len(df_dict)
    fig = plt.figure(figsize=(8*2.5, 5.5*num_row), tight_layout=True)
    fig.patch.set_facecolor('white')

    # セクション到達温度
    plt.rcParams["font.size"] = 16
    plt.subplot(num_row, 1, 1)
    df = df_dict["セクション到達温度"]
    df_obs = pd.read_excel(data_dir + "20240624_ICTへの提供データ_PSA実験_編集_メイン.xlsx",
                           sheet_name="python実装用_吸着のみ", index_col="time")
    for stream in range(1,4):
        for section in range(1,4):
            plt.plot(df[f"temp_reached_{str(stream)}{str(section)}"],
                    label = f"(str,sec) = ({str(stream)}, {str(section)})",
                    linestyle = linestyle_dict[str(section)],
                    c = color_dict[str(stream)],
                    )
    for stream in range(1,3):
        for section in range(1,4):
            plt.plot(df_obs[f"temp_{str(stream)}{str(section)}"],
                     label=f"(str,sec) = ({str(stream)}, {str(section)})",
                     linestyle = linestyle_dict[str(section)],
                     c = color_dict_obs[str(stream)]
                     )
    plt.title("セクション到達温度")
    plt.grid()
    plt.legend(fontsize=12)

    # その他
    tgt_keys = [key for key in df_dict.keys() if key != "セクション到達温度"]
    for i, key in enumerate(tgt_keys):
        plt.rcParams["font.size"] = 16
        plt.subplot(num_row,1,i+2)
        for col in df_dict[key].columns:
            plt.plot(df_dict[key][col],
                    label = f"(str,sec) = ({col[-2]}, {col[-1]})",
                    linestyle = linestyle_dict[col[-1]],
                    c = color_dict[col[-2]],
                    #  linewidth = linewidth_dict[col[-1]]
                    )
        plt.title(key + " " + unit_dict[key])
        plt.grid()
        plt.legend()
    plt.savefig(output_foldapath + "all_2.png", dpi=100)
    plt.close()

    ### 可視化(indivisual) ----------------------------------------------------------------------
    # 温度
    df = df_dict["セクション到達温度"]
    df_obs = pd.read_excel(data_dir + "20240624_ICTへの提供データ_PSA実験_編集_メイン.xlsx",
                           sheet_name="python実装用_吸着のみ", index_col="time")
    plt.rcParams["font.size"] = 14
    fig = plt.figure(figsize=(16, 5), tight_layout=True)
    fig.patch.set_facecolor('white')
    for stream in range(1,4):
        for section in range(1,4):
            plt.plot(df[f"temp_reached_{str(stream)}{str(section)}"],
                    label = f"(str,sec) = ({str(stream)}, {str(section)})",
                    linestyle = linestyle_dict[str(section)],
                    c = color_dict[str(stream)],
                    )
    for stream in range(1,3):
        for section in range(1,4):
            plt.plot(df_obs[f"temp_{str(stream)}{str(section)}"],
                     label=f"(str,sec) = ({str(stream)}, {str(section)})",
                     linestyle = linestyle_dict[str(section)],
                     c = color_dict_obs[str(stream)]
                     )
    plt.title("セクション到達温度")
    plt.grid()
    plt.legend(fontsize=10)
    plt.savefig(output_foldapath + "セクション到達温度_観測値.png", dpi=100)
    plt.close()

    # その他
    for i, key in enumerate(tgt_keys):
        plt.rcParams["font.size"] = 14
        fig = plt.figure(figsize=(16, 5), tight_layout=True)
        fig.patch.set_facecolor('white')
        for col in df_dict[key].columns:
            plt.plot(df_dict[key][col],
                    label = f"(str,sec) = ({col[-2]}, {col[-1]})",
                    linestyle = linestyle_dict[col[-1]],
                    c = color_dict[col[-2]],
                    #  linewidth = linewidth_dict[col[-1]]
                    )
        plt.title(key + " " + unit_dict[key])
        plt.grid()
        plt.legend()
        plt.savefig(output_foldapath + key + ".png", dpi=100)
        plt.close()

    # 