import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
import glob

def plot_csv_files(tgt_foldapath, unit_dict):
    # csv読み込み
    filename_list = glob.glob(tgt_foldapath + "csv/*")
    df_dict = {}
    for filename in filename_list:
        df_dict[filename[len(tgt_foldapath)+4:-4]] = pd.read_csv(filename, index_col="timestamp")

    # 可視化パラメータ
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
    linewidth_dict = {
        "1": 3,
        "2": 2,
        "3": 1.5,
    }
    num_row = math.ceil(len(df_dict)/3)
    output_foldapath = tgt_foldapath + "png/"
    os.makedirs(output_foldapath, exist_ok=True)

    # 可視化(all)
    fig = plt.figure(figsize=(8*3, 5.5*num_row), tight_layout=True)
    fig.patch.set_facecolor('white')
    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 16
        plt.subplot(num_row,3,i+1)
        for col in df.columns:
            plt.plot(df[col],
                    label = f"({col[-2]}, {col[-1]})",
                    linestyle = linestyle_dict[col[-1]],
                    c = color_dict[col[-2]],
                    #  linewidth = linewidth_dict[col[-1]]
                    )
        plt.title(key + " " + unit_dict[key])
        plt.grid()
        plt.legend()
    plt.savefig(output_foldapath + "all.png", dpi=100)
    plt.close()

    # 可視化(indivisual)
    for i, (key, df) in enumerate(df_dict.items()):
        plt.rcParams["font.size"] = 14
        fig = plt.figure(figsize=(8, 5), tight_layout=True)
        fig.patch.set_facecolor('white')
        for col in df.columns:
            plt.plot(df[col],
                    label = f"({col[-2]}, {col[-1]})",
                    linestyle = linestyle_dict[col[-1]],
                    c = color_dict[col[-2]],
                    #  linewidth = linewidth_dict[col[-1]]
                    )
        plt.title(key + " " + unit_dict[key])
        plt.grid()
        plt.legend()
        plt.savefig(output_foldapath + key + ".png", dpi=100)
        plt.close()