import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
import glob

def plot_csv_files(tgt_foldapath, unit_dict, data_dir, tgt_sections, sheet_name):
    """ csvファイルを可視化する関数
        (stream, section) = (2, 3)のときのみ使用

    Args:
        tgt_foldapath (str): 出力先フォルダパス
        unit_dict (str): 日本語名と単位のdict
        data_dir (str): dataフォルダパス
        tgt_sections (list): 可視化対象のセクション
        sheet_name (str): 観測値エクセルファイルのシート名
    """
    # csv読み込み
    filename_list = glob.glob(tgt_foldapath + "csv/*")
    df_dict = {}
    for filename in filename_list:
        df_dict[filename[len(tgt_foldapath)+4:-4]] = pd.read_csv(filename, index_col="timestamp")

    ### 可視化パラメータ
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
    linewidth_dict = {
        tgt_sections[0]: 3,
        tgt_sections[1]: 2,
        tgt_sections[2]: 1.5,
    }
    output_foldapath = tgt_foldapath + "png/"
    os.makedirs(output_foldapath, exist_ok=True)

    ### 可視化(all) ----------------------------------------------------------------------
    num_row = math.ceil((len(df_dict) + 2)/3)
    fig = plt.figure(figsize=(8*3, 5.5*num_row), tight_layout=True)
    fig.patch.set_facecolor('white')

    # セクション到達温度(吸着層)
    plt.rcParams["font.size"] = 16
    plt.subplot(num_row, 1, 1)
    df = df_dict["セクション到達温度"] # 計算値
    df_obs = pd.read_excel(data_dir + "20240624_ICTへの提供データ_PSA実験_編集_メイン.xlsx", # 観測値
                           sheet_name=sheet_name, index_col="time")
    for stream in range(1,3):
        for section in tgt_sections:
            plt.plot(df[f"temp_reached_{str(stream).zfill(3)}_{str(section).zfill(3)}"],
                    label = f"(str,sec) = ({stream}, {section})",
                    linestyle = linestyle_dict[section],
                    c = color_dict[stream],
                    )
    for stream in range(1,3):
        for i, section in enumerate(range(1,4)):
            plt.plot(df_obs[f"temp_{str(stream).zfill(3)}_{str(section).zfill(3)}"],
                     label=f"(str,sec) = ({stream}, {tgt_sections[i]})",
                     linestyle = linestyle_dict[tgt_sections[i]],
                     c = color_dict_obs[stream]
                     )
    plt.title("セクション到達温度（吸着層）")
    plt.grid()
    plt.legend(fontsize=12)

    # セクション到達温度(壁面)
    plt.rcParams["font.size"] = 16
    plt.subplot(num_row, 3, 4)
    for section in tgt_sections:
        plt.plot(df[f"temp_reached_003_{str(section).zfill(3)}"],
                label = f"(sec) = ({section})",
                linestyle = linestyle_dict[section],
                c = color_dict[3],
                )
    plt.title("セクション到達温度（壁面）")
    plt.grid()
    plt.legend(fontsize=12)

    # セクション到達温度(上下蓋)
    plt.rcParams["font.size"] = 16
    plt.subplot(num_row, 3, 5)
    plt.plot(df[f"temp_reached_up"], label = "up")
    plt.plot(df[f"temp_reached_dw"], label = "dw")
    plt.title("セクション到達温度（上下蓋）")
    plt.grid()
    plt.legend(fontsize=12)

    # その他
    tgt_keys = [key for key in df_dict.keys() if key != "セクション到達温度"]
    # カテゴリごとに可視化
    for i, key in enumerate(tgt_keys):
        plt.rcParams["font.size"] = 16
        plt.subplot(num_row,3,i+6)
        # 可視化対象のcolumnsを抽出
        plt_tgt_cols = [col for col in df_dict[key].columns if int(col.split("_")[-1]) in tgt_sections]
        for col in plt_tgt_cols:
            stream = int(col.split("_")[-2])
            section = int(col.split("_")[-1])
            plt.plot(df_dict[key][col],
                    label = f"(str,sec) = ({stream}, {section})",
                    linestyle = linestyle_dict[section],
                    c = color_dict[stream],
                    #  linewidth = linewidth_dict[section]
                    )
        plt.title(key + " " + unit_dict[key])
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("timestamp")
    plt.savefig(output_foldapath + "all.png", dpi=100)
    plt.close()

    ### 可視化(all_2) ----------------------------------------------------------------------
    # num_row = len(df_dict)
    # num_row += 2 # セクション到達温度を3分割するため
    # fig = plt.figure(figsize=(8*2.5, 5.5*num_row), tight_layout=True)
    # fig.patch.set_facecolor('white')

    # # セクション到達温度（吸着層）
    # plt.rcParams["font.size"] = 16
    # plt.subplot(num_row, 1, 1)
    # df = df_dict["セクション到達温度"]
    # df_obs = pd.read_excel(data_dir + "20240624_ICTへの提供データ_PSA実験_編集_メイン.xlsx",
    #                        sheet_name="python実装用_吸着のみ_立ち上がり修正", index_col="time")
    # for stream in range(1,3):
    #     for section in tgt_sections:
    #         plt.plot(df[f"temp_reached_{str(stream).zfill(3)}_{str(section).zfill(3)}"],
    #                 label = f"(str,sec) = ({stream}, {section})",
    #                 linestyle = linestyle_dict[section],
    #                 c = color_dict[stream],
    #                 )
    # for stream in range(1,3):
    #     for i, section in enumerate(range(1,4)):
    #         plt.plot(df_obs[f"temp_{str(stream).zfill(3)}_{str(section).zfill(3)}"],
    #                  label=f"(str,sec) = ({stream}, {section})",
    #                  linestyle = linestyle_dict[tgt_sections[i]],
    #                  c = color_dict_obs[stream]
    #                  )
    # plt.title("セクション到達温度（吸着層）")
    # plt.grid()
    # plt.xlabel("timestamp")
    # plt.legend(fontsize=12)

    # # セクション到達温度（壁面）
    # plt.rcParams["font.size"] = 16
    # plt.subplot(num_row, 1, 2)
    # for section in tgt_sections:
    #     plt.plot(df[f"temp_reached_003_{str(section).zfill(3)}"],
    #             label = f"(sec) = ({section})",
    #             linestyle = linestyle_dict[section],
    #             c = color_dict[3],
    #             )
    # plt.title("セクション到達温度（壁面）")
    # plt.grid()
    # plt.xlabel("timestamp")
    # plt.legend(fontsize=12)

    # # セクション到達温度（上下蓋）
    # plt.rcParams["font.size"] = 16
    # plt.subplot(num_row, 1, 3)
    # plt.plot(df[f"temp_reached_up"], label="up")
    # plt.plot(df[f"temp_reached_dw"], label="down")
    # plt.title("セクション到達温度（上下蓋）")
    # plt.grid()
    # plt.xlabel("timestamp")
    # plt.legend(fontsize=12)

    # # その他
    # tgt_keys = [key for key in df_dict.keys() if key != "セクション到達温度"]
    # for i, key in enumerate(tgt_keys):
    #     plt.rcParams["font.size"] = 16
    #     plt.subplot(num_row,1,i+4)
    #     plt_tgt_cols = [col for col in df_dict[key].columns if int(col.split("_")[-1]) in tgt_sections]
    #     for col in plt_tgt_cols:
    #         stream = int(col.split("_")[-2])
    #         section = int(col.split("_")[-1])
    #         plt.plot(df_dict[key][col],
    #                 label = f"(str,sec) = ({stream}, {section})",
    #                 linestyle = linestyle_dict[section],
    #                 c = color_dict[stream],
    #                 #  linewidth = linewidth_dict[col[-1]]
    #                 )
    #     plt.title(key + " " + unit_dict[key])
    #     plt.grid()
    #     plt.legend()
    #     plt.xlabel("timestamp")
    # plt.savefig(output_foldapath + "all_2.png", dpi=100)
    # plt.close()

    ### 可視化(indivisual) ----------------------------------------------------------------------
    # # 温度
    # df = df_dict["セクション到達温度"]
    # df_obs = pd.read_excel(data_dir + "20240624_ICTへの提供データ_PSA実験_編集_メイン.xlsx",
    #                        sheet_name="python実装用_吸着のみ_立ち上がり修正", index_col="time")
    # plt.rcParams["font.size"] = 14
    # fig = plt.figure(figsize=(16, 5), tight_layout=True)
    # fig.patch.set_facecolor('white')
    # for stream in range(1,4):
    #     for section in tgt_sections:
    #         plt.plot(df[f"temp_reached_{stream}_{section}"],
    #                 label = f"(str,sec) = ({stream}, {section})",
    #                 linestyle = linestyle_dict[section],
    #                 c = color_dict[stream],
    #                 )
    # for stream in range(1,3):
    #     for section in tgt_sections:
    #         plt.plot(df_obs[f"temp_{stream}_{section}"],
    #                  label=f"(str,sec) = ({stream}, {section})",
    #                  linestyle = linestyle_dict[section],
    #                  c = color_dict_obs[stream]
    #                  )
    # plt.title("セクション到達温度")
    # plt.grid()
    # plt.legend(fontsize=10)
    # plt.savefig(output_foldapath + "セクション到達温度_観測値.png", dpi=100)
    # plt.close()

    # # その他
    # for i, key in enumerate(tgt_keys):
    #     plt.rcParams["font.size"] = 14
    #     fig = plt.figure(figsize=(16, 5), tight_layout=True)
    #     fig.patch.set_facecolor('white')
    #     for col in df_dict[key].columns:
    #         plt.plot(df_dict[key][col],
    #                 label = f"(str,sec) = ({col[-2]}, {col[-1]})",
    #                 linestyle = linestyle_dict[col[-1]],
    #                 c = color_dict[col[-2]],
    #                 #  linewidth = linewidth_dict[col[-1]]
    #                 )
    #     plt.title(key + " " + unit_dict[key])
    #     plt.grid()
    #     plt.legend()
    #     plt.savefig(output_foldapath + key + ".png", dpi=100)
    #     plt.close()