import numpy as np
import pandas as pd

import base_models

import warnings
warnings.simplefilter('ignore')


def initial_adsorption(sim_conds, stream_conds, variables):
    """ 吸着開始時の圧力調整 
        説明: 規定圧力に達するまでガスを導入する（吸着も起こる）
        ベースモデル: バッチ吸着モデル
        終了条件: 圧力到達

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 0 # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        mb_dict[stream][1] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                 stream_conds=stream_conds,
                                                                 stream=stream,
                                                                 section=1,
                                                                 variables=variables,
                                                                 inflow_gas=None)
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                      stream_conds=stream_conds,
                                                      stream=stream,
                                                      section=1,
                                                      variables=variables,
                                                      mode=mode,
                                                      material_output=mb_dict[stream][1],
                                                      heat_output=None,
                                                      vaccume_output=None)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                           stream_conds=stream_conds,
                                                                           stream=stream,
                                                                           section=section,
                                                                           variables=variables,
                                                                           inflow_gas=mb_dict[stream][section-1])
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                stream_conds=stream_conds,
                                                                stream=stream,
                                                                section=section,
                                                                variables=variables,
                                                                mode=mode,
                                                                material_output=mb_dict[stream][section],
                                                                heat_output=hb_dict[stream][section-1],
                                                                vaccume_output=None)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                               stream_conds=stream_conds,
                                               section=1,
                                               variables=variables,
                                               heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                               heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                         stream_conds=stream_conds,
                                                         section=section,
                                                         variables=variables,
                                                         heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                         heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                        position=position,
                                                        variables=variables,
                                                        heat_output=hb_dict,
                                                        heat_wall_output=hb_wall)
    # 4. バッチ吸着後圧力変化
    total_press_after_batch_adsorp = base_models.total_press_after_batch_adsorp(sim_conds=sim_conds,
                                                                                variables=variables)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
        "total_press_after_batch_adsorp": total_press_after_batch_adsorp, # バッチ吸着後圧力
    }

    return output

def stop_mode(sim_conds, stream_conds, variables):
    """ 停止
        説明: モードとモードの間の吸着停止期間
        ベースモデル: 脱着モデル
        終了条件: 時間経過
        補足: 真空脱着との違いは排気後圧力計算で排気速度が0である点のみ

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 2 # 脱着

    ### セル計算 --------------------------------------------------------------

    # 0. 排気後圧力の計算
    vaccume_output = base_models.total_press_after_vacuuming(sim_conds=sim_conds,
                                                             variables=variables,
                                                             stop_mode=True) # 排気速度=0
    # tgt_index = df_obs.index[np.argmin(np.abs(df_obs.index - timestamp))] # 圧力を観測値で上書き
    # vaccume_output["total_press_after_vaccume"] = df_obs.loc[tgt_index, "Pressure"] / 1000 # 圧力[Mpa]
    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        mb_dict[stream][1] = base_models.material_balance_desorp(sim_conds=sim_conds,
                                                                 stream_conds=stream_conds,
                                                                 stream=stream,
                                                                 section=1,
                                                                 variables=variables,
                                                                 vaccume_output=vaccume_output)
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                      stream_conds=stream_conds,
                                                      stream=stream,
                                                      section=1,
                                                      variables=variables,
                                                      mode=mode,
                                                      material_output=mb_dict[stream][1],
                                                      heat_output=None,
                                                      vaccume_output=vaccume_output)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_desorp(sim_conds=sim_conds,
                                                                           stream_conds=stream_conds,
                                                                           stream=stream,
                                                                           section=section,
                                                                           variables=variables,
                                                                           vaccume_output=vaccume_output)
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                stream_conds=stream_conds,
                                                                stream=stream,
                                                                section=section,
                                                                variables=variables,
                                                                mode=mode,
                                                                material_output=mb_dict[stream][section],
                                                                heat_output=hb_dict[stream][section-1],
                                                                vaccume_output=vaccume_output)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                               stream_conds=stream_conds,
                                               section=1,
                                               variables=variables,
                                               heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                               heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                         stream_conds=stream_conds,
                                                         section=section,
                                                         variables=variables,
                                                         heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                         heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                        position=position,
                                                        variables=variables,
                                                        heat_output=hb_dict,
                                                        heat_wall_output=hb_wall)
    # 4. 排気後モル分率計算
    mf_after_vaccume = base_models.mf_after_vaccume(sim_conds=sim_conds,
                                                    vaccume_output=vaccume_output,
                                                    mb_dict=mb_dict,)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
        "total_press_and_mf": {**vaccume_output, **mf_after_vaccume}, # 排気結果
    }

    return output

def flow_adsorption_single_or_upstream(sim_conds, stream_conds, variables):
    """ 流通吸着_単独/直列吸着_上流
        説明: 1つの吸着塔の流通吸着 or 2つの吸着塔の直列吸着における上流
        ベースモデル: 流通吸着モデル
        終了条件: 時間経過・温度上昇

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 0 # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        mb_dict[stream][1] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                      stream_conds=stream_conds,
                                                                      stream=stream,
                                                                      section=1,
                                                                      variables=variables,
                                                                      inflow_gas=None)
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                           stream_conds=stream_conds,
                                                           stream=stream,
                                                           section=1,
                                                           variables=variables,
                                                           mode=mode,
                                                           material_output=mb_dict[stream][1],
                                                           heat_output=None,
                                                           vaccume_output=None)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                                stream_conds=stream_conds,
                                                                                stream=stream,
                                                                                section=section,
                                                                                variables=variables,
                                                                                inflow_gas=mb_dict[stream][section-1])
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                     stream_conds=stream_conds,
                                                                     stream=stream,
                                                                     section=section,
                                                                     variables=variables,
                                                                     mode=mode,
                                                                     material_output=mb_dict[stream][section],
                                                                     heat_output=hb_dict[stream][section-1],
                                                                     vaccume_output=None)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                    stream_conds=stream_conds,
                                                    section=1,
                                                    variables=variables,
                                                    heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                                    heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                              stream_conds=stream_conds,
                                                              section=section,
                                                              variables=variables,
                                                              heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                              heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                             position=position,
                                                             variables=variables,
                                                             heat_output=hb_dict,
                                                             heat_wall_output=hb_wall)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
    }

    return output

def batch_adsorption_upstream(sim_conds, stream_conds, variables):
    """ バッチ吸着_上流
        説明: ２つの吸着塔のバッチ吸着における上流側（下流側の圧力回復が目的）
        ベースモデル: バッチ吸着モデル
        終了条件: 下流の温度上昇

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻
    """
    # 初期のバッチ吸着と同じ仕組み
    return initial_adsorption(sim_conds, stream_conds, variables)

def equalization_pressure_depressurization(sim_conds, stream_conds, variables,
                                           downflow_total_press):
    """ バッチ均圧（減圧）
        説明: 均圧における減圧側
        ベースモデル: 脱着モデル
        終了条件: 圧力到達・時間経過

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 2 # 脱着

    ### セル計算 --------------------------------------------------------------

    vaccume_output = {}
    vaccume_output["total_press_after_vaccume"] = variables["total_press"]
    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        mb_dict[stream][1] = base_models.material_balance_desorp(sim_conds=sim_conds,
                                                                 stream_conds=stream_conds,
                                                                 stream=stream,
                                                                 section=1,
                                                                 variables=variables,
                                                                 vaccume_output=vaccume_output)
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                      stream_conds=stream_conds,
                                                      stream=stream,
                                                      section=1,
                                                      variables=variables,
                                                      mode=mode,
                                                      material_output=mb_dict[stream][1],
                                                      heat_output=None,
                                                      vaccume_output=vaccume_output)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_desorp(sim_conds=sim_conds,
                                                                           stream_conds=stream_conds,
                                                                           stream=stream,
                                                                           section=section,
                                                                           variables=variables,
                                                                           vaccume_output=vaccume_output)
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                stream_conds=stream_conds,
                                                                stream=stream,
                                                                section=section,
                                                                variables=variables,
                                                                mode=mode,
                                                                material_output=mb_dict[stream][section],
                                                                heat_output=hb_dict[stream][section-1],
                                                                vaccume_output=vaccume_output)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                               stream_conds=stream_conds,
                                               section=1,
                                               variables=variables,
                                               heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                               heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                         stream_conds=stream_conds,
                                                         section=section,
                                                         variables=variables,
                                                         heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                         heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                        position=position,
                                                        variables=variables,
                                                        heat_output=hb_dict,
                                                        heat_wall_output=hb_wall)
    # 4. 減圧後圧力・モル分率
    total_press_and_mf = base_models.total_press_after_decompression(sim_conds=sim_conds,
                                                                     variables=variables,
                                                                     downflow_total_press=downflow_total_press,
                                                                     mb_dict=mb_dict)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
        "total_press_and_mf": total_press_and_mf, # 減圧後圧力・モル分率
    }

    return output

def desorption_by_vaccuming(sim_conds, stream_conds, variables):
    """ 真空脱着
        説明: 真空引きによる脱着
        ベースモデル: 脱着モデル
        終了条件: 時間経過
        補足: 停止モードとほぼ同じだが、排気速度が0ではない点のみ異なる

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 2 # 脱着

    ### セル計算 --------------------------------------------------------------

    # 0. 排気後圧力の計算
    vaccume_output = base_models.total_press_after_vacuuming(sim_conds=sim_conds,
                                                             variables=variables,
                                                             stop_mode=False) # 排気速度が0ではない
    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        mb_dict[stream][1] = base_models.material_balance_desorp(sim_conds=sim_conds,
                                                                 stream_conds=stream_conds,                                                                      
                                                                 stream=stream,
                                                                 section=1,
                                                                 variables=variables,
                                                                 vaccume_output=vaccume_output)
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                      stream_conds=stream_conds,
                                                      stream=stream,
                                                      section=1,
                                                      variables=variables,
                                                      mode=mode,
                                                      material_output=mb_dict[stream][1],
                                                      heat_output=None,
                                                      vaccume_output=vaccume_output)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_desorp(sim_conds=sim_conds,
                                                                           stream_conds=stream_conds,
                                                                           stream=stream,
                                                                           section=section,
                                                                           variables=variables,
                                                                           vaccume_output=vaccume_output)
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                stream_conds=stream_conds,
                                                                stream=stream,
                                                                section=section,
                                                                variables=variables,
                                                                mode=mode,
                                                                material_output=mb_dict[stream][section],
                                                                heat_output=hb_dict[stream][section-1],
                                                                vaccume_output=vaccume_output)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                               stream_conds=stream_conds,
                                               section=1,
                                               variables=variables,
                                               heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                               heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                         stream_conds=stream_conds,
                                                         section=section,
                                                         variables=variables,
                                                         heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                         heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                        position=position,
                                                        variables=variables,
                                                        heat_output=hb_dict,
                                                        heat_wall_output=hb_wall)
    # 4. 排気後モル分率計算
    mf_after_vaccume = base_models.mf_after_vaccume(sim_conds=sim_conds,
                                                    vaccume_output=vaccume_output,
                                                    mb_dict=mb_dict)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
        "total_press_and_mf": {**vaccume_output, **mf_after_vaccume}, # 排気結果
    }

    return output

def equalization_pressure_pressurization(sim_conds, stream_conds, variables,
                                         upstream_flow_amount):
    """ バッチ均圧（加圧）
        説明: 均圧における加圧側
        ベースモデル: バッチ吸着モデル
        終了条件: 圧力到達・時間経過
        補足: バッチ吸着とほぼ同じだが、バッチ吸着後の圧力変化に上流側の均圧菅流量を使用している

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 0 # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        mb_dict[stream][1] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                      stream_conds=stream_conds,
                                                                      stream=stream,
                                                                      section=1,
                                                                      variables=variables,
                                                                      inflow_gas=None)
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                           stream_conds=stream_conds,
                                                           stream=stream,
                                                           section=1,
                                                           variables=variables,
                                                           mode=mode,
                                                           material_output=mb_dict[stream][1],
                                                           heat_output=None,
                                                           vaccume_output=None)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                                stream_conds=stream_conds,
                                                                                stream=stream,
                                                                                section=section,
                                                                                variables=variables,
                                                                                inflow_gas=mb_dict[stream][section-1])
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                     stream_conds=stream_conds,
                                                                     stream=stream,
                                                                     section=section,
                                                                     variables=variables,
                                                                     mode=mode,
                                                                     material_output=mb_dict[stream][section],
                                                                     heat_output=hb_dict[stream][section-1],
                                                                     vaccume_output=None)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                    stream_conds=stream_conds,
                                                    section=1,
                                                    variables=variables,
                                                    heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                                    heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                              stream_conds=stream_conds,
                                                              section=section,
                                                              variables=variables,
                                                              heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                              heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                             position=position,
                                                             variables=variables,
                                                             heat_output=hb_dict,
                                                             heat_wall_output=hb_wall)
    # 4. バッチ吸着後圧力変化
    total_press_after_batch_adsorp = base_models.total_press_after_batch_adsorp(sim_conds=sim_conds,
                                                                                variables=variables,
                                                                                equalization_mode=True,
                                                                                upstream_flow_amount=upstream_flow_amount)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
        "total_press_after_batch_adsorp": total_press_after_batch_adsorp, # バッチ吸着後圧力
    }

    return output

def batch_adsorption_downstream(sim_conds, stream_conds, variables,
                                inflow_gas):
    """ バッチ吸着（下流）
        説明: ２つの吸着塔のバッチ吸着における下流側（下流側の圧力回復が目的）
        ベースモデル: バッチ吸着モデル
        終了条件: 下流の温度上昇
        補足: 通常のバッチ吸着とほぼ同じだが、section1の流入ガスが上流の流出ガスになっている点のみ異なる

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 0 # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        # NOTE: # 流入ガスが上流の流出ガス
        mb_dict[stream][1] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                 stream_conds=stream_conds,
                                                                 stream=stream,
                                                                 section=1,
                                                                 variables=variables,
                                                                 inflow_gas=inflow_gas[stream][sim_conds["CELL_SPLIT"]["num_sec"]])
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                      stream_conds=stream_conds,
                                                      stream=stream,
                                                      section=1,
                                                      variables=variables,
                                                      mode=mode,
                                                      material_output=mb_dict[stream][1],
                                                      heat_output=None,
                                                      vaccume_output=None)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                           stream_conds=stream_conds,
                                                                           stream=stream,
                                                                           section=section,
                                                                           variables=variables,
                                                                           inflow_gas=mb_dict[stream][section-1])
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                stream_conds=stream_conds,
                                                                stream=stream,
                                                                section=section,
                                                                variables=variables,
                                                                mode=mode,
                                                                material_output=mb_dict[stream][section],
                                                                heat_output=hb_dict[stream][section-1],
                                                                vaccume_output=None)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                               stream_conds=stream_conds,
                                               section=1,
                                               variables=variables,
                                               heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                               heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                         stream_conds=stream_conds,
                                                         section=section,
                                                         variables=variables,
                                                         heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                         heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                        position=position,
                                                        variables=variables,
                                                        heat_output=hb_dict,
                                                        heat_wall_output=hb_wall)
    # 4. バッチ吸着後圧力変化
    total_press_after_batch_adsorp = base_models.total_press_after_batch_adsorp(sim_conds=sim_conds,
                                                                                variables=variables)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
        "total_press_after_batch_adsorp": total_press_after_batch_adsorp, # バッチ吸着後圧力
    }

    return output

def flow_adsorption_downstream(sim_conds, stream_conds, variables,
                               inflow_gas):
    """ 流通吸着（下流）
        説明: ２つの吸着塔による流通吸着における下流側
        ベースモデル: 流通吸着モデル
        終了条件: 上流塔の温度上昇、時間経過
        補足: 通常の流通吸着とほぼ同じだが、section1の流入ガスが上流の流出ガスになっている点のみ異なる

    Args:
        sim_conds (dict): 実験条件
        stream_conds (dict): 実験条件から定義された各stream条件
        df_obs (pd.DataFrame): 観測値
        variables (dict): 状態変数
        timestamp (float): 時刻

    Returns:
        output (dict): 各モデルの計算結果
    """
    mb_dict = {}
    hb_dict = {}
    hb_wall = {}
    hb_lid = {}
    mode = 0 # 吸着

    ### セル計算 --------------------------------------------------------------

    # 1. マテバラ・熱バラ計算
    for stream in range(1, 1+sim_conds["CELL_SPLIT"]["num_str"]):
        mb_dict[stream] = {}
        hb_dict[stream] = {}
        # sec_1は手動で実施
        # NOTE: # 流入ガスが上流の流出ガス
        mb_dict[stream][1] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                 stream_conds=stream_conds,
                                                                 stream=stream,
                                                                 section=1,
                                                                 variables=variables,
                                                                 inflow_gas=inflow_gas[stream][sim_conds["CELL_SPLIT"]["num_sec"]])
        hb_dict[stream][1] = base_models.heat_balance(sim_conds=sim_conds,
                                                           stream_conds=stream_conds,
                                                           stream=stream,
                                                           section=1,
                                                           variables=variables,
                                                           mode=mode,
                                                           material_output=mb_dict[stream][1],
                                                           heat_output=None,
                                                           vaccume_output=None)
        # sec_2以降は自動で実施
        for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
            mb_dict[stream][section] = base_models.material_balance_adsorp(sim_conds=sim_conds,
                                                                           stream_conds=stream_conds,
                                                                           stream=stream,
                                                                           section=section,
                                                                           variables=variables,
                                                                           inflow_gas=mb_dict[stream][section-1])
            hb_dict[stream][section] = base_models.heat_balance(sim_conds=sim_conds,
                                                                stream_conds=stream_conds,
                                                                stream=stream,
                                                                section=section,
                                                                variables=variables,
                                                                mode=mode,
                                                                material_output=mb_dict[stream][section],
                                                                heat_output=hb_dict[stream][section-1],
                                                                vaccume_output=None)
    # 2. 壁面熱バラ（stream = 1+sim_conds["CELL_SPLIT"]["num_str"]）
    hb_wall[1] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                               stream_conds=stream_conds,
                                               section=1,
                                               variables=variables,
                                               heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][1],
                                               heat_wall_output=None)
    for section in range(2, 1+sim_conds["CELL_SPLIT"]["num_sec"]):
        hb_wall[section] = base_models.heat_balance_wall(sim_conds=sim_conds,
                                                         stream_conds=stream_conds,
                                                         section=section,
                                                         variables=variables,
                                                         heat_output=hb_dict[sim_conds["CELL_SPLIT"]["num_str"]][section],
                                                         heat_wall_output=hb_wall[section-1])
    # 3. 上下蓋熱バラ
    for position in ["up", "down"]:
        hb_lid[position] = base_models.heat_balance_lid(sim_conds=sim_conds,
                                                        position=position,
                                                        variables=variables,
                                                        heat_output=hb_dict,
                                                        heat_wall_output=hb_wall)
    # 出力
    output = {
        "material": mb_dict, # マテバラ
        "heat": hb_dict, # 熱バラ
        "heat_wall": hb_wall, # 熱バラ（壁面）
        "heat_lid": hb_lid, # 熱バラ（蓋）
    }

    return output