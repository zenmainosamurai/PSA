#!/usr/bin/env python3
"""
工程9の最初のステップを詳細に比較するデバッグスクリプト
工程8まで旧コードで実行し、工程9の最初のステップだけを新旧両方で実行して比較
"""
import sys
import os
sys.path.insert(0, "/home/user/webapp")
os.chdir("/home/user/webapp")

import utils.prop_table  # 物性テーブル高速化

import numpy as np
import pandas as pd
from copy import deepcopy

from config.sim_conditions import SimulationConditions
from core.state.state_variables import StateVariables
from core.physics import operation_models
from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
from process.termination_conditions import should_continue_process
from utils import const

print("=" * 60)
print("工程9 最初のステップの詳細比較")
print("=" * 60)

cond_id = "5_08_mod_logging2"
sim_conds = SimulationConditions(cond_id)
num_towers = sim_conds.num_towers
num_streams = sim_conds.get_tower(1).common.num_streams
num_sections = sim_conds.get_tower(1).common.num_sections
dt = sim_conds.get_tower(1).common.calculation_step_time

# 稼働工程表読み込み
filepath = const.CONDITIONS_DIR + cond_id + "/" + "稼働工程表.xlsx"
df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")

# 状態変数初期化
old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)

old_residual = None
new_residual = None
timestamp = 0

# 工程1-8を両方のコードで実行（同一性を保つため）
print("\n工程1-8を実行中...")

def _check_termination(termination_cond_str, state_manager, timestamp, timestamp_p, num_sections):
    """終了条件判定"""
    parts = termination_cond_str.split("_")
    
    if parts[0] == "圧力到達":
        tower_num = int(parts[1][-1])
        target = float(parts[2])
        return state_manager.towers[tower_num].total_press < target
    elif parts[0] == "温度到達":
        tower_num = int(parts[1][-1])
        target = float(parts[2])
        temp_now = np.mean(state_manager.towers[tower_num].temp[:, num_sections - 1])
        return temp_now < target
    elif parts[0] == "時間経過":
        time = float(parts[1])
        unit = parts[2] if len(parts) > 2 else "min"
        if unit == "s":
            time /= 60
        return timestamp_p < time
    elif parts[0] == "時間到達":
        time = float(parts[1])
        return timestamp + timestamp_p < time
    return False


def _prepare_batch_pressure_old(state_manager, sim_conds, mode_list):
    """バッチ吸着圧力平均化（旧コード用）"""
    pairs = [
        ("バッチ吸着_上流", "バッチ吸着_下流"),
    ]
    for up, down in pairs:
        if up in mode_list and down in mode_list:
            upstream_num = mode_list.index(up) + 1
            downstream_num = mode_list.index(down) + 1
            us = state_manager.towers[upstream_num]
            ds = state_manager.towers[downstream_num]
            uv = sim_conds.get_tower(upstream_num).packed_bed.void_volume
            dv = sim_conds.get_tower(downstream_num).packed_bed.void_volume
            mean_p = (us.total_press * uv + ds.total_press * dv) / (uv + dv)
            us.total_press = mean_p
            ds.total_press = mean_p


def _branch_mode(tower_conds, mode, tower_num, state_manager, other_tower_params, residual_gas_composition, operation_models):
    """モード分岐"""
    from copy import deepcopy
    
    if mode == "初回ガス導入":
        tower_conds_copy = deepcopy(tower_conds)
        tower_conds_copy.feed_gas.co2_flow_rate = 20
        tower_conds_copy.feed_gas.n2_flow_rate = 25.2
        calc_output = operation_models.initial_adsorption(
            tower_conds=tower_conds_copy, state_manager=state_manager, tower_num=tower_num
        )
    elif mode == "停止":
        calc_output = operation_models.stop_mode(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    elif mode == "流通吸着_単独/上流":
        calc_output = operation_models.flow_adsorption_single_or_upstream(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    elif mode == "流通吸着_下流":
        calc_output = operation_models.flow_adsorption_downstream(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num,
            inflow_gas=other_tower_params
        )
    elif mode == "バッチ吸着_上流":
        calc_output = operation_models.batch_adsorption_upstream(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num,
            is_series_operation=True
        )
    elif mode == "バッチ吸着_下流":
        calc_output = operation_models.batch_adsorption_downstream(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num,
            is_series_operation=True, inflow_gas=other_tower_params,
            residual_gas_composition=residual_gas_composition
        )
    elif mode == "バッチ吸着_上流（圧調弁あり）":
        calc_output = operation_models.batch_adsorption_upstream_with_pressure_valve(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    elif mode == "バッチ吸着_下流（圧調弁あり）":
        calc_output = operation_models.batch_adsorption_downstream_with_pressure_valve(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num,
            is_series_operation=True, inflow_gas=other_tower_params,
            residual_gas_composition=residual_gas_composition
        )
    elif mode == "真空脱着":
        calc_output = operation_models.vacuum_desorption(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    else:
        raise ValueError(f"未対応モード: {mode}")
    
    state_manager.update_from_calc_output(tower_num, mode, calc_output)
    return calc_output


def _get_upstream_output(tower_conds, mode, tower_num, state_manager, operation_models):
    """上流塔の計算出力を取得（状態更新なし）"""
    if mode == "流通吸着_単独/上流":
        return operation_models.flow_adsorption_single_or_upstream(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    elif mode == "バッチ吸着_上流":
        return operation_models.batch_adsorption_upstream(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num,
            is_series_operation=True
        )
    elif mode == "バッチ吸着_上流（圧調弁あり）":
        return operation_models.batch_adsorption_upstream_with_pressure_valve(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    else:
        raise ValueError(f"未対応の上流モード: {mode}")


def _calc_mode_list(sim_conds, mode_list, state_manager, residual_gas_composition, operation_models):
    """モードリスト実行"""
    num_towers = sim_conds.num_towers
    
    # 上流・下流ペア
    up_down_pairs = [
        ("流通吸着_単独/上流", "流通吸着_下流"),
        ("バッチ吸着_上流", "バッチ吸着_下流"),
        ("バッチ吸着_上流（圧調弁あり）", "バッチ吸着_下流（圧調弁あり）"),
    ]
    
    has_pair = False
    upstream_mode = None
    downstream_mode = None
    for up, down in up_down_pairs:
        if up in mode_list and down in mode_list:
            has_pair = True
            upstream_mode = up
            downstream_mode = down
            break
    
    if has_pair:
        upstream_tower_num = mode_list.index(upstream_mode) + 1
        downstream_tower_num = mode_list.index(downstream_mode) + 1
        
        # 上流塔
        _branch_mode(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, None, residual_gas_composition, operation_models
        )
        
        # 上流の物質収支結果を再取得
        upstream_calc = _get_upstream_output(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, operation_models
        )
        
        # 下流塔
        _branch_mode(
            sim_conds.get_tower(downstream_tower_num), downstream_mode,
            downstream_tower_num, state_manager,
            upstream_calc.material, residual_gas_composition, operation_models
        )
        
        # 残りの塔
        for tower_num in range(1, num_towers + 1):
            if tower_num in [upstream_tower_num, downstream_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    elif "均圧_減圧" in mode_list and "均圧_加圧" in mode_list:
        depressurization_tower_num = mode_list.index("均圧_減圧") + 1
        pressurization_tower_num = mode_list.index("均圧_加圧") + 1
        
        pressurization_tower_pressure = state_manager.towers[pressurization_tower_num].total_press
        
        # 減圧
        calc_output = operation_models.equalization_depressurization(
            tower_conds=sim_conds.get_tower(depressurization_tower_num),
            state_manager=state_manager,
            tower_num=depressurization_tower_num,
            downstream_tower_pressure=pressurization_tower_pressure
        )
        state_manager.update_from_calc_output(depressurization_tower_num, "均圧_減圧", calc_output)
        
        # 加圧
        calc_output2 = operation_models.equalization_pressurization(
            tower_conds=sim_conds.get_tower(pressurization_tower_num),
            state_manager=state_manager,
            tower_num=pressurization_tower_num,
            inflow_from_upstream_tower=calc_output.downflow_params
        )
        state_manager.update_from_calc_output(pressurization_tower_num, "均圧_加圧", calc_output2)
        
        # 残りの塔
        for tower_num in range(1, num_towers + 1):
            if tower_num in [depressurization_tower_num, pressurization_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    else:
        # 独立運転
        for tower_num in range(1, num_towers + 1):
            mode = mode_list[tower_num - 1]
            _branch_mode(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )


# 工程1-8を実行（両方同じ状態に）
for process_index in range(1, 9):
    mode_list = list(df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
    termination_cond_str = df_operation.loc[process_index, "終了条件"]
    
    _prepare_batch_pressure_old(old_state, sim_conds, mode_list)
    _prepare_batch_pressure_old(new_state, sim_conds, mode_list)
    
    timestamp_p = 0
    while _check_termination(termination_cond_str, old_state, timestamp, timestamp_p, num_sections):
        _calc_mode_list(sim_conds, mode_list, old_state, old_residual, operation_models)
        _calc_mode_list(sim_conds, mode_list, new_state, new_residual, operation_models)
        timestamp_p += dt
        if timestamp_p >= 20:
            break
    
    timestamp += timestamp_p
    print(f"工程{process_index} 完了 (timestamp={timestamp:.2f})")

print("\n" + "=" * 60)
print("工程8終了後の状態（両方同一のはず）:")
for t in range(1, num_towers + 1):
    ot = old_state.towers[t]
    nt = new_state.towers[t]
    print(f"塔{t}: old_press={ot.total_press:.6f}, new_press={nt.total_press:.6f}, diff={ot.total_press - nt.total_press:.2e}")
    print(f"       old_temp[0,0]={ot.temp[0,0]:.4f}, new_temp[0,0]={nt.temp[0,0]:.4f}, diff={ot.temp[0,0] - nt.temp[0,0]:.2e}")

# 工程9の準備
print("\n" + "=" * 60)
print("工程9の最初のステップを比較")
print("=" * 60)

mode_list_9 = list(df_operation.loc[9, ["塔1", "塔2", "塔3"]])
termination_cond_str_9 = df_operation.loc[9, "終了条件"]
print(f"モード: {mode_list_9}")
print(f"終了条件: {termination_cond_str_9}")

# バッチ吸着圧力平均化
_prepare_batch_pressure_old(old_state, sim_conds, mode_list_9)
prepare_batch_adsorption_pressure(new_state, sim_conds, mode_list_9)

print("\n圧力平均化後の状態:")
for t in range(1, num_towers + 1):
    ot = old_state.towers[t]
    nt = new_state.towers[t]
    print(f"塔{t}: old_press={ot.total_press:.6f}, new_press={nt.total_press:.6f}, diff={ot.total_press - nt.total_press:.2e}")

# 旧コードで1ステップ（状態を保存してから）
old_state_before = deepcopy(old_state)
new_state_before = deepcopy(new_state)

print("\n--- 旧コードで1ステップ実行 ---")
_calc_mode_list(sim_conds, mode_list_9, old_state, old_residual, operation_models)

print("\n--- 新コードで1ステップ実行 ---")
outputs, new_residual = execute_mode_list(
    sim_conds=sim_conds,
    mode_list=mode_list_9,
    state_manager=new_state,
    residual_gas_composition=new_residual,
)

print("\n" + "=" * 60)
print("1ステップ後の比較")
print("=" * 60)

for t in range(1, num_towers + 1):
    mode = mode_list_9[t - 1]
    ot = old_state.towers[t]
    nt = new_state.towers[t]
    
    print(f"\n塔{t} ({mode}):")
    print(f"  全圧: old={ot.total_press:.6f}, new={nt.total_press:.6f}, diff={ot.total_press - nt.total_press:.2e}")
    
    temp_diff = ot.temp - nt.temp
    max_idx = np.unravel_index(np.argmax(np.abs(temp_diff)), temp_diff.shape)
    print(f"  温度[{max_idx}]: old={ot.temp[max_idx]:.4f}, new={nt.temp[max_idx]:.4f}, diff={temp_diff[max_idx]:.4f}")
    print(f"  温度[0,0]: old={ot.temp[0,0]:.4f}, new={nt.temp[0,0]:.4f}, diff={ot.temp[0,0] - nt.temp[0,0]:.4f}")
    
    loading_diff = ot.loading - nt.loading
    max_idx_l = np.unravel_index(np.argmax(np.abs(loading_diff)), loading_diff.shape)
    print(f"  吸着量[{max_idx_l}]: old={ot.loading[max_idx_l]:.6f}, new={nt.loading[max_idx_l]:.6f}, diff={loading_diff[max_idx_l]:.6f}")
    
    co2_diff = ot.co2_mole_fraction - nt.co2_mole_fraction
    max_idx_c = np.unravel_index(np.argmax(np.abs(co2_diff)), co2_diff.shape)
    print(f"  CO2モル分率[{max_idx_c}]: old={ot.co2_mole_fraction[max_idx_c]:.6f}, new={nt.co2_mole_fraction[max_idx_c]:.6f}, diff={co2_diff[max_idx_c]:.6f}")

print("\n" + "=" * 60)
print("テスト完了")
