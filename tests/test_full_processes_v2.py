#!/usr/bin/env python3
"""
Phase 5: 全工程の計算結果比較テスト（v2）

旧コード（core/physics/operation_models.py）と新コード（operation_modes/）の
計算結果が一致することを確認する。

方針：
- 各工程の開始前に旧コードの状態を新コードにコピー
- 1ステップごとに新旧を比較
- 差異があれば報告
"""
import sys
import os
sys.path.insert(0, "/home/user/webapp")
os.chdir("/home/user/webapp")

# 物性テーブル高速化を最初にインポート
import utils.prop_table

import numpy as np
import pandas as pd
from copy import deepcopy

from config.sim_conditions import SimulationConditions
from core.state.state_variables import StateVariables
from core.physics import operation_models
from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
from process.termination_conditions import should_continue_process
from utils import const


def compare_tower_states(old_state, new_state, tower_num: int, tolerance: float = 1e-4) -> list:
    """塔の状態を比較"""
    differences = []
    old_tower = old_state.towers[tower_num]
    new_tower = new_state.towers[tower_num]
    
    # 全圧
    if abs(old_tower.total_press - new_tower.total_press) > tolerance:
        differences.append(
            f"塔{tower_num} 全圧: old={old_tower.total_press:.6f}, new={new_tower.total_press:.6f}"
        )
    
    # 温度
    temp_diff = np.abs(old_tower.temp - new_tower.temp)
    if np.max(temp_diff) > tolerance:
        max_idx = np.unravel_index(np.argmax(temp_diff), temp_diff.shape)
        differences.append(
            f"塔{tower_num} 温度[{max_idx}]: old={old_tower.temp[max_idx]:.4f}, new={new_tower.temp[max_idx]:.4f}"
        )
    
    # 吸着量
    loading_diff = np.abs(old_tower.loading - new_tower.loading)
    if np.max(loading_diff) > tolerance:
        max_idx = np.unravel_index(np.argmax(loading_diff), loading_diff.shape)
        differences.append(
            f"塔{tower_num} 吸着量[{max_idx}]: old={old_tower.loading[max_idx]:.6f}, new={new_tower.loading[max_idx]:.6f}"
        )
    
    # CO2モル分率
    co2_diff = np.abs(old_tower.co2_mole_fraction - new_tower.co2_mole_fraction)
    if np.max(co2_diff) > tolerance:
        max_idx = np.unravel_index(np.argmax(co2_diff), co2_diff.shape)
        differences.append(
            f"塔{tower_num} CO2モル分率[{max_idx}]: old={old_tower.co2_mole_fraction[max_idx]:.6f}, new={new_tower.co2_mole_fraction[max_idx]:.6f}"
        )
    
    return differences


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


def _calc_mode_list_old(sim_conds, mode_list, state_manager, residual_gas_composition, operation_models):
    """旧コードでモードリストを実行"""
    num_towers = sim_conds.num_towers
    
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
        
        # 上流塔を計算し、結果を保存
        upstream_calc = _branch_mode_old(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, None, residual_gas_composition, operation_models
        )
        
        # 下流塔に上流の結果を渡す
        _branch_mode_old(
            sim_conds.get_tower(downstream_tower_num), downstream_mode,
            downstream_tower_num, state_manager,
            upstream_calc.material, residual_gas_composition, operation_models
        )
        
        for tower_num in range(1, num_towers + 1):
            if tower_num in [upstream_tower_num, downstream_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    elif "均圧_減圧" in mode_list and "均圧_加圧" in mode_list:
        depressurization_tower_num = mode_list.index("均圧_減圧") + 1
        pressurization_tower_num = mode_list.index("均圧_加圧") + 1
        
        pressurization_tower_pressure = state_manager.towers[pressurization_tower_num].total_press
        
        calc_output = operation_models.equalization_depressurization(
            tower_conds=sim_conds.get_tower(depressurization_tower_num),
            state_manager=state_manager,
            tower_num=depressurization_tower_num,
            downstream_tower_pressure=pressurization_tower_pressure
        )
        state_manager.update_from_calc_output(depressurization_tower_num, "均圧_減圧", calc_output)
        
        calc_output2 = operation_models.equalization_pressurization(
            tower_conds=sim_conds.get_tower(pressurization_tower_num),
            state_manager=state_manager,
            tower_num=pressurization_tower_num,
            inflow_from_upstream_tower=calc_output.downflow_params
        )
        state_manager.update_from_calc_output(pressurization_tower_num, "均圧_加圧", calc_output2)
        
        for tower_num in range(1, num_towers + 1):
            if tower_num in [depressurization_tower_num, pressurization_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    else:
        for tower_num in range(1, num_towers + 1):
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )


def _branch_mode_old(tower_conds, mode, tower_num, state_manager, other_tower_params, residual_gas_composition, operation_models):
    """旧コードでモード分岐して状態更新"""
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


# _get_upstream_output は削除（_branch_mode_old が結果を返すように変更）


def run_full_comparison(cond_id: str):
    """全工程の比較テストを実行（v2: ステップごとに同期）"""
    
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    dt = sim_conds.get_tower(1).common.calculation_step_time
    
    filepath = const.CONDITIONS_DIR + cond_id + "/" + "稼働工程表.xlsx"
    df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")
    
    # 単一の状態変数で進行
    state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    residual = None
    
    timestamp = 0
    all_pass = True
    total_steps = 0
    
    print(f"\n条件ID: {cond_id}")
    print(f"総工程数: {len(df_operation)}")
    print("-" * 60)
    
    for process_index in df_operation.index:
        mode_list = list(df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
        termination_cond_str = df_operation.loc[process_index, "終了条件"]
        
        # バッチ吸着の圧力平均化
        _prepare_batch_pressure_old(state, sim_conds, mode_list)
        
        timestamp_p = 0
        step_count = 0
        process_diffs = 0
        
        while _check_termination(termination_cond_str, state, timestamp, timestamp_p, num_sections):
            # 状態を保存
            old_state = deepcopy(state)
            new_state = deepcopy(state)
            
            # 旧コードで1ステップ
            _calc_mode_list_old(sim_conds, mode_list, old_state, residual, operation_models)
            
            # 新コードで1ステップ
            outputs, _ = execute_mode_list(
                sim_conds=sim_conds,
                mode_list=mode_list,
                state_manager=new_state,
                residual_gas_composition=residual,
            )
            
            # ステップごとに比較
            step_differences = []
            for tower_num in range(1, num_towers + 1):
                diffs = compare_tower_states(old_state, new_state, tower_num)
                step_differences.extend(diffs)
            
            if step_differences:
                process_diffs += 1
                if process_diffs <= 2:  # 最初の2つの差異だけ詳細表示
                    print(f"  工程{process_index} ステップ{step_count + 1}: 差異検出")
                    for diff in step_differences[:3]:
                        print(f"    {diff}")
            
            # 旧コードの状態を採用して進行
            state = old_state
            
            step_count += 1
            total_steps += 1
            timestamp_p += dt
            
            if timestamp_p >= 20:
                break
        
        timestamp += timestamp_p
        
        if process_diffs == 0:
            print(f"工程{process_index}: ✅ 一致 ({step_count}ステップ)")
        else:
            print(f"工程{process_index}: ❌ 差異あり ({process_diffs}/{step_count}ステップで差異)")
            all_pass = False
    
    print("-" * 60)
    print(f"総ステップ数: {total_steps}")
    
    return all_pass


def main():
    """メイン処理"""
    print("=" * 60)
    print("Phase 5: 全工程の計算結果比較テスト (v2)")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    
    success = run_full_comparison(cond_id)
    
    if success:
        print("\n✅ 全工程で計算結果が一致しました")
        return 0
    else:
        print("\n❌ 一部の工程で差異が検出されました")
        return 1


if __name__ == "__main__":
    exit(main())
