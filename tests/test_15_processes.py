"""工程15までの計算結果比較テスト

旧コードと新コードで工程15まで実行し、各工程終了時の状態が一致することを確認します。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from copy import deepcopy

from config.sim_conditions import SimulationConditions
from core.state import StateVariables
import core.physics.operation_models as operation_models
from process.process_executor import execute_mode_list


def compare_states(old_state, new_state, num_towers, tolerance=5e-2):
    """状態を比較して差異を返す"""
    differences = []
    
    for tower_num in range(1, num_towers + 1):
        old_tower = old_state.towers[tower_num]
        new_tower = new_state.towers[tower_num]
        
        # 全圧
        if abs(old_tower.total_press - new_tower.total_press) > tolerance:
            differences.append(f"塔{tower_num} 全圧: old={old_tower.total_press:.6f}, new={new_tower.total_press:.6f}")
        
        # 温度
        temp_diff = np.abs(old_tower.temp - new_tower.temp)
        if np.max(temp_diff) > tolerance:
            max_idx = np.unravel_index(np.argmax(temp_diff), temp_diff.shape)
            differences.append(f"塔{tower_num} 温度[{max_idx}]: old={old_tower.temp[max_idx]:.4f}, new={new_tower.temp[max_idx]:.4f}, diff={temp_diff[max_idx]:.6f}")
        
        # 吸着量
        loading_diff = np.abs(old_tower.loading - new_tower.loading)
        if np.max(loading_diff) > tolerance:
            max_idx = np.unravel_index(np.argmax(loading_diff), loading_diff.shape)
            differences.append(f"塔{tower_num} 吸着量[{max_idx}]: old={old_tower.loading[max_idx]:.6f}, new={new_tower.loading[max_idx]:.6f}, diff={loading_diff[max_idx]:.6f}")
        
        # CO2モル分率
        co2_diff = np.abs(old_tower.co2_mole_fraction - new_tower.co2_mole_fraction)
        if np.max(co2_diff) > tolerance:
            max_idx = np.unravel_index(np.argmax(co2_diff), co2_diff.shape)
            differences.append(f"塔{tower_num} CO2モル分率[{max_idx}]: old={old_tower.co2_mole_fraction[max_idx]:.6f}, new={new_tower.co2_mole_fraction[max_idx]:.6f}, diff={co2_diff[max_idx]:.6f}")
    
    return differences


def execute_step_old(sim_conds, mode_list, state_manager, residual_gas_composition):
    """旧コードで1ステップ実行"""
    num_towers = sim_conds.num_towers
    
    # 上流・下流ペアの確認
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
    
    updated_residual = residual_gas_composition
    
    if has_pair:
        upstream_tower_num = mode_list.index(upstream_mode) + 1
        downstream_tower_num = mode_list.index(downstream_mode) + 1
        
        # 上流塔
        upstream_output = _branch_mode_old(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, None, residual_gas_composition
        )
        
        # 下流塔
        _branch_mode_old(
            sim_conds.get_tower(downstream_tower_num), downstream_mode,
            downstream_tower_num, state_manager,
            upstream_output.material, residual_gas_composition
        )
        
        # 残りの塔
        for tower_num in range(1, num_towers + 1):
            if tower_num in [upstream_tower_num, downstream_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition
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
        updated_residual = calc_output2.material
        
        # 残りの塔
        for tower_num in range(1, num_towers + 1):
            if tower_num in [depressurization_tower_num, pressurization_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition
            )
    
    else:
        # 独立運転
        for tower_num in range(1, num_towers + 1):
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition
            )
    
    return updated_residual


def _branch_mode_old(tower_conds, mode, tower_num, state_manager, other_tower_params, residual_gas_composition):
    """旧コードでモード分岐して状態更新"""
    
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


def check_termination(termination_cond_str, state_manager, timestamp, timestamp_p, num_sections):
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


def prepare_batch_pressure(sim_conds, mode_list, state_manager):
    """バッチ吸着の圧力平均化（圧調弁なしの場合のみ）"""
    if "バッチ吸着_上流" in mode_list and "バッチ吸着_下流" in mode_list:
        upstream_tower_num = mode_list.index("バッチ吸着_上流") + 1
        downstream_tower_num = mode_list.index("バッチ吸着_下流") + 1
        upstream_state = state_manager.towers[upstream_tower_num]
        downstream_state = state_manager.towers[downstream_tower_num]
        upstream_void = sim_conds.get_tower(upstream_tower_num).packed_bed.void_volume
        downstream_void = sim_conds.get_tower(downstream_tower_num).packed_bed.void_volume
        total_press_mean = (
            upstream_state.total_press * upstream_void +
            downstream_state.total_press * downstream_void
        ) / (upstream_void + downstream_void)
        upstream_state.total_press = total_press_mean
        downstream_state.total_press = total_press_mean


def run_test(num_processes=15, max_steps_per_process=500):
    """工程までの比較テスト"""
    import sys as _sys
    cond_id = '5_08_mod_logging2'
    
    print("=" * 60, flush=True)
    print(f"工程{num_processes}までの計算結果比較テスト", flush=True)
    print("=" * 60, flush=True)
    
    # 条件読み込み
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    tower_conds = sim_conds.get_tower(1)
    num_streams = tower_conds.common.num_streams
    num_sections = tower_conds.common.num_sections
    dt = tower_conds.common.calculation_step_time
    
    # 稼働工程表読み込み
    operation_sheet = pd.read_excel(
        Path(f'conditions/{cond_id}/稼働工程表.xlsx'),
        sheet_name=0,
        header=None
    )
    
    # 両方の状態を初期化
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    old_residual = None
    new_residual = None
    
    timestamp = 0
    all_passed = True
    
    for process_idx in range(1, num_processes + 1):
        row = operation_sheet.iloc[process_idx]
        mode_list = [str(row[i]) for i in range(1, 1 + num_towers)]
        termination_cond_str = str(row[4])
        
        # バッチ吸着の圧力平均化（両方の状態に適用）
        prepare_batch_pressure(sim_conds, mode_list, old_state)
        prepare_batch_pressure(sim_conds, mode_list, new_state)
        
        timestamp_p = 0
        step_count = 0
        
        # 終了条件を満たすまでループ
        while check_termination(termination_cond_str, old_state, timestamp, timestamp_p, num_sections):
            # 旧コードで1ステップ
            old_residual = execute_step_old(sim_conds, mode_list, old_state, old_residual)
            
            # 新コードで1ステップ
            _, new_residual = execute_mode_list(sim_conds, mode_list, new_state, new_residual)
            
            timestamp_p += dt
            step_count += 1
            
            # タイムアウト
            if step_count >= max_steps_per_process:
                print(f"  工程{process_idx}: ステップ上限到達（{step_count}ステップ）")
                break
        
        timestamp += timestamp_p
        
        # 工程終了時の状態比較
        differences = compare_states(old_state, new_state, num_towers)
        
        if differences:
            print(f"工程{process_idx}: ❌ 差異あり（{step_count}ステップ, timestamp={timestamp:.2f}）", flush=True)
            for diff in differences[:5]:
                print(f"  {diff}", flush=True)
            if len(differences) > 5:
                print(f"  ...他 {len(differences) - 5} 件", flush=True)
            all_passed = False
        else:
            print(f"工程{process_idx}: ✅ 一致（{step_count}ステップ, timestamp={timestamp:.2f}）", flush=True)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 全工程で新旧コードの計算結果が一致しました")
    else:
        print("❌ 一部の工程で差異が検出されました")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    num_processes = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    run_test(num_processes, max_steps)
