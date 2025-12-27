"""差異の発生源を調査

工程14終了時点での新旧状態の詳細差異を確認
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


def execute_step_old(sim_conds, mode_list, state_manager, residual_gas_composition):
    """旧コードで1ステップ実行"""
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
    
    updated_residual = residual_gas_composition
    
    if has_pair:
        upstream_tower_num = mode_list.index(upstream_mode) + 1
        downstream_tower_num = mode_list.index(downstream_mode) + 1
        
        upstream_output = _branch_mode_old(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, None, residual_gas_composition
        )
        
        _branch_mode_old(
            sim_conds.get_tower(downstream_tower_num), downstream_mode,
            downstream_tower_num, state_manager,
            upstream_output.material, residual_gas_composition
        )
        
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
        updated_residual = calc_output2.material
        
        for tower_num in range(1, num_towers + 1):
            if tower_num in [depressurization_tower_num, pressurization_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition
            )
    
    else:
        for tower_num in range(1, num_towers + 1):
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition
            )
    
    return updated_residual


def _branch_mode_old(tower_conds, mode, tower_num, state_manager, other_tower_params, residual_gas_composition):
    """旧コードでモード分岐"""
    
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


def prepare_batch_pressure(sim_conds, mode_list, state_manager):
    """バッチ吸着の圧力平均化"""
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


def run_debug():
    """差異発生源の調査"""
    cond_id = '5_08_mod_logging2'
    
    print("=" * 60, flush=True)
    print("差異発生源の調査", flush=True)
    print("=" * 60, flush=True)
    
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    tower_conds = sim_conds.get_tower(1)
    num_streams = tower_conds.common.num_streams
    num_sections = tower_conds.common.num_sections
    dt = tower_conds.common.calculation_step_time
    
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
    
    # 工程8まで旧コードで実行（両方同じ状態）
    print("\n工程8まで旧コードで実行...", flush=True)
    for process_idx in range(1, 9):
        row = operation_sheet.iloc[process_idx]
        mode_list = [str(row[i]) for i in range(1, 1 + num_towers)]
        termination_cond_str = str(row[4])
        
        prepare_batch_pressure(sim_conds, mode_list, old_state)
        prepare_batch_pressure(sim_conds, mode_list, new_state)
        
        timestamp_p = 0
        while check_termination(termination_cond_str, old_state, timestamp, timestamp_p, num_sections):
            old_residual = execute_step_old(sim_conds, mode_list, old_state, old_residual)
            new_residual = execute_step_old(sim_conds, mode_list, new_state, new_residual)
            timestamp_p += dt
            if timestamp_p >= 5:
                break
        
        timestamp += timestamp_p
        print(f"  工程{process_idx} 完了 (timestamp={timestamp:.2f})", flush=True)
    
    print("\n工程8終了時の状態（新旧同じはず）:", flush=True)
    for tower_num in range(1, num_towers + 1):
        old_t = old_state.towers[tower_num]
        new_t = new_state.towers[tower_num]
        temp_diff = np.abs(old_t.temp - new_t.temp).max()
        load_diff = np.abs(old_t.loading - new_t.loading).max()
        print(f"  塔{tower_num}: 全圧差={old_t.total_press - new_t.total_press:.2e}, 温度max差={temp_diff:.2e}, 吸着量max差={load_diff:.2e}", flush=True)
    
    # 工程9を新旧で分けて実行
    print("\n工程9を新旧コードで分けて実行...", flush=True)
    row = operation_sheet.iloc[9]
    mode_list = [str(row[i]) for i in range(1, 1 + num_towers)]
    termination_cond_str = str(row[4])
    print(f"  モード: {mode_list}", flush=True)
    
    prepare_batch_pressure(sim_conds, mode_list, old_state)
    prepare_batch_pressure(sim_conds, mode_list, new_state)
    
    timestamp_p = 0
    step = 0
    while check_termination(termination_cond_str, old_state, timestamp, timestamp_p, num_sections):
        # 旧コードで1ステップ
        old_residual = execute_step_old(sim_conds, mode_list, old_state, old_residual)
        # 新コードで1ステップ
        _, new_residual = execute_mode_list(sim_conds, mode_list, new_state, new_residual)
        
        timestamp_p += dt
        step += 1
        
        # ステップ10, 30, 50で差異確認
        if step in [10, 30, 50, 74]:
            print(f"\n  ステップ{step}時点での差異:", flush=True)
            for tower_num in range(1, num_towers + 1):
                old_t = old_state.towers[tower_num]
                new_t = new_state.towers[tower_num]
                temp_diff = np.abs(old_t.temp - new_t.temp).max()
                load_diff = np.abs(old_t.loading - new_t.loading).max()
                if temp_diff > 1e-6 or load_diff > 1e-6:
                    print(f"    塔{tower_num}: 全圧差={old_t.total_press - new_t.total_press:.2e}, 温度max差={temp_diff:.2e}, 吸着量max差={load_diff:.2e}", flush=True)
        
        if timestamp_p >= 5:
            break
    
    timestamp += timestamp_p
    print(f"\n工程9終了（{step}ステップ, timestamp={timestamp:.2f}）", flush=True)
    
    print("\n工程9終了時の詳細差異:", flush=True)
    for tower_num in range(1, num_towers + 1):
        old_t = old_state.towers[tower_num]
        new_t = new_state.towers[tower_num]
        
        temp_diff = np.abs(old_t.temp - new_t.temp)
        if temp_diff.max() > 1e-6:
            max_idx = np.unravel_index(np.argmax(temp_diff), temp_diff.shape)
            print(f"  塔{tower_num} 温度: max_diff={temp_diff.max():.6f} at {max_idx}", flush=True)
        
        load_diff = np.abs(old_t.loading - new_t.loading)
        if load_diff.max() > 1e-6:
            max_idx = np.unravel_index(np.argmax(load_diff), load_diff.shape)
            print(f"  塔{tower_num} 吸着量: max_diff={load_diff.max():.6f} at {max_idx}", flush=True)


if __name__ == "__main__":
    run_debug()
