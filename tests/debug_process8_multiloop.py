"""工程8の複数ステップデバッグ

工程7終了後の状態から工程8を25ステップ実行して差異を調査
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from copy import deepcopy

from config.sim_conditions import SimulationConditions


def run_debug():
    """デバッグ実行"""
    cond_id = '5_08_mod_logging2'
    sim_conds = SimulationConditions(cond_id)
    
    # 稼働工程表を読み込み
    from pathlib import Path
    operation_sheet = pd.read_excel(
        Path(f'conditions/{cond_id}/稼働工程表.xlsx'),
        sheet_name=0,
        header=None
    )
    
    # 旧コードをインポート
    import core.physics.operation_models as operation_models
    from core.state import StateVariables
    
    # 新コードをインポート
    from process.process_executor import execute_mode_list
    from operation_modes.mode_types import OperationMode
    
    # 両方の状態を初期化
    num_towers = sim_conds.num_towers
    tower_conds = sim_conds.get_tower(1)
    num_streams = tower_conds.common.num_streams
    num_sections = tower_conds.common.num_sections
    dt = tower_conds.common.calculation_step_time
    
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    print("=" * 60)
    print("工程7まで両方のコードで実行（旧コード使用）")
    print("=" * 60)
    
    # 工程7まで実行
    for process_idx in range(1, 8):
        row = operation_sheet.iloc[process_idx]
        mode_list = [str(row[i]) for i in range(1, 1 + num_towers)]
        
        _execute_process_old_single(sim_conds, mode_list, old_state, operation_models)
        _execute_process_old_single(sim_conds, mode_list, new_state, operation_models)
        
        print(f"工程{process_idx} 完了")
    
    print("\n" + "=" * 60)
    print("工程7終了時の状態確認")
    print("=" * 60)
    
    _print_state_comparison(old_state, new_state, num_towers)
    
    # 工程8のモードを取得
    row = operation_sheet.iloc[8]
    mode_list_8 = [str(row[i]) for i in range(1, 1 + num_towers)]
    print(f"\n工程8 モード: {mode_list_8}")
    print(f"終了条件: 時間経過_15_s (= 25ステップ)")
    
    print("\n" + "=" * 60)
    print("工程8を25ステップ実行して比較")
    print("=" * 60)
    
    # 25ステップ実行
    for step in range(25):
        # 旧コードで1ステップ
        _calc_mode_list_old_step(sim_conds, mode_list_8, old_state, operation_models)
        
        # 新コードで1ステップ
        outputs, _ = execute_mode_list(sim_conds, mode_list_8, new_state, None)
        
        # 各ステップで比較
        differences = _compare_states(old_state, new_state, num_towers)
        if differences:
            print(f"\nステップ{step+1}で差異検出:")
            for diff in differences[:3]:
                print(f"  {diff}")
            if len(differences) > 3:
                print(f"  ...他 {len(differences) - 3} 件")
        else:
            print(f"ステップ{step+1}: 一致")
    
    print("\n" + "=" * 60)
    print("工程8終了後の最終状態比較")
    print("=" * 60)
    
    _print_state_comparison(old_state, new_state, num_towers)


def _execute_process_old_single(sim_conds, mode_list, state_manager, operation_models):
    """旧コードで1ステップ実行"""
    num_towers = sim_conds.num_towers
    
    # 均圧の場合
    if "均圧_減圧" in mode_list and "均圧_加圧" in mode_list:
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
            _branch_mode_old(sim_conds.get_tower(tower_num), mode, tower_num, state_manager, None, None, operation_models)
    
    else:
        # その他のモード
        for tower_num in range(1, num_towers + 1):
            mode = mode_list[tower_num - 1]
            _branch_mode_old(sim_conds.get_tower(tower_num), mode, tower_num, state_manager, None, None, operation_models)


def _calc_mode_list_old_step(sim_conds, mode_list, state_manager, operation_models):
    """旧コードで1ステップのモードリスト実行"""
    _execute_process_old_single(sim_conds, mode_list, state_manager, operation_models)


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
    elif mode == "真空脱着":
        calc_output = operation_models.vacuum_desorption(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    else:
        raise ValueError(f"未対応モード: {mode}")
    
    state_manager.update_from_calc_output(tower_num, mode, calc_output)


def _compare_states(old_state, new_state, num_towers, tolerance=1e-5):
    """状態を比較"""
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
            differences.append(f"塔{tower_num} 温度[{max_idx}]: old={old_tower.temp[max_idx]:.4f}, new={new_tower.temp[max_idx]:.4f}")
        
        # 吸着量
        loading_diff = np.abs(old_tower.loading - new_tower.loading)
        if np.max(loading_diff) > tolerance:
            max_idx = np.unravel_index(np.argmax(loading_diff), loading_diff.shape)
            differences.append(f"塔{tower_num} 吸着量[{max_idx}]: old={old_tower.loading[max_idx]:.6f}, new={new_tower.loading[max_idx]:.6f}")
    
    return differences


def _print_state_comparison(old_state, new_state, num_towers):
    """状態比較を印刷"""
    for tower_num in range(1, num_towers + 1):
        old_tower = old_state.towers[tower_num]
        new_tower = new_state.towers[tower_num]
        
        print(f"\n塔{tower_num}:")
        print(f"  全圧: old={old_tower.total_press:.6f}, new={new_tower.total_press:.6f}, diff={old_tower.total_press - new_tower.total_press:.6e}")
        
        temp_diff = np.abs(old_tower.temp - new_tower.temp)
        max_idx = np.unravel_index(np.argmax(temp_diff), temp_diff.shape)
        print(f"  温度最大差: idx={max_idx}, old={old_tower.temp[max_idx]:.4f}, new={new_tower.temp[max_idx]:.4f}, diff={temp_diff[max_idx]:.4f}")
        
        loading_diff = np.abs(old_tower.loading - new_tower.loading)
        max_idx = np.unravel_index(np.argmax(loading_diff), loading_diff.shape)
        print(f"  吸着量最大差: idx={max_idx}, old={old_tower.loading[max_idx]:.6f}, new={new_tower.loading[max_idx]:.6f}, diff={loading_diff[max_idx]:.6f}")


if __name__ == "__main__":
    run_debug()
