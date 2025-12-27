"""工程9の詳細デバッグ

工程8終了後の状態から工程9のみを実行して差異を調査
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
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    print("=" * 60)
    print("工程7まで旧コードで実行（両方同じ）")
    print("=" * 60)
    
    # 工程7まで実行（両方同じ）
    for process_idx in range(1, 8):  # 工程1-7
        row = operation_sheet.iloc[process_idx]
        mode_list = [str(row[i]) for i in range(1, 1 + num_towers)]
        
        # 旧コードで実行
        _execute_process_old(sim_conds, mode_list, old_state, operation_models)
        _execute_process_old(sim_conds, mode_list, new_state, operation_models)
        
        print(f"工程{process_idx} 完了: モード={mode_list}")
    
    print("\n" + "=" * 60)
    print("工程7終了時の状態確認")
    print("=" * 60)
    for tower_num in range(1, num_towers + 1):
        old_tower = old_state.towers[tower_num]
        new_tower = new_state.towers[tower_num]
        print(f"\n塔{tower_num}:")
        print(f"  全圧: old={old_tower.total_press:.6f}, new={new_tower.total_press:.6f}")
    
    print("\n" + "=" * 60)
    print("工程8を旧コードと新コードでそれぞれ実行")
    print("=" * 60)
    
    # 工程8のモードを取得
    row = operation_sheet.iloc[8]
    mode_list_8 = [str(row[i]) for i in range(1, 1 + num_towers)]
    print(f"工程8 モード: {mode_list_8}")
    
    # old_stateは旧コードで工程8を実行
    _execute_process_old(sim_conds, mode_list_8, old_state, operation_models)
    
    # new_stateは新コードで工程8を実行
    from process.process_executor import execute_mode_list
    outputs8, _ = execute_mode_list(sim_conds, mode_list_8, new_state, None)
    
    print("工程8 完了")
    
    print("\n" + "=" * 60)
    print("工程8終了時の状態確認（両方同じはず）")
    print("=" * 60)
    for tower_num in range(1, num_towers + 1):
        old_tower = old_state.towers[tower_num]
        new_tower = new_state.towers[tower_num]
        print(f"\n塔{tower_num}:")
        print(f"  全圧: old={old_tower.total_press:.6f}, new={new_tower.total_press:.6f}")
        print(f"  温度[0,0]: old={old_tower.temp[0,0]:.4f}, new={new_tower.temp[0,0]:.4f}")
        print(f"  吸着量[1,0]: old={old_tower.loading[1,0]:.6f}, new={new_tower.loading[1,0]:.6f}")
    
    # 工程9のモードを取得
    row = operation_sheet.iloc[9]
    mode_list = [str(row[i]) for i in range(1, 1 + num_towers)]
    print(f"\n工程9 モード: {mode_list}")
    
    print("\n" + "=" * 60)
    print("工程9を旧コードで実行")
    print("=" * 60)
    _execute_process_old(sim_conds, mode_list, old_state, operation_models)
    
    print("\n" + "=" * 60)
    print("工程9を新コードで実行")
    print("=" * 60)
    # 新コードで工程9を実行
    outputs, _ = execute_mode_list(sim_conds, mode_list, new_state, None)
    
    print("\n" + "=" * 60)
    print("工程9終了後の状態比較")
    print("=" * 60)
    
    for tower_num in range(1, num_towers + 1):
        old_tower = old_state.towers[tower_num]
        new_tower = new_state.towers[tower_num]
        
        print(f"\n塔{tower_num}:")
        print(f"  全圧: old={old_tower.total_press:.6f}, new={new_tower.total_press:.6f}, diff={old_tower.total_press - new_tower.total_press:.6e}")
        
        # 温度の最大差
        temp_diff = np.abs(old_tower.temp - new_tower.temp)
        max_idx = np.unravel_index(np.argmax(temp_diff), temp_diff.shape)
        print(f"  温度最大差: idx={max_idx}, old={old_tower.temp[max_idx]:.4f}, new={new_tower.temp[max_idx]:.4f}, diff={temp_diff[max_idx]:.4f}")
        
        # 吸着量の最大差
        loading_diff = np.abs(old_tower.loading - new_tower.loading)
        max_idx = np.unravel_index(np.argmax(loading_diff), loading_diff.shape)
        print(f"  吸着量最大差: idx={max_idx}, old={old_tower.loading[max_idx]:.6f}, new={new_tower.loading[max_idx]:.6f}, diff={loading_diff[max_idx]:.6f}")
        
        # CO2モル分率の最大差
        co2_diff = np.abs(old_tower.co2_mole_fraction - new_tower.co2_mole_fraction)
        max_idx = np.unravel_index(np.argmax(co2_diff), co2_diff.shape)
        print(f"  CO2モル分率最大差: idx={max_idx}, old={old_tower.co2_mole_fraction[max_idx]:.6f}, new={new_tower.co2_mole_fraction[max_idx]:.6f}, diff={co2_diff[max_idx]:.6f}")


def _execute_process_old(sim_conds, mode_list, state_manager, operation_models):
    """旧コードでの1工程実行（test_10_processes.pyから抜粋）"""
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
    
    if has_pair:
        upstream_tower_num = mode_list.index(upstream_mode) + 1
        downstream_tower_num = mode_list.index(downstream_mode) + 1
        
        # 上流塔
        _branch_mode_old(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, None, None, operation_models
        )
        
        # 上流の物質収支結果を取得
        upstream_calc, _ = _branch_mode_old_get_output(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, None, None, operation_models
        )
        
        # 下流塔
        _branch_mode_old(
            sim_conds.get_tower(downstream_tower_num), downstream_mode,
            downstream_tower_num, state_manager,
            upstream_calc.material, None, operation_models
        )
        
        # 残りの塔
        for tower_num in range(1, num_towers + 1):
            if tower_num in [upstream_tower_num, downstream_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, None, operation_models
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
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, None, operation_models
            )
    
    else:
        # 独立運転
        for tower_num in range(1, num_towers + 1):
            mode = mode_list[tower_num - 1]
            _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, None, operation_models
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
    
    # 状態更新
    state_manager.update_from_calc_output(tower_num, mode, calc_output)


def _branch_mode_old_get_output(tower_conds, mode, tower_num, state_manager, other_tower_params, residual_gas_composition, operation_models):
    """旧コードでモード分岐（状態更新なし、結果のみ返す）"""
    from copy import deepcopy
    
    if mode == "初回ガス導入":
        tower_conds_copy = deepcopy(tower_conds)
        tower_conds_copy.feed_gas.co2_flow_rate = 20
        tower_conds_copy.feed_gas.n2_flow_rate = 25.2
        calc_output = operation_models.initial_adsorption(
            tower_conds=tower_conds_copy, state_manager=state_manager, tower_num=tower_num
        )
    elif mode == "バッチ吸着_上流（圧調弁あり）":
        calc_output = operation_models.batch_adsorption_upstream_with_pressure_valve(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    else:
        raise ValueError(f"未対応モード: {mode}")
    
    return calc_output, None


if __name__ == "__main__":
    run_debug()
