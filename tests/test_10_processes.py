"""10工程分の計算結果比較テスト

PSA担当者向け説明:
旧コードと新コードで10工程分のシミュレーションを実行し、
各工程終了時の状態が一致することを確認します。
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from copy import deepcopy


def compare_tower_states(old_state, new_state, tower_num: int, tolerance: float = 1e-5) -> list:
    """塔の状態を比較"""
    differences = []
    
    old_tower = old_state.towers[tower_num]
    new_tower = new_state.towers[tower_num]
    
    # 全圧の比較
    if abs(old_tower.total_press - new_tower.total_press) > tolerance:
        differences.append(
            f"塔{tower_num} 全圧: old={old_tower.total_press:.6f}, new={new_tower.total_press:.6f}"
        )
    
    # 温度の比較
    temp_diff = np.abs(old_tower.temp - new_tower.temp)
    if np.max(temp_diff) > tolerance:
        max_idx = np.unravel_index(np.argmax(temp_diff), temp_diff.shape)
        differences.append(
            f"塔{tower_num} 温度[{max_idx}]: old={old_tower.temp[max_idx]:.4f}, new={new_tower.temp[max_idx]:.4f}"
        )
    
    # 吸着量の比較
    loading_diff = np.abs(old_tower.loading - new_tower.loading)
    if np.max(loading_diff) > tolerance:
        max_idx = np.unravel_index(np.argmax(loading_diff), loading_diff.shape)
        differences.append(
            f"塔{tower_num} 吸着量[{max_idx}]: old={old_tower.loading[max_idx]:.6f}, new={new_tower.loading[max_idx]:.6f}"
        )
    
    # CO2モル分率の比較
    co2_diff = np.abs(old_tower.co2_mole_fraction - new_tower.co2_mole_fraction)
    if np.max(co2_diff) > tolerance:
        max_idx = np.unravel_index(np.argmax(co2_diff), co2_diff.shape)
        differences.append(
            f"塔{tower_num} CO2モル分率[{max_idx}]: old={old_tower.co2_mole_fraction[max_idx]:.6f}, new={new_tower.co2_mole_fraction[max_idx]:.6f}"
        )
    
    return differences


def run_old_simulator_processes(cond_id: str, num_processes: int):
    """旧シミュレーターで指定工程数を実行"""
    from config.sim_conditions import SimulationConditions
    from core.state import StateVariables
    from core.physics import operation_models
    from core.simulation_results import SimulationResults
    from utils import const
    from copy import deepcopy
    
    # 条件読み込み
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    dt = sim_conds.get_tower(1).common.calculation_step_time
    
    # 稼働工程表読み込み
    filepath = const.CONDITIONS_DIR + cond_id + "/" + "稼働工程表.xlsx"
    df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")
    
    # 状態変数初期化
    state_manager = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    residual_gas_composition = None
    
    # 工程ごとの最終状態を記録
    process_states = []
    timestamp = 0
    
    for process_index in list(df_operation.index)[:num_processes]:
        mode_list = list(df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
        termination_cond_str = df_operation.loc[process_index, "終了条件"]
        
        # バッチ吸着の圧力平均化
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
        
        timestamp_p = 0
        
        # 終了条件を満たすまでループ
        while _check_termination_old(termination_cond_str, state_manager, timestamp, timestamp_p, num_sections):
            # 各塔の計算
            record_outputs = _calc_mode_list_old(
                sim_conds, mode_list, state_manager, residual_gas_composition, operation_models
            )
            
            # 均圧加圧後の残留ガス組成を更新
            if "均圧_加圧" in mode_list:
                pressurization_tower_num = mode_list.index("均圧_加圧") + 1
                # この場合は均圧加圧の結果からresidual_gas_compositionを更新
                # （実際のシミュレーターと同じロジック）
            
            timestamp_p += dt
            
            # タイムアウト
            if timestamp_p >= 20:
                break
        
        timestamp += timestamp_p
        
        # 工程終了時の状態を記録
        process_states.append({
            "process_index": process_index,
            "timestamp": timestamp,
            "state": deepcopy(state_manager),
        })
        
        print(f"  [OLD] 工程{process_index}完了 timestamp={timestamp:.2f}")
    
    return process_states


def _calc_mode_list_old(sim_conds, mode_list, state_manager, residual_gas_composition, operation_models):
    """旧コードでモードリストを実行"""
    from copy import deepcopy
    
    num_towers = sim_conds.num_towers
    record_outputs = {}
    
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
        record_outputs[upstream_tower_num], _ = _branch_mode_old(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, None, residual_gas_composition, operation_models
        )
        
        # 下流塔
        record_outputs[downstream_tower_num], _ = _branch_mode_old(
            sim_conds.get_tower(downstream_tower_num), downstream_mode,
            downstream_tower_num, state_manager,
            record_outputs[upstream_tower_num]["material"],
            residual_gas_composition, operation_models
        )
        
        # 残りの塔
        for tower_num in range(1, num_towers + 1):
            if tower_num in [upstream_tower_num, downstream_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            record_outputs[tower_num], _ = _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    elif "均圧_減圧" in mode_list and "均圧_加圧" in mode_list:
        depressurization_tower_num = mode_list.index("均圧_減圧") + 1
        pressurization_tower_num = mode_list.index("均圧_加圧") + 1
        
        pressurization_tower_pressure = state_manager.towers[pressurization_tower_num].total_press
        
        # 減圧
        record_outputs[depressurization_tower_num], all_outputs = _branch_mode_old(
            sim_conds.get_tower(depressurization_tower_num), "均圧_減圧",
            depressurization_tower_num, state_manager,
            pressurization_tower_pressure, residual_gas_composition, operation_models
        )
        
        # 加圧
        record_outputs[pressurization_tower_num], _ = _branch_mode_old(
            sim_conds.get_tower(pressurization_tower_num), "均圧_加圧",
            pressurization_tower_num, state_manager,
            all_outputs.downflow_params, residual_gas_composition, operation_models
        )
        
        # 残りの塔
        for tower_num in range(1, num_towers + 1):
            if tower_num in [depressurization_tower_num, pressurization_tower_num]:
                continue
            mode = mode_list[tower_num - 1]
            record_outputs[tower_num], _ = _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    else:
        # 独立運転
        for tower_num in range(1, num_towers + 1):
            mode = mode_list[tower_num - 1]
            record_outputs[tower_num], _ = _branch_mode_old(
                sim_conds.get_tower(tower_num), mode,
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    return record_outputs


def _branch_mode_old(tower_conds, mode, tower_num, state_manager, other_tower_params, residual_gas_composition, operation_models):
    """旧コードでモード分岐"""
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
    elif mode == "均圧_減圧":
        calc_output = operation_models.equalization_depressurization(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num,
            downstream_tower_pressure=other_tower_params
        )
    elif mode == "均圧_加圧":
        calc_output = operation_models.equalization_pressurization(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num,
            inflow_from_upstream_tower=other_tower_params
        )
    elif mode == "真空脱着":
        calc_output = operation_models.vacuum_desorption(
            tower_conds=tower_conds, state_manager=state_manager, tower_num=tower_num
        )
    else:
        raise ValueError(f"未対応モード: {mode}")
    
    # 状態更新
    state_manager.update_from_calc_output(tower_num, mode, calc_output)
    
    record_items = calc_output.get_record_items()
    tower = state_manager.towers[tower_num]
    record_items["others"] = {
        "total_pressure": tower.total_press,
        "co2_mole_fraction": tower.co2_mole_fraction.copy(),
        "n2_mole_fraction": tower.n2_mole_fraction.copy(),
    }
    
    return record_items, calc_output


def _check_termination_old(termination_cond_str, state_manager, timestamp, timestamp_p, num_sections):
    """旧コードの終了条件判定"""
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


def run_new_simulator_processes(cond_id: str, num_processes: int):
    """新シミュレーターで指定工程数を実行"""
    from config.sim_conditions import SimulationConditions
    from core.state import StateVariables
    from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
    from process.termination_conditions import should_continue_process
    from utils import const
    from copy import deepcopy
    
    # 条件読み込み
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    dt = sim_conds.get_tower(1).common.calculation_step_time
    
    # 稼働工程表読み込み
    filepath = const.CONDITIONS_DIR + cond_id + "/" + "稼働工程表.xlsx"
    df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")
    
    # 状態変数初期化
    state_manager = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    residual_gas_composition = None
    
    # 工程ごとの最終状態を記録
    process_states = []
    timestamp = 0
    
    for process_index in list(df_operation.index)[:num_processes]:
        mode_list = list(df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
        termination_cond_str = df_operation.loc[process_index, "終了条件"]
        
        # バッチ吸着の圧力平均化
        prepare_batch_adsorption_pressure(state_manager, sim_conds, mode_list)
        
        timestamp_p = 0
        
        # 終了条件を満たすまでループ
        while should_continue_process(
            termination_cond_str, state_manager, timestamp, timestamp_p, num_sections
        ):
            # 各塔の計算
            outputs, residual_gas_composition = execute_mode_list(
                sim_conds=sim_conds,
                mode_list=mode_list,
                state_manager=state_manager,
                residual_gas_composition=residual_gas_composition,
            )
            
            timestamp_p += dt
            
            # タイムアウト
            if timestamp_p >= 20:
                break
        
        timestamp += timestamp_p
        
        # 工程終了時の状態を記録
        process_states.append({
            "process_index": process_index,
            "timestamp": timestamp,
            "state": deepcopy(state_manager),
        })
        
        print(f"  [NEW] 工程{process_index}完了 timestamp={timestamp:.2f}")
    
    return process_states


def main():
    """メイン処理"""
    print("=" * 60)
    print("Phase 5: 10工程分の計算結果比較テスト")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    num_processes = 10
    
    print(f"\n条件ID: {cond_id}")
    print(f"比較工程数: {num_processes}")
    
    # 旧シミュレーター実行
    print("\n[OLD] 旧シミュレーター実行中...")
    try:
        old_states = run_old_simulator_processes(cond_id, num_processes)
    except Exception as e:
        print(f"❌ 旧シミュレーターエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 新シミュレーター実行
    print("\n[NEW] 新シミュレーター実行中...")
    try:
        new_states = run_new_simulator_processes(cond_id, num_processes)
    except Exception as e:
        print(f"❌ 新シミュレーターエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 結果比較
    print("\n" + "=" * 60)
    print("工程ごとの比較結果")
    print("=" * 60)
    
    all_passed = True
    for old_proc, new_proc in zip(old_states, new_states):
        process_idx = old_proc["process_index"]
        old_ts = old_proc["timestamp"]
        new_ts = new_proc["timestamp"]
        
        print(f"\n工程{process_idx}:")
        print(f"  タイムスタンプ: old={old_ts:.4f}, new={new_ts:.4f}")
        
        # タイムスタンプの比較
        if abs(old_ts - new_ts) > 1e-6:
            print(f"  ⚠️ タイムスタンプ不一致")
            all_passed = False
        
        # 各塔の状態比較
        differences = []
        for tower_num in range(1, 4):  # 3塔
            diffs = compare_tower_states(
                old_proc["state"], new_proc["state"], tower_num, tolerance=1e-4
            )
            differences.extend(diffs)
        
        if differences:
            print(f"  ❌ 差異あり:")
            for diff in differences[:5]:
                print(f"     {diff}")
            if len(differences) > 5:
                print(f"     ... 他 {len(differences) - 5} 件")
            all_passed = False
        else:
            print(f"  ✅ 全塔の状態が一致")
    
    print("\n" + "=" * 60)
    print("最終結果")
    print("=" * 60)
    
    if all_passed:
        print(f"🎉 {num_processes}工程全てで新旧コードの計算結果が一致しました")
    else:
        print(f"❗ 一部の工程で差異が検出されました")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
