"""全工程の計算結果比較テスト

PSA担当者向け説明:
旧コードと新コードで全工程のシミュレーションを実行し、
各工程終了時の状態が一致することを確認します。

CoolPropの物性テーブルを活用して高速化します。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 物性テーブルをインポート（CoolPropをモンキーパッチ）
import utils.prop_table

import numpy as np
import pandas as pd
from copy import deepcopy

from config.sim_conditions import SimulationConditions
from core.state import StateVariables
from utils import const


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


def run_full_comparison(cond_id: str):
    """全工程の比較テストを実行"""
    from core.physics import operation_models
    from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
    from process.termination_conditions import should_continue_process
    
    # 条件読み込み
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    dt = sim_conds.get_tower(1).common.calculation_step_time
    
    # 稼働工程表読み込み
    filepath = const.CONDITIONS_DIR + cond_id + "/" + "稼働工程表.xlsx"
    df_operation = pd.read_excel(filepath, index_col="工程", sheet_name="工程")
    
    # 状態変数初期化（両方同じ初期状態から開始）
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    old_residual = None
    new_residual = None
    
    timestamp = 0
    all_pass = True
    
    print(f"\n条件ID: {cond_id}")
    print(f"総工程数: {len(df_operation)}")
    print("-" * 60)
    
    for process_index in df_operation.index:
        mode_list = list(df_operation.loc[process_index, ["塔1", "塔2", "塔3"]])
        termination_cond_str = df_operation.loc[process_index, "終了条件"]
        
        # バッチ吸着の圧力平均化（両方に適用）
        _prepare_batch_pressure_old(old_state, sim_conds, mode_list)
        prepare_batch_adsorption_pressure(new_state, sim_conds, mode_list)
        
        # 工程9開始前の状態を確認
        if process_index == 9:
            print(f"\n工程9開始前の状態:")
            for t in range(1, num_towers + 1):
                ot = old_state.towers[t]
                nt = new_state.towers[t]
                print(f"  塔{t}: old_press={ot.total_press:.6f}, new_press={nt.total_press:.6f}")
                print(f"       old_temp[0,0]={ot.temp[0,0]:.4f}, new_temp[0,0]={nt.temp[0,0]:.4f}")
        
        timestamp_p = 0
        
        # 終了条件を満たすまでループ
        step_count = 0
        while _check_termination(termination_cond_str, old_state, timestamp, timestamp_p, num_sections):
            # 旧コードで1ステップ
            _calc_mode_list_old(sim_conds, mode_list, old_state, old_residual, operation_models)
            
            # 新コードで1ステップ
            outputs, new_residual = execute_mode_list(
                sim_conds=sim_conds,
                mode_list=mode_list,
                state_manager=new_state,
                residual_gas_composition=new_residual,
            )
            
            step_count += 1
            
            # 工程9の最初の数ステップをデバッグ
            if process_index == 9 and step_count <= 3:
                print(f"  工程9 ステップ{step_count}:")
                print(f"    塔2圧力: old={old_state.towers[2].total_press:.6f}, new={new_state.towers[2].total_press:.6f}")
                print(f"    塔2温度[0,0]: old={old_state.towers[2].temp[0,0]:.4f}, new={new_state.towers[2].temp[0,0]:.4f}")
            
            timestamp_p += dt
            
            # タイムアウト
            if timestamp_p >= 20:
                break
        
        timestamp += timestamp_p
        
        # 工程終了時の状態を比較
        differences = []
        for tower_num in range(1, num_towers + 1):
            diffs = compare_tower_states(old_state, new_state, tower_num)
            differences.extend(diffs)
        
        if differences:
            print(f"工程{process_index}: ❌ 差異あり (timestamp={timestamp:.2f})")
            for diff in differences[:3]:
                print(f"  {diff}")
            if len(differences) > 3:
                print(f"  ...他 {len(differences) - 3} 件")
            all_pass = False
            # 最初の差異で詳細を出力して停止
            if process_index == 9:
                print("\n詳細デバッグ（工程9開始前の状態）:")
                print("  これは工程8終了後の状態です")
                break
        else:
            print(f"工程{process_index}: ✅ 一致 (timestamp={timestamp:.2f})")
    
    print("-" * 60)
    if all_pass:
        print("🎉 全工程で新旧コードの計算結果が一致しました")
    else:
        print("❗ 一部の工程で差異が検出されました")
    
    return all_pass


def _prepare_batch_pressure_old(state_manager, sim_conds, mode_list):
    """旧コード用の圧力平均化"""
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


def _calc_mode_list_old(sim_conds, mode_list, state_manager, residual_gas_composition, operation_models):
    """旧コードでモードリストを実行"""
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
            upstream_tower_num, state_manager, None, residual_gas_composition, operation_models
        )
        
        # 上流の物質収支結果を取得（再計算）
        upstream_calc = _get_upstream_output(
            sim_conds.get_tower(upstream_tower_num), upstream_mode,
            upstream_tower_num, state_manager, operation_models
        )
        
        # 下流塔
        _branch_mode_old(
            sim_conds.get_tower(downstream_tower_num), downstream_mode,
            downstream_tower_num, state_manager,
            upstream_calc.material, residual_gas_composition, operation_models
        )
        
        # 残りの塔
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
                tower_num, state_manager, None, residual_gas_composition, operation_models
            )
    
    else:
        # 独立運転
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
    
    # 状態更新
    state_manager.update_from_calc_output(tower_num, mode, calc_output)


def _get_upstream_output(tower_conds, mode, tower_num, state_manager, operation_models):
    """上流塔の計算出力を取得（状態更新なし）"""
    from copy import deepcopy
    
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


def main():
    """メイン処理"""
    print("=" * 60)
    print("Phase 5: 全工程の計算結果比較テスト")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    
    success = run_full_comparison(cond_id)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
