#!/usr/bin/env python3
"""
インデックス修正前後の結果比較テスト

修正前のシミュレーション結果を保存し、修正後と比較するためのスクリプト
"""
import sys
import os
import json
sys.path.insert(0, "/home/user/webapp")
os.chdir("/home/user/webapp")

# 物性テーブル高速化
import utils.prop_table

import numpy as np

from config.sim_conditions import SimulationConditions
from state.state_variables import StateVariables
from process.process_executor import execute_mode_list


def run_all_mode_tests():
    """全モードを実行して結果を取得"""
    sim_conds = SimulationConditions('5_08_mod_logging2')
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    results = {}
    
    # テストするモードリスト
    mode_lists = [
        ['停止', '停止', '停止'],
        ['初回ガス導入', '停止', '停止'],
        ['流通吸着_単独/上流', '停止', '停止'],
        ['流通吸着_単独/上流', '流通吸着_下流', '停止'],
        ['バッチ吸着_上流', 'バッチ吸着_下流', '停止'],
        ['バッチ吸着_上流（圧調弁あり）', 'バッチ吸着_下流（圧調弁あり）', '停止'],
        ['真空脱着', '停止', '停止'],
        ['流通吸着_単独/上流', '均圧_加圧', '均圧_減圧'],
    ]
    
    for i, mode_list in enumerate(mode_lists):
        print(f"テスト {i+1}/{len(mode_lists)}: {mode_list}")
        state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
        # is_first_step=True で工程初回の初期化処理が実行される
        outputs, _ = execute_mode_list(sim_conds, mode_list, state, None, is_first_step=True)
        
        # 結果を抽出（数値のみ）
        mode_results = {}
        for tower_num, output in outputs.items():
            tower_results = {
                'total_pressure': output.others['total_pressure'],
                'cumulative_co2_recovered': output.others['cumulative_co2_recovered'],
                'cumulative_n2_recovered': output.others['cumulative_n2_recovered'],
            }
            
            # 状態変数の最終値
            tower_state = state.towers[tower_num]
            tower_results['temp'] = tower_state.temp.tolist()
            tower_results['loading'] = tower_state.loading.tolist()
            tower_results['co2_mole_fraction'] = tower_state.co2_mole_fraction.tolist()
            tower_results['n2_mole_fraction'] = tower_state.n2_mole_fraction.tolist()
            tower_results['temp_wall'] = tower_state.temp_wall.tolist()
            tower_results['top_temperature'] = tower_state.top_temperature
            tower_results['bottom_temperature'] = tower_state.bottom_temperature
            
            # 物質収支結果 - 代表セル (stream=1, section=1) と (stream=3, section=18)
            for stream, section in [(1, 1), (1, 18), (3, 1), (3, 18)]:
                try:
                    mb = output.material.get_result(stream, section)
                    key = f'mass_balance_{stream}_{section}'
                    tower_results[key] = {
                        'inlet_co2': mb.inlet_gas.co2_volume,
                        'inlet_n2': mb.inlet_gas.n2_volume,
                        'outlet_co2': mb.outlet_gas.co2_volume,
                        'outlet_n2': mb.outlet_gas.n2_volume,
                        'updated_loading': mb.adsorption_state.updated_loading,
                    }
                except Exception as e:
                    tower_results[f'mass_balance_{stream}_{section}'] = {'error': str(e)}
            
            # 熱収支結果 - 代表セル
            for stream, section in [(1, 1), (1, 18), (3, 1), (3, 18)]:
                try:
                    hb = output.heat.get_result(stream, section)
                    key = f'heat_balance_{stream}_{section}'
                    tower_results[key] = {
                        'bed_temperature': hb.cell_temperatures.bed_temperature,
                        'thermocouple_temperature': hb.cell_temperatures.thermocouple_temperature,
                        'wall_to_bed_htc': hb.heat_transfer_coefficients.wall_to_bed,
                        'bed_to_bed_htc': hb.heat_transfer_coefficients.bed_to_bed,
                    }
                except Exception as e:
                    tower_results[f'heat_balance_{stream}_{section}'] = {'error': str(e)}
            
            mode_results[str(tower_num)] = tower_results
        
        results[str(mode_list)] = mode_results
    
    return results


def save_results(results, filename):
    """結果をJSONファイルとして保存"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"結果を保存: {filename}")


def load_results(filename):
    """結果をJSONファイルから読み込み"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results(before, after, tolerance=1e-9):
    """2つの結果を比較"""
    differences = []
    
    for mode_list in before.keys():
        if mode_list not in after:
            differences.append(f"モード {mode_list} が修正後の結果に存在しない")
            continue
        
        for tower_num in before[mode_list].keys():
            if tower_num not in after[mode_list]:
                differences.append(f"塔{tower_num}が修正後の結果に存在しない ({mode_list})")
                continue
            
            before_tower = before[mode_list][tower_num]
            after_tower = after[mode_list][tower_num]
            
            for key in before_tower.keys():
                if key not in after_tower:
                    differences.append(f"キー {key} が修正後の結果に存在しない (塔{tower_num}, {mode_list})")
                    continue
                
                b_val = before_tower[key]
                a_val = after_tower[key]
                
                if isinstance(b_val, dict):
                    # ネストされた辞書の比較
                    for sub_key in b_val.keys():
                        if sub_key not in a_val:
                            differences.append(f"サブキー {sub_key} が存在しない ({key}, 塔{tower_num}, {mode_list})")
                        else:
                            b_sub = b_val[sub_key]
                            a_sub = a_val[sub_key]
                            if isinstance(b_sub, (int, float)) and isinstance(a_sub, (int, float)):
                                if abs(b_sub - a_sub) > tolerance:
                                    differences.append(
                                        f"値が異なる: {key}.{sub_key} (塔{tower_num}, {mode_list})\n"
                                        f"  修正前: {b_sub}\n  修正後: {a_sub}\n  差: {abs(b_sub - a_sub)}"
                                    )
                elif isinstance(b_val, list):
                    # リスト（配列）の比較
                    b_arr = np.array(b_val)
                    a_arr = np.array(a_val)
                    if b_arr.shape != a_arr.shape:
                        differences.append(f"配列形状が異なる: {key} (塔{tower_num}, {mode_list})\n  修正前: {b_arr.shape}\n  修正後: {a_arr.shape}")
                    else:
                        max_diff = np.max(np.abs(b_arr - a_arr))
                        if max_diff > tolerance:
                            differences.append(
                                f"配列値が異なる: {key} (塔{tower_num}, {mode_list})\n"
                                f"  最大差: {max_diff}"
                            )
                elif isinstance(b_val, (int, float)):
                    if abs(b_val - a_val) > tolerance:
                        differences.append(
                            f"値が異なる: {key} (塔{tower_num}, {mode_list})\n"
                            f"  修正前: {b_val}\n  修正後: {a_val}\n  差: {abs(b_val - a_val)}"
                        )
    
    return differences


def main_save_before():
    """修正前の結果を保存"""
    print("=" * 60)
    print("修正前の結果を保存")
    print("=" * 60)
    
    results = run_all_mode_tests()
    save_results(results, 'tests/results_before.json')
    
    print("\n✅ 修正前の結果を保存完了")
    return 0


def main_compare():
    """修正後の結果と比較"""
    print("=" * 60)
    print("修正前後の結果を比較")
    print("=" * 60)
    
    # 修正後の結果を取得
    results_after = run_all_mode_tests()
    save_results(results_after, 'tests/results_after.json')
    
    # 修正前の結果を読み込み
    results_before = load_results('tests/results_before.json')
    
    # 比較
    differences = compare_results(results_before, results_after)
    
    if not differences:
        print("\n✅ 全ての結果が一致しています！")
        return 0
    else:
        print(f"\n❌ {len(differences)} 件の差異が見つかりました:")
        for diff in differences:
            print(f"  - {diff}")
        return 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        exit(main_compare())
    else:
        exit(main_save_before())
