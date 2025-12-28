#!/usr/bin/env python3
"""
インデックス修正のクイックテスト
1つのモードだけ実行して結果を確認
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


def run_single_mode_test():
    """単一モードを実行して結果を取得"""
    print("初期化中...")
    sim_conds = SimulationConditions('5_08_mod_logging2')
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    print(f"num_towers={num_towers}, num_streams={num_streams}, num_sections={num_sections}")
    
    state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # 停止モード - 最も単純
    mode_list = ['停止', '停止', '停止']
    print(f"モード実行: {mode_list}")
    outputs, _ = execute_mode_list(sim_conds, mode_list, state, None)
    
    # 結果を抽出
    results = {}
    for tower_num, output in outputs.items():
        tower_results = {
            'total_pressure': output.others['total_pressure'],
        }
        
        # 状態変数の最終値
        tower_state = state.towers[tower_num]
        tower_results['temp_shape'] = list(tower_state.temp.shape)
        tower_results['temp_0_0'] = float(tower_state.temp[0, 0])
        tower_results['temp_1_17'] = float(tower_state.temp[1, 17])  # 2ストリームなので1が最大
        tower_results['loading_0_0'] = float(tower_state.loading[0, 0])
        tower_results['lid_temperature'] = tower_state.lid_temperature
        
        results[tower_num] = tower_results
    
    return results


def main_save():
    """結果を保存"""
    print("=" * 60)
    print("クイックテスト - 修正前の結果を保存")
    print("=" * 60)
    
    results = run_single_mode_test()
    
    with open('tests/quick_results_before.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果: {json.dumps(results, indent=2)}")
    print("\n✅ 保存完了")
    return 0


def main_compare():
    """比較"""
    print("=" * 60)
    print("クイックテスト - 結果比較")
    print("=" * 60)
    
    results_after = run_single_mode_test()
    
    with open('tests/quick_results_before.json', 'r', encoding='utf-8') as f:
        results_before = json.load(f)
    
    # キーを整数に戻す
    results_before = {int(k): v for k, v in results_before.items()}
    
    # 比較
    all_match = True
    for tower_num in results_before.keys():
        print(f"\n塔{tower_num}:")
        for key in results_before[tower_num]:
            b = results_before[tower_num][key]
            a = results_after[tower_num][key]
            if isinstance(b, list):
                match = b == a
            elif isinstance(b, float):
                match = abs(b - a) < 1e-10
            else:
                match = b == a
            
            status = "✅" if match else "❌"
            print(f"  {key}: {status} (before={b}, after={a})")
            if not match:
                all_match = False
    
    if all_match:
        print("\n✅ 全て一致")
        return 0
    else:
        print("\n❌ 差異あり")
        return 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        exit(main_compare())
    else:
        exit(main_save())
