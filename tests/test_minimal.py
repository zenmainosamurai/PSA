#!/usr/bin/env python3
"""
最小限のテスト - CellAccessorの動作確認
"""
import sys
import os
import json
sys.path.insert(0, "/home/user/webapp")
os.chdir("/home/user/webapp")

import numpy as np
from state.state_variables import TowerStateArrays

def test_cell_accessor():
    """CellAccessorのテスト"""
    print("CellAccessorテスト...")
    
    # テスト用の配列を作成
    num_streams = 3
    num_sections = 20
    
    tower_state = TowerStateArrays(
        temp=np.arange(num_streams * num_sections, dtype=np.float64).reshape(num_streams, num_sections),
        thermocouple_temperature=np.zeros((num_streams, num_sections), dtype=np.float64),
        loading=np.zeros((num_streams, num_sections), dtype=np.float64),
        previous_loading=np.zeros((num_streams, num_sections), dtype=np.float64),
        co2_mole_fraction=np.zeros((num_streams, num_sections), dtype=np.float64),
        n2_mole_fraction=np.zeros((num_streams, num_sections), dtype=np.float64),
        wall_to_bed_heat_transfer_coef=np.zeros((num_streams, num_sections), dtype=np.float64),
        bed_heat_transfer_coef=np.zeros((num_streams, num_sections), dtype=np.float64),
        outlet_co2_partial_pressure=np.zeros((num_streams, num_sections), dtype=np.float64),
        temp_wall=np.zeros(num_sections, dtype=np.float64),
        top_temperature=25.0,
        bottom_temperature=25.0,
        total_press=0.1,
        cumulative_co2_recovered=0.0,
        cumulative_n2_recovered=0.0,
    )
    
    # temp配列の内容を確認
    # temp[stream][section] = stream * num_sections + section
    print(f"  temp[0, 0] = {tower_state.temp[0, 0]}")  # 0
    print(f"  temp[0, 1] = {tower_state.temp[0, 1]}")  # 1
    print(f"  temp[1, 0] = {tower_state.temp[1, 0]}")  # 20
    print(f"  temp[2, 19] = {tower_state.temp[2, 19]}")  # 59
    
    # 0オリジンのAPIでアクセス（内部インデックス）
    print("\n0オリジンAPIでアクセス:")
    cell_0_0 = tower_state.cell(0, 0)  # stream=0, section=0 -> 配列[0, 0]
    cell_0_1 = tower_state.cell(0, 1)  # stream=0, section=1 -> 配列[0, 1]
    cell_1_0 = tower_state.cell(1, 0)  # stream=1, section=0 -> 配列[1, 0]
    cell_2_19 = tower_state.cell(2, 19)  # stream=2, section=19 -> 配列[2, 19]
    
    print(f"  cell(0, 0).temp = {cell_0_0.temp} (期待値: 0)")
    print(f"  cell(0, 1).temp = {cell_0_1.temp} (期待値: 1)")
    print(f"  cell(1, 0).temp = {cell_1_0.temp} (期待値: 20)")
    print(f"  cell(2, 19).temp = {cell_2_19.temp} (期待値: 59)")
    
    # 検証
    results = {
        "cell_0_0_temp": cell_0_0.temp,
        "cell_0_1_temp": cell_0_1.temp,
        "cell_1_0_temp": cell_1_0.temp,
        "cell_2_19_temp": cell_2_19.temp,
    }
    
    expected = {
        "cell_0_0_temp": 0.0,
        "cell_0_1_temp": 1.0,
        "cell_1_0_temp": 20.0,
        "cell_2_19_temp": 59.0,
    }
    
    all_pass = True
    for key in expected:
        if abs(results[key] - expected[key]) > 1e-10:
            print(f"  ❌ {key}: 期待値={expected[key]}, 実値={results[key]}")
            all_pass = False
        else:
            print(f"  ✅ {key}: OK")
    
    return all_pass, results


def main_save():
    """結果を保存"""
    print("=" * 60)
    print("最小テスト - 修正前")
    print("=" * 60)
    
    all_pass, results = test_cell_accessor()
    
    with open('tests/minimal_results_before.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    if all_pass:
        print("\n✅ テスト成功")
        return 0
    else:
        print("\n❌ テスト失敗")
        return 1


def main_compare():
    """比較"""
    print("=" * 60)
    print("最小テスト - 比較")
    print("=" * 60)
    
    _, results_after = test_cell_accessor()
    
    with open('tests/minimal_results_before.json', 'r', encoding='utf-8') as f:
        results_before = json.load(f)
    
    all_match = True
    for key in results_before:
        b = results_before[key]
        a = results_after[key]
        if abs(b - a) < 1e-10:
            print(f"  ✅ {key}: {b} == {a}")
        else:
            print(f"  ❌ {key}: {b} != {a}")
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
