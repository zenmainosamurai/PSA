"""計算結果の詳細比較テスト

PSA担当者向け説明:
旧コードと新コードで計算結果が数値的に一致することを確認します。
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def compare_results(old_result, new_result, tolerance=1e-5):
    """結果の数値比較"""
    differences = []
    
    # 物質収支結果の比較
    try:
        for stream in range(1, 3):  # 2ストリーム
            for section in range(1, 21):  # 20セクション
                old_mat = old_result.material.get_result(stream, section)
                new_mat = new_result.material.get_result(stream, section)
                
                # 吸着量の比較
                old_loading = old_mat.adsorption_state.updated_loading
                new_loading = new_mat.adsorption_state.updated_loading
                if abs(old_loading - new_loading) > tolerance:
                    differences.append(f"吸着量[{stream},{section}]: old={old_loading:.6f}, new={new_loading:.6f}")
                
                # 出口ガス量の比較
                old_co2 = old_mat.outlet_gas.co2_volume
                new_co2 = new_mat.outlet_gas.co2_volume
                if abs(old_co2 - new_co2) > tolerance:
                    differences.append(f"出口CO2[{stream},{section}]: old={old_co2:.6f}, new={new_co2:.6f}")
    except Exception as e:
        differences.append(f"物質収支比較エラー: {e}")
    
    # 熱収支結果の比較
    try:
        for stream in range(1, 3):
            for section in range(1, 21):
                old_heat = old_result.heat.get_result(stream, section)
                new_heat = new_result.heat.get_result(stream, section)
                
                old_temp = old_heat.cell_temperatures.bed_temperature
                new_temp = new_heat.cell_temperatures.bed_temperature
                if abs(old_temp - new_temp) > tolerance:
                    differences.append(f"層温度[{stream},{section}]: old={old_temp:.6f}, new={new_temp:.6f}")
    except Exception as e:
        differences.append(f"熱収支比較エラー: {e}")
    
    return differences


def test_stop_mode_comparison():
    """停止モードの詳細比較"""
    print("=" * 60)
    print("テスト: 停止モード計算結果の詳細比較")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    
    from config.sim_conditions import SimulationConditions
    from core.state import StateVariables
    from core.physics import operation_models
    from operation_modes import execute_stop_mode
    
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    tower_conds = sim_conds.get_tower(1)
    
    # 同じ初期状態で計算
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # 旧コード
    old_result = operation_models.stop_mode(
        tower_conds=tower_conds,
        state_manager=old_state,
        tower_num=1
    )
    
    # 新コード
    new_result = execute_stop_mode(
        tower_conds=tower_conds,
        state_manager=new_state,
        tower_num=1
    )
    
    differences = compare_results(old_result, new_result)
    
    if not differences:
        print("✅ 停止モード: 全ての計算結果が一致")
        return True
    else:
        print("❌ 停止モード: 差異あり")
        for diff in differences[:10]:  # 最初の10件のみ表示
            print(f"  {diff}")
        if len(differences) > 10:
            print(f"  ... 他 {len(differences) - 10} 件")
        return False


def test_flow_adsorption_comparison():
    """流通吸着モードの詳細比較"""
    print("\n" + "=" * 60)
    print("テスト: 流通吸着モード計算結果の詳細比較")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    
    from config.sim_conditions import SimulationConditions
    from core.state import StateVariables
    from core.physics import operation_models
    from operation_modes import execute_flow_adsorption_upstream
    
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    tower_conds = sim_conds.get_tower(1)
    
    # 同じ初期状態で計算
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # 旧コード
    old_result = operation_models.flow_adsorption_single_or_upstream(
        tower_conds=tower_conds,
        state_manager=old_state,
        tower_num=1
    )
    
    # 新コード
    new_result = execute_flow_adsorption_upstream(
        tower_conds=tower_conds,
        state_manager=new_state,
        tower_num=1
    )
    
    differences = compare_results(old_result, new_result)
    
    if not differences:
        print("✅ 流通吸着モード: 全ての計算結果が一致")
        return True
    else:
        print("❌ 流通吸着モード: 差異あり")
        for diff in differences[:10]:
            print(f"  {diff}")
        if len(differences) > 10:
            print(f"  ... 他 {len(differences) - 10} 件")
        return False


def test_vacuum_desorption_comparison():
    """真空脱着モードの詳細比較"""
    print("\n" + "=" * 60)
    print("テスト: 真空脱着モード計算結果の詳細比較")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    
    from config.sim_conditions import SimulationConditions
    from core.state import StateVariables
    from core.physics import operation_models
    from operation_modes import execute_vacuum_desorption
    
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    # 塔2は初期圧力が低いので真空脱着に適している
    tower_conds = sim_conds.get_tower(2)
    
    # 同じ初期状態で計算
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # 旧コード
    old_result = operation_models.vacuum_desorption(
        tower_conds=tower_conds,
        state_manager=old_state,
        tower_num=2
    )
    
    # 新コード
    new_result = execute_vacuum_desorption(
        tower_conds=tower_conds,
        state_manager=new_state,
        tower_num=2
    )
    
    differences = compare_results(old_result, new_result)
    
    if not differences:
        print("✅ 真空脱着モード: 全ての計算結果が一致")
        return True
    else:
        print("❌ 真空脱着モード: 差異あり")
        for diff in differences[:10]:
            print(f"  {diff}")
        if len(differences) > 10:
            print(f"  ... 他 {len(differences) - 10} 件")
        return False


def main():
    """メイン処理"""
    print("=" * 60)
    print("Phase 5: 計算結果詳細比較テスト")
    print("=" * 60)
    print("旧コード（core/physics/operation_models.py）と")
    print("新コード（operation_modes/）の計算結果を比較します")
    print()
    
    results = []
    
    results.append(("停止モード", test_stop_mode_comparison()))
    results.append(("流通吸着モード", test_flow_adsorption_comparison()))
    results.append(("真空脱着モード", test_vacuum_desorption_comparison()))
    
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 全テスト成功: 新旧コードの計算結果が一致しています")
    else:
        print("❗ 一部テスト失敗: 計算結果に差異があります")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
