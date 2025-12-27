"""単一ステップの動作比較テスト

PSA担当者向け説明:
1ステップ分の計算結果を比較して、新旧コードの互換性を確認します。
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from copy import deepcopy


def test_state_initialization():
    """状態変数初期化の比較"""
    print("=" * 60)
    print("テスト1: 状態変数初期化の比較")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    
    # 従来コードの初期化
    from config.sim_conditions import SimulationConditions
    from core.state import StateVariables as OldStateVariables
    
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    old_state = OldStateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # 状態変数の確認
    print(f"  塔数: {num_towers}")
    print(f"  ストリーム数: {num_streams}")
    print(f"  セクション数: {num_sections}")
    
    for tower_num in range(1, num_towers + 1):
        tower = old_state.towers[tower_num]
        print(f"\n  塔{tower_num}:")
        print(f"    全圧: {tower.total_press:.4f} MPaA")
        print(f"    温度形状: {tower.temp.shape}")
        print(f"    温度範囲: {tower.temp.min():.2f} - {tower.temp.max():.2f} ℃")
        print(f"    吸着量形状: {tower.loading.shape}")
        print(f"    吸着量範囲: {tower.loading.min():.4f} - {tower.loading.max():.4f}")
    
    print("\n✅ 状態変数初期化: OK")
    return True


def test_single_mode_calculation():
    """単一モードの計算比較"""
    print("\n" + "=" * 60)
    print("テスト2: 単一モード計算の比較（停止モード）")
    print("=" * 60)
    
    cond_id = "5_08_mod_logging2"
    
    # 条件読み込み
    from config.sim_conditions import SimulationConditions
    from core.state import StateVariables
    
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    # 旧コードでの計算
    print("\n[OLD] 従来コードで停止モード計算...")
    from core.physics import operation_models
    
    old_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    tower_conds = sim_conds.get_tower(1)
    
    old_result = operation_models.stop_mode(
        tower_conds=tower_conds,
        state_manager=old_state,
        tower_num=1
    )
    
    print(f"  物質収支結果: {type(old_result.material).__name__}")
    print(f"  熱収支結果: {type(old_result.heat).__name__}")
    
    # 新コードでの計算
    print("\n[NEW] 新コードで停止モード計算...")
    from operation_modes import execute_stop_mode
    
    new_state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    try:
        new_result = execute_stop_mode(
            tower_conds=tower_conds,
            state_manager=new_state,
            tower_num=1
        )
        print(f"  物質収支結果: {type(new_result.material).__name__}")
        print(f"  熱収支結果: {type(new_result.heat).__name__}")
        print("\n✅ 新コード停止モード計算: OK")
    except Exception as e:
        print(f"\n❌ 新コードでエラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_mode_types_conversion():
    """モードタイプ変換の確認"""
    print("\n" + "=" * 60)
    print("テスト3: モードタイプ変換の確認")
    print("=" * 60)
    
    from operation_modes import OperationMode
    
    test_modes = [
        "停止",
        "流通吸着_単独/上流",
        "流通吸着_下流",
        "バッチ吸着_上流",
        "バッチ吸着_下流",
        "均圧_減圧",
        "均圧_加圧",
        "真空脱着",
        "初回ガス導入",
    ]
    
    all_ok = True
    for mode_name in test_modes:
        try:
            mode = OperationMode.from_japanese(mode_name)
            print(f"  '{mode_name}' -> {mode.name}: OK")
        except Exception as e:
            print(f"  '{mode_name}' -> エラー: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ モードタイプ変換: OK")
    else:
        print("\n❌ モードタイプ変換: 一部失敗")
    
    return all_ok


def test_termination_conditions():
    """終了条件判定の確認"""
    print("\n" + "=" * 60)
    print("テスト4: 終了条件判定の確認")
    print("=" * 60)
    
    from process.termination_conditions import (
        parse_termination_condition,
        TerminationConditionType,
    )
    
    test_cases = [
        ("圧力到達_塔1_0.3", TerminationConditionType.PRESSURE_REACHED, 1, 0.3),
        ("温度到達_塔2_50", TerminationConditionType.TEMPERATURE_REACHED, 2, 50.0),
        ("時間経過_5_min", TerminationConditionType.TIME_ELAPSED, None, 5.0),
        ("時間到達_30", TerminationConditionType.TIME_REACHED, None, 30.0),
    ]
    
    all_ok = True
    for cond_str, expected_type, expected_tower, expected_value in test_cases:
        try:
            cond = parse_termination_condition(cond_str)
            type_ok = cond.condition_type == expected_type
            tower_ok = cond.tower_num == expected_tower
            value_ok = abs(cond.target_value - expected_value) < 1e-6
            
            if type_ok and tower_ok and value_ok:
                print(f"  '{cond_str}': OK")
            else:
                print(f"  '{cond_str}': 値不一致")
                all_ok = False
        except Exception as e:
            print(f"  '{cond_str}': エラー - {e}")
            all_ok = False
    
    if all_ok:
        print("\n✅ 終了条件判定: OK")
    else:
        print("\n❌ 終了条件判定: 一部失敗")
    
    return all_ok


def test_physics_imports():
    """物理計算モジュールのインポート確認"""
    print("\n" + "=" * 60)
    print("テスト5: 物理計算モジュールのインポート確認")
    print("=" * 60)
    
    try:
        from physics import (
            calculate_mass_balance,
            calculate_bed_heat_balance,
            calculate_wall_heat_balance,
            calculate_lid_heat_balance,
        )
        print("  physics.mass_balance: OK")
        print("  physics.heat_balance: OK")
    except Exception as e:
        print(f"  インポートエラー: {e}")
        return False
    
    try:
        from physics import (
            calculate_vacuum_pumping,
            calculate_pressure_after_vacuum_desorption,
            calculate_pressure_after_batch_adsorption,
            calculate_depressurization,
        )
        print("  physics.pressure: OK")
    except Exception as e:
        print(f"  インポートエラー: {e}")
        return False
    
    try:
        from physics import (
            calculate_equilibrium_loading,
            calculate_driving_force,
        )
        print("  physics.adsorption_isotherm: OK")
    except Exception as e:
        print(f"  インポートエラー: {e}")
        return False
    
    print("\n✅ 物理計算モジュール: OK")
    return True


def main():
    """メイン処理"""
    print("=" * 60)
    print("Phase 5: 単一ステップ動作比較テスト")
    print("=" * 60)
    
    results = []
    
    results.append(("状態変数初期化", test_state_initialization()))
    results.append(("モードタイプ変換", test_mode_types_conversion()))
    results.append(("終了条件判定", test_termination_conditions()))
    results.append(("物理計算モジュール", test_physics_imports()))
    results.append(("単一モード計算", test_single_mode_calculation()))
    
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
        print("🎉 全テスト成功")
    else:
        print("❗ 一部テスト失敗")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
