#!/usr/bin/env python3
"""
シミュレーション動作確認テスト

新コード（operation_modes/）を使用したシミュレーターの動作確認
"""
import sys
import os
sys.path.insert(0, "/home/user/webapp")
os.chdir("/home/user/webapp")

# 物性テーブル高速化
import utils.prop_table

import numpy as np
import pandas as pd

from config.sim_conditions import SimulationConditions
from core.state.state_variables import StateVariables
from core import GasAdosorptionBreakthroughsimulator
from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
from utils import const


def test_simulator_initialization():
    """シミュレーター初期化テスト"""
    print("テスト1: シミュレーター初期化...")
    sim = GasAdosorptionBreakthroughsimulator('5_08_mod_logging2')
    assert sim.sim_conds is not None
    assert sim.state_manager is not None
    print("  ✅ 成功")
    return sim


def test_mode_list_execution(sim):
    """モードリスト実行テスト"""
    print("テスト2: モードリスト実行...")
    
    # 停止モード
    mode_list = ['停止', '停止', '停止']
    result = sim.calc_adsorption_mode_list(sim.sim_conds, mode_list)
    assert len(result) == 3
    assert all(k in result[1] for k in ['material', 'heat', 'heat_wall', 'heat_lid', 'others'])
    print("  ✅ 停止モード成功")
    
    # 流通吸着モード
    mode_list = ['流通吸着_単独/上流', '流通吸着_下流', '停止']
    result = sim.calc_adsorption_mode_list(sim.sim_conds, mode_list)
    assert len(result) == 3
    print("  ✅ 流通吸着モード成功")
    
    # 均圧モード
    mode_list = ['流通吸着_単独/上流', '均圧_加圧', '均圧_減圧']
    result = sim.calc_adsorption_mode_list(sim.sim_conds, mode_list)
    assert len(result) == 3
    print("  ✅ 均圧モード成功")


def test_process_executor():
    """process_executor単体テスト"""
    print("テスト3: process_executor単体テスト...")
    
    sim_conds = SimulationConditions('5_08_mod_logging2')
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # 停止モード
    mode_list = ['停止', '停止', '停止']
    outputs, residual = execute_mode_list(sim_conds, mode_list, state, None)
    assert len(outputs) == 3
    print("  ✅ execute_mode_list成功")
    
    # 圧力平均化
    mode_list = ['バッチ吸着_上流', 'バッチ吸着_下流', '停止']
    prepare_batch_adsorption_pressure(state, sim_conds, mode_list)
    print("  ✅ prepare_batch_adsorption_pressure成功")


def test_all_modes():
    """全モードテスト"""
    print("テスト4: 全モードテスト...")
    
    sim_conds = SimulationConditions('5_08_mod_logging2')
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # 全モードリスト
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
    
    for mode_list in mode_lists:
        state = StateVariables(num_towers, num_streams, num_sections, sim_conds)
        prepare_batch_adsorption_pressure(state, sim_conds, mode_list)
        outputs, _ = execute_mode_list(sim_conds, mode_list, state, None)
        assert len(outputs) == 3, f"Failed for {mode_list}"
    
    print("  ✅ 全モード成功")


def main():
    """メイン処理"""
    print("=" * 60)
    print("シミュレーション動作確認テスト")
    print("=" * 60)
    
    try:
        sim = test_simulator_initialization()
        test_mode_list_execution(sim)
        test_process_executor()
        test_all_modes()
        
        print("\n" + "=" * 60)
        print("✅ 全テスト成功")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
