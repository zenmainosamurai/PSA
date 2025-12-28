#!/usr/bin/env python3
"""
シミュレーション全体の動作確認テスト

CSV/XLSX出力をスキップして、計算ロジックのみを検証します。
"""
import sys
import os
import time
sys.path.insert(0, "/home/user/webapp")
os.chdir("/home/user/webapp")

# モンキーパッチを有効化（CoolProp高速化）
import utils.prop_table

from config.sim_conditions import SimulationConditions
from state import StateVariables
from process.process_executor import execute_mode_list, prepare_batch_adsorption_pressure
from process.simulation_results import SimulationResults
import pandas as pd


def run_full_simulation(cond_id: str) -> dict:
    """
    シミュレーション全体を実行（出力なし）
    
    Args:
        cond_id: 条件ID
        
    Returns:
        実行結果の要約
    """
    print(f"\n{'='*60}")
    print(f"シミュレーション実行: {cond_id}")
    print(f"{'='*60}")
    
    # 条件読み込み
    sim_conds = SimulationConditions(cond_id)
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    dt = sim_conds.get_tower(1).common.calculation_step_time
    
    print(f"塔数: {num_towers}, ストリーム数: {num_streams}, セクション数: {num_sections}")
    print(f"計算時間刻み: {dt} min")
    
    # 稼働工程表読み込み
    from common.paths import CONDITIONS_DIR
    operation_schedule_path = f"{CONDITIONS_DIR}{cond_id}/稼働工程表.xlsx"
    df_operation = pd.read_excel(operation_schedule_path, index_col="工程", sheet_name="工程")
    num_processes = len(df_operation)
    print(f"総工程数: {num_processes}")
    
    # 状態変数初期化
    state_manager = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # シミュレーション結果格納
    simulation_results = SimulationResults()
    for tower_num in range(1, 1 + num_towers):
        simulation_results.initialize_tower(tower_num)
    
    # 残留ガス組成
    residual_gas_composition = None
    
    # タイムスタンプ
    timestamp = 0.0
    total_steps = 0
    
    start_time = time.time()
    
    # 全工程ループ
    for process_index in df_operation.index:
        row = df_operation.loc[process_index]
        mode_list = [row[f"塔{i}"] for i in range(1, num_towers + 1)]
        termination_cond_str = row["終了条件"]
        
        print(f"\n工程 {process_index}: {mode_list}")
        print(f"  終了条件: {termination_cond_str}")
        
        # バッチ吸着の圧力準備
        prepare_batch_adsorption_pressure(state_manager, sim_conds, mode_list)
        
        # 終了条件の解析
        from process.termination_conditions import should_continue_process
        
        # 工程内ループ
        timestamp_p = 0.0
        step_count = 0
        max_steps = 10000  # 安全のため上限設定
        
        while should_continue_process(termination_cond_str, state_manager, timestamp, timestamp_p, num_sections):
            # 1ステップ実行
            outputs, residual_gas_composition = execute_mode_list(
                sim_conds, mode_list, state_manager, residual_gas_composition
            )
            
            timestamp_p += dt
            step_count += 1
            total_steps += 1
            
            # タイムアウト（20分）
            if timestamp_p >= 20:
                print(f"  警告: タイムアウト（20分超過）")
                break
            
            # 安全のため
            if step_count >= max_steps:
                print(f"  警告: 最大ステップ数到達")
                break
        
        timestamp += timestamp_p
        print(f"  完了: {step_count}ステップ, timestamp={timestamp:.2f}min")
        
        # 塔の状態を表示（最終工程または5工程ごと）
        if process_index == num_processes or process_index % 5 == 0:
            for tower_num in range(1, num_towers + 1):
                tower = state_manager.towers[tower_num]
                print(f"    塔{tower_num}: 圧力={tower.total_press:.4f}MPa, "
                      f"温度(1,1)={tower.temp[0,0]:.2f}℃, "
                      f"吸着量(1,1)={tower.loading[0,0]:.2f}")
    
    elapsed_time = time.time() - start_time
    
    # 結果要約
    result = {
        "cond_id": cond_id,
        "num_processes": num_processes,
        "total_steps": total_steps,
        "final_timestamp": timestamp,
        "elapsed_time": elapsed_time,
    }
    
    print(f"\n{'='*60}")
    print(f"シミュレーション完了")
    print(f"  総工程数: {num_processes}")
    print(f"  総ステップ数: {total_steps}")
    print(f"  最終タイムスタンプ: {timestamp:.2f} min")
    print(f"  実行時間: {elapsed_time:.1f} 秒")
    print(f"{'='*60}")
    
    return result


def main():
    """メイン処理"""
    print("="*60)
    print("シミュレーション全体動作確認テスト")
    print("="*60)
    
    # テスト条件
    cond_id = "5_08_mod_logging2"
    
    try:
        result = run_full_simulation(cond_id)
        
        # 検証
        assert result["num_processes"] > 0, "工程数が0"
        assert result["total_steps"] > 0, "ステップ数が0"
        assert result["final_timestamp"] > 0, "タイムスタンプが0"
        
        print("\n" + "="*60)
        print("✅ テスト成功")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
