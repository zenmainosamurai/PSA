"""
インデックス0オリジン化リファクタリングのテスト

修正前と修正後の計算結果が一致することを確認するスクリプト
"""

import json
import numpy as np
from state import StateVariables
from process import SimulationRunner
from process.simulation_io import SimulationIO



def extract_state_snapshot(state_manager: StateVariables, num_towers: int, num_streams: int, num_sections: int) -> dict:
    """状態変数のスナップショットを取得"""
    snapshot = {}
    for tower_num in range(1, num_towers + 1):
        tower = state_manager.towers[tower_num]
        snapshot[f"tower_{tower_num}"] = {
            "temp": tower.temp.tolist(),
            "loading": tower.loading.tolist(),
            "co2_mole_fraction": tower.co2_mole_fraction.tolist(),
            "n2_mole_fraction": tower.n2_mole_fraction.tolist(),
            "total_press": tower.total_press,
            "temp_wall": tower.temp_wall.tolist(),
            "lid_temperature": tower.lid_temperature,
            "bottom_temperature": tower.bottom_temperature,
        }
    return snapshot


def run_limited_simulation(cond_id: str, num_processes: int = 3) -> dict:
    """
    限定的なシミュレーションを実行して結果を返す
    
    Args:
        cond_id: 条件ID
        num_processes: 実行するプロセス数（デフォルト3）
    
    Returns:
        dict: 結果のスナップショット
    """
    # 条件読み込み
    io = SimulationIO()
    sim_conds = io.load_conditions(cond_id)
    df_operation = io.load_operation_schedule(cond_id)
    
    # 最初の数工程だけ実行するために稼働工程表をスライス
    df_operation = df_operation.iloc[:num_processes].copy()
    
    # 状態変数初期化
    num_towers = sim_conds.num_towers
    num_streams = sim_conds.get_tower(1).common.num_streams
    num_sections = sim_conds.get_tower(1).common.num_sections
    
    state_manager = StateVariables(num_towers, num_streams, num_sections, sim_conds)
    
    # シミュレーション実行
    runner = SimulationRunner(sim_conds, state_manager, df_operation)
    output = runner.run()
    
    # 結果のスナップショット
    results = {
        "final_timestamp": output.final_timestamp,
        "process_completion_log": output.process_completion_log,
        "final_state": extract_state_snapshot(state_manager, num_towers, num_streams, num_sections),
    }
    
    # シミュレーション結果からサンプルデータを取得
    if output.results.tower_simulation_results:
        tower_results = output.results.tower_simulation_results[1]
        if tower_results.time_series_data.timestamps:
            last_idx = -1
            results["sample_output"] = {
                "timestamp": tower_results.time_series_data.timestamps[last_idx],
                "total_pressure": tower_results.time_series_data.others[last_idx]["total_pressure"],
            }
    
    return results


def compare_results(before: dict, after: dict, tolerance: float = 1e-10) -> bool:
    """
    2つの結果を比較
    
    Args:
        before: 修正前の結果
        after: 修正後の結果
        tolerance: 許容誤差
    
    Returns:
        bool: 一致する場合True
    """
    all_match = True
    
    # タイムスタンプの比較
    if abs(before["final_timestamp"] - after["final_timestamp"]) > tolerance:
        print(f"MISMATCH: final_timestamp - before: {before['final_timestamp']}, after: {after['final_timestamp']}")
        all_match = False
    else:
        print(f"OK: final_timestamp = {before['final_timestamp']}")
    
    # 状態変数の比較
    for tower_key, tower_before in before["final_state"].items():
        tower_after = after["final_state"][tower_key]
        
        for var_name, val_before in tower_before.items():
            val_after = tower_after[var_name]
            
            if isinstance(val_before, list):
                arr_before = np.array(val_before)
                arr_after = np.array(val_after)
                max_diff = np.max(np.abs(arr_before - arr_after))
                if max_diff > tolerance:
                    print(f"MISMATCH: {tower_key}.{var_name} - max_diff: {max_diff}")
                    all_match = False
                else:
                    print(f"OK: {tower_key}.{var_name} - max_diff: {max_diff:.2e}")
            else:
                diff = abs(val_before - val_after)
                if diff > tolerance:
                    print(f"MISMATCH: {tower_key}.{var_name} - before: {val_before}, after: {val_after}")
                    all_match = False
                else:
                    print(f"OK: {tower_key}.{var_name} = {val_before}")
    
    return all_match


def save_results(results: dict, filename: str):
    """結果をJSONファイルに保存"""
    # numpy配列をリストに変換するためのカスタムエンコーダ
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return super().default(obj)
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved to {filename}")


def load_results(filename: str) -> dict:
    """結果をJSONファイルから読み込み"""
    with open(filename, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    
    cond_id = "5_08_mod_logging2"
    num_processes = 5  # 最初の5工程のみ実行
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "before":
            print("=" * 60)
            print("Running BEFORE modification test...")
            print("=" * 60)
            results = run_limited_simulation(cond_id, num_processes)
            save_results(results, "test_results_before.json")
            print("Before results saved.")
            
        elif sys.argv[1] == "after":
            print("=" * 60)
            print("Running AFTER modification test...")
            print("=" * 60)
            results = run_limited_simulation(cond_id, num_processes)
            save_results(results, "test_results_after.json")
            print("After results saved.")
            
        elif sys.argv[1] == "compare":
            print("=" * 60)
            print("Comparing BEFORE and AFTER results...")
            print("=" * 60)
            before = load_results("test_results_before.json")
            after = load_results("test_results_after.json")
            
            if compare_results(before, after):
                print("\n" + "=" * 60)
                print("SUCCESS: All results match!")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("FAILURE: Results do not match!")
                print("=" * 60)
                sys.exit(1)
    else:
        print("Usage: python test_index_refactor.py [before|after|compare]")
