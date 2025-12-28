"""
全工程のインデックス0オリジン化リファクタリングテスト

修正前と修正後の全工程の計算結果が一致することを確認するスクリプト
"""

import json
import numpy as np
from config.sim_conditions import SimulationConditions
from state import StateVariables
from process import SimulationRunner
from process.simulation_io import SimulationIO

import utils.prop_table


def extract_state_snapshot(state_manager: StateVariables, num_towers: int) -> dict:
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
            "cumulative_co2_recovered": tower.cumulative_co2_recovered,
            "cumulative_n2_recovered": tower.cumulative_n2_recovered,
        }
    return snapshot


def extract_time_series_data(output) -> dict:
    """時系列データを抽出"""
    time_series = {}
    for tower_num, tower_results in output.results.tower_simulation_results.items():
        ts_data = tower_results.time_series_data
        time_series[f"tower_{tower_num}"] = {
            "timestamps": ts_data.timestamps,
            "total_pressure": [o.get("total_pressure", None) for o in ts_data.others],
            "num_records": len(ts_data.timestamps),
        }
    return time_series


def run_full_simulation(cond_id: str) -> dict:
    """
    全工程のシミュレーションを実行して結果を返す
    
    Args:
        cond_id: 条件ID
    
    Returns:
        dict: 結果のスナップショット
    """
    # 条件読み込み
    io = SimulationIO()
    sim_conds = io.load_conditions(cond_id)
    df_operation = io.load_operation_schedule(cond_id)
    
    print(f"全工程数: {len(df_operation)}")
    
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
        "num_processes": len(df_operation),
        "final_timestamp": output.final_timestamp,
        "process_completion_log": output.process_completion_log,
        "final_state": extract_state_snapshot(state_manager, num_towers),
        "time_series_summary": extract_time_series_data(output),
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
    
    # 工程数の比較
    if before["num_processes"] != after["num_processes"]:
        print(f"MISMATCH: num_processes - before: {before['num_processes']}, after: {after['num_processes']}")
        all_match = False
    else:
        print(f"OK: num_processes = {before['num_processes']}")
    
    # タイムスタンプの比較
    if abs(before["final_timestamp"] - after["final_timestamp"]) > tolerance:
        print(f"MISMATCH: final_timestamp - before: {before['final_timestamp']}, after: {after['final_timestamp']}")
        all_match = False
    else:
        print(f"OK: final_timestamp = {before['final_timestamp']}")
    
    # 工程完了ログの比較
    for proc_id, ts_before in before["process_completion_log"].items():
        ts_after = after["process_completion_log"].get(proc_id)
        if ts_after is None:
            print(f"MISMATCH: process {proc_id} missing in after")
            all_match = False
        elif abs(ts_before - ts_after) > tolerance:
            print(f"MISMATCH: process {proc_id} timestamp - before: {ts_before}, after: {ts_after}")
            all_match = False
    print(f"OK: process_completion_log - {len(before['process_completion_log'])} processes checked")
    
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
                diff = abs(val_before - val_after) if val_before is not None and val_after is not None else 0
                if diff > tolerance:
                    print(f"MISMATCH: {tower_key}.{var_name} - before: {val_before}, after: {val_after}")
                    all_match = False
                else:
                    print(f"OK: {tower_key}.{var_name} = {val_before}")
    
    # 時系列データのサマリー比較
    for tower_key, ts_before in before.get("time_series_summary", {}).items():
        ts_after = after.get("time_series_summary", {}).get(tower_key, {})
        if ts_before.get("num_records") != ts_after.get("num_records"):
            print(f"MISMATCH: {tower_key} time_series num_records - before: {ts_before.get('num_records')}, after: {ts_after.get('num_records')}")
            all_match = False
        else:
            print(f"OK: {tower_key} time_series num_records = {ts_before.get('num_records')}")
    
    return all_match


def save_results(results: dict, filename: str):
    """結果をJSONファイルに保存"""
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
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "before":
            print("=" * 60)
            print("Running BEFORE modification (FULL process) test...")
            print("=" * 60)
            results = run_full_simulation(cond_id)
            save_results(results, "test_full_results_before.json")
            print("Before results saved.")
            
        elif sys.argv[1] == "after":
            print("=" * 60)
            print("Running AFTER modification (FULL process) test...")
            print("=" * 60)
            results = run_full_simulation(cond_id)
            save_results(results, "test_full_results_after.json")
            print("After results saved.")
            
        elif sys.argv[1] == "compare":
            print("=" * 60)
            print("Comparing BEFORE and AFTER (FULL process) results...")
            print("=" * 60)
            before = load_results("test_full_results_before.json")
            after = load_results("test_full_results_after.json")
            
            if compare_results(before, after):
                print("\n" + "=" * 60)
                print("SUCCESS: All FULL process results match!")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("FAILURE: FULL process results do not match!")
                print("=" * 60)
                sys.exit(1)
    else:
        print("Usage: python test_full_process.py [before|after|compare]")
