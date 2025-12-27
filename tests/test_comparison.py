"""従来コードと新コードの動作比較テスト

PSA担当者向け説明:
リファクタリング後のコードが従来と同じ結果を出力することを確認します。

テスト内容:
1. 従来シミュレーター（GasAdosorptionBreakthroughsimulator）の実行
2. 新シミュレーター（PSASimulator）の実行
3. 出力結果の比較
"""

import os
import sys
import shutil
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


def run_old_simulator(cond_id: str, output_dir: str) -> bool:
    """従来シミュレーターの実行"""
    print(f"[OLD] 従来シミュレーター実行中: {cond_id}")
    try:
        from core import GasAdosorptionBreakthroughsimulator
        
        instance = GasAdosorptionBreakthroughsimulator(cond_id)
        instance.execute_simulation(output_folderpath=output_dir)
        print(f"[OLD] 完了: {output_dir}")
        return True
    except Exception as e:
        print(f"[OLD] エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_new_simulator(cond_id: str, output_dir: str) -> bool:
    """新シミュレーターの実行"""
    print(f"[NEW] 新シミュレーター実行中: {cond_id}")
    try:
        from process import PSASimulator
        
        simulator = PSASimulator(cond_id)
        simulator.run(output_path=output_dir)
        print(f"[NEW] 完了: {output_dir}")
        return True
    except Exception as e:
        print(f"[NEW] エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_csv_files(old_dir: str, new_dir: str) -> dict:
    """CSVファイルの比較"""
    results = {
        "matched": [],
        "mismatched": [],
        "old_only": [],
        "new_only": [],
        "errors": [],
    }
    
    old_path = Path(old_dir)
    new_path = Path(new_dir)
    
    # 両方のディレクトリからCSVファイルを収集
    old_csvs = set()
    new_csvs = set()
    
    if old_path.exists():
        old_csvs = {f.relative_to(old_path) for f in old_path.rglob("*.csv")}
    if new_path.exists():
        new_csvs = {f.relative_to(new_path) for f in new_path.rglob("*.csv")}
    
    # 両方に存在するファイルを比較
    common_files = old_csvs & new_csvs
    results["old_only"] = list(old_csvs - new_csvs)
    results["new_only"] = list(new_csvs - old_csvs)
    
    for csv_file in common_files:
        old_file = old_path / csv_file
        new_file = new_path / csv_file
        
        try:
            # CSVを読み込み
            df_old = pd.read_csv(old_file, encoding='shift-jis', index_col=0)
            df_new = pd.read_csv(new_file, encoding='shift-jis', index_col=0)
            
            # 形状チェック
            if df_old.shape != df_new.shape:
                results["mismatched"].append({
                    "file": str(csv_file),
                    "reason": f"形状不一致: old={df_old.shape}, new={df_new.shape}"
                })
                continue
            
            # 数値比較（相対誤差1e-6以内を許容）
            if df_old.select_dtypes(include=[np.number]).empty:
                # 数値列がない場合は文字列比較
                if df_old.equals(df_new):
                    results["matched"].append(str(csv_file))
                else:
                    results["mismatched"].append({
                        "file": str(csv_file),
                        "reason": "内容不一致"
                    })
            else:
                # 数値列がある場合は近似比較
                numeric_old = df_old.select_dtypes(include=[np.number])
                numeric_new = df_new.select_dtypes(include=[np.number])
                
                if numeric_old.shape != numeric_new.shape:
                    results["mismatched"].append({
                        "file": str(csv_file),
                        "reason": "数値列形状不一致"
                    })
                    continue
                
                # 相対誤差の計算
                with np.errstate(divide='ignore', invalid='ignore'):
                    relative_diff = np.abs(numeric_old.values - numeric_new.values)
                    max_vals = np.maximum(np.abs(numeric_old.values), np.abs(numeric_new.values))
                    relative_error = np.where(max_vals > 1e-10, relative_diff / max_vals, relative_diff)
                
                max_error = np.nanmax(relative_error)
                
                if max_error < 1e-6 or np.isnan(max_error):
                    results["matched"].append(str(csv_file))
                else:
                    # 誤差の位置を特定
                    error_idx = np.unravel_index(np.nanargmax(relative_error), relative_error.shape)
                    results["mismatched"].append({
                        "file": str(csv_file),
                        "reason": f"数値誤差: max={max_error:.2e}, at row={error_idx[0]}, col={error_idx[1]}"
                    })
        
        except Exception as e:
            results["errors"].append({
                "file": str(csv_file),
                "error": str(e)
            })
    
    return results


def print_comparison_results(results: dict):
    """比較結果の表示"""
    print("\n" + "=" * 60)
    print("比較結果サマリー")
    print("=" * 60)
    
    print(f"\n✅ 一致: {len(results['matched'])} ファイル")
    if results['matched']:
        for f in results['matched'][:5]:
            print(f"   - {f}")
        if len(results['matched']) > 5:
            print(f"   ... 他 {len(results['matched']) - 5} ファイル")
    
    print(f"\n❌ 不一致: {len(results['mismatched'])} ファイル")
    for item in results['mismatched']:
        print(f"   - {item['file']}: {item['reason']}")
    
    print(f"\n⚠️ 旧のみ: {len(results['old_only'])} ファイル")
    for f in results['old_only']:
        print(f"   - {f}")
    
    print(f"\n⚠️ 新のみ: {len(results['new_only'])} ファイル")
    for f in results['new_only']:
        print(f"   - {f}")
    
    print(f"\n💥 エラー: {len(results['errors'])} ファイル")
    for item in results['errors']:
        print(f"   - {item['file']}: {item['error']}")
    
    print("\n" + "=" * 60)
    
    # 判定
    if len(results['mismatched']) == 0 and len(results['errors']) == 0:
        print("🎉 テスト成功: 全ファイルが一致しました")
        return True
    else:
        print("❗ テスト失敗: 不一致またはエラーがあります")
        return False


def main():
    """メイン処理"""
    cond_id = "5_08_mod_logging2"
    
    # 出力先ディレクトリ
    old_output_dir = str(project_root / "output" / f"{cond_id}_old/")
    new_output_dir = str(project_root / "output" / f"{cond_id}_new/")
    
    # 既存の出力を削除
    for d in [old_output_dir, new_output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    print("=" * 60)
    print("Phase 5: 従来コードと新コードの動作比較テスト")
    print("=" * 60)
    print(f"条件ID: {cond_id}")
    print(f"旧出力: {old_output_dir}")
    print(f"新出力: {new_output_dir}")
    print()
    
    # 従来シミュレーター実行
    old_success = run_old_simulator(cond_id, old_output_dir)
    
    if not old_success:
        print("\n❌ 従来シミュレーターの実行に失敗しました")
        return False
    
    # 新シミュレーター実行
    new_success = run_new_simulator(cond_id, new_output_dir)
    
    if not new_success:
        print("\n❌ 新シミュレーターの実行に失敗しました")
        return False
    
    # 結果比較
    print("\n[COMPARE] CSVファイルを比較中...")
    results = compare_csv_files(old_output_dir, new_output_dir)
    
    return print_comparison_results(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
