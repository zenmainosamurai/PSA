# -*- coding: utf-8 -*-
"""
utils.prop_table
-------------------------------------------------
* 温度 1 D テーブル（P 固定）の高速 PropsSI ラッパー
* 固定圧力 P_FIXED は prop_table.npz 内のスカラー "P_FIXED" から取得
* 固定圧力と少しでも異なる場合は即座に CoolProp へフォールバック
"""
import atexit
from pathlib import Path
from functools import lru_cache
import numpy as np
import CoolProp.CoolProp as _CP

# -------------------------------------------------
# 1. テーブル読み込み
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_TABLE_FILE = BASE_DIR / "data" / "prop_table.npz"
if DATA_TABLE_FILE.exists():
    TABLE_FILE = DATA_TABLE_FILE

print(f"Loading prop_table from: {TABLE_FILE}")

if not TABLE_FILE.exists():
    raise FileNotFoundError("prop_table.npz が見つかりません。\n" "先に build_prop_table.py を実行してください。")

_npz = np.load(TABLE_FILE, allow_pickle=False)
_T = _npz["T"].astype(np.float32)  # 温度軸
_P_FIXED = float(_npz["P_FIXED"])  # スカラー (Pa)

_prop_arrays = {tuple(k.split("_", 1)): _npz[k].astype(np.float32) for k in _npz.files if k not in ("T", "P_FIXED")}

_T_MIN, _T_MAX = float(_T[0]), float(_T[-1])


# -------------------------------------------------
# 2. 線形補間（キャッシュ付き）
# -------------------------------------------------
@lru_cache(maxsize=32_768)
def _interp(fluid: str, prop: str, T: float):
    """T が範囲内なら補間値、範囲外なら None"""
    if not (_T_MIN <= T <= _T_MAX):
        return None
    arr = _prop_arrays.get((fluid, prop))
    if arr is None:
        return None
    return float(np.interp(T, _T, arr))


_TOTAL_CALLS = 0  # PropsSI ラッパーが呼ばれた総数
_FAST_HITS = 0  # テーブル補間で返せた回数


@atexit.register
def _print_stats():
    if _TOTAL_CALLS == 0:
        return
    hit_ratio = 100.0 * _FAST_HITS / _TOTAL_CALLS
    print(f"[prop_table] PropsSI total={_TOTAL_CALLS}, " f"fastpath={_FAST_HITS} ({hit_ratio:.1f}% hit)")


# -------------------------------------------------
# 3. ラッパー定義
# -------------------------------------------------
_ORIG_PROPSI = _CP.PropsSI  # オリジナル退避


def PropsSI(output_key: str, inp1: str, val1: float, inp2: str, val2: float, fluid: str):
    global _TOTAL_CALLS, _FAST_HITS
    _TOTAL_CALLS += 1
    inp1, inp2 = inp1.upper(), inp2.upper()
    fluid_lc = fluid.lower()

    # (T, P) or (P, T) のみ対象
    if {inp1, inp2} == {"T", "P"}:
        T = val1 if inp1 == "T" else val2
        P = val2 if inp2 == "P" else val1
        # 圧力が固定値と完全一致するときのみ補間
        if P == _P_FIXED:
            v = _interp(fluid_lc, output_key, T)
            if v is not None:
                _FAST_HITS += 1
                return v

    # フォールバック
    return _ORIG_PROPSI(output_key, inp1, val1, inp2, val2, fluid)


# -------------------------------------------------
# 4. モンキーパッチ
# -------------------------------------------------
_CP.PropsSI_original = _ORIG_PROPSI
_CP.PropsSI = PropsSI
