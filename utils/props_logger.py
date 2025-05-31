# -*- coding: utf-8 -*-
"""
PropsSI ロガー
------------------------------------------------
CoolProp.CoolProp.PropsSI をラップし、
呼び出しパラメータ (fluid, prop, T[K], P[Pa]) を CSV へ記録。
------------------------------------------------
使い方:
    import utils.props_logger   # これだけでパッチされる

環境変数 LOG_PROPS_FILE が設定されていればそのパスに、
無ければ ./props_usage.csv へ追記する。
"""

import os
from csv import writer
from pathlib import Path
from datetime import datetime
import CoolProp.CoolProp as _CP

# ==== 1) オリジナル関数を退避（ここが重要） ====
_ORIG_PROPSI = _CP.PropsSI  # -> 以後はこれだけを呼ぶ

# ==== 2) ログファイル準備 ====
_LOG_FILE = Path(os.getenv("LOG_PROPS_FILE", "props_usage.csv"))
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

if not _LOG_FILE.exists():
    _LOG_FILE.write_text("timestamp,fluid,prop,T_K,P_Pa\n", encoding="utf-8")

_log_fh = _LOG_FILE.open("w", newline="", encoding="utf-8")
_csv_writer = writer(_log_fh)


def _log(fluid: str, prop: str, T: float, P: float):
    _csv_writer.writerow([datetime.utcnow().isoformat(), fluid, prop, f"{T:.6f}", f"{P:.2f}"])
    _log_fh.flush()


# ==== 3) ラッパー ====
def PropsSI(output_key: str, inp1: str, val1: float, inp2: str, val2: float, fluid: str):

    # --- まずオリジナルを実行 ---
    result = _ORIG_PROPSI(output_key, inp1, val1, inp2, val2, fluid)

    # --- ログ条件判定 ---
    inp1, inp2 = inp1.upper(), inp2.upper()
    if {inp1, inp2} == {"T", "P"}:
        T = val1 if inp1 == "T" else val2
        P = val2 if inp2 == "P" else val1
        _log(fluid.lower(), output_key, float(T), float(P))

    return result


# ==== 4) モンキーパッチ ====
_CP.PropsSI_original = _ORIG_PROPSI  # 予備的に残しておく
_CP.PropsSI = PropsSI
