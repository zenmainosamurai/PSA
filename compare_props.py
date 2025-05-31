"""
CP.PropsSIのL, V, CPMASS, Dについて高速版とオリジナルの誤差を比較する
"""

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
import utils.prop_table as pt  # ← PropsSI高速化版

# ----------------  設定  ----------------
FLUID = "co2"
PROP_KEYS = ["L", "V", "CPMASS", "D"]
P_FIXED = pt._P_FIXED  # テーブル想定圧力 [Pa]
N_SAMPLES = 100  # 乱数サンプル数
SEED = 42  # 変えると乱数が変わる（再現性用）

rng = np.random.default_rng(SEED)
T_samples = rng.uniform(pt._T_MIN, pt._T_MAX, size=N_SAMPLES)

Props_fast = CP.PropsSI
Props_orig = CP.PropsSI_original

# -------------  計算ループ  -------------
records = []
for T in T_samples:
    rec = {"T[K]": T}
    for k in PROP_KEYS:
        fast = Props_fast(k, "T", T, "P", P_FIXED, FLUID)
        orig = Props_orig(k, "T", T, "P", P_FIXED, FLUID)
        abs_err = abs(fast - orig)
        rel_err = abs_err / abs(orig) if orig else np.nan
        rec[f"{k}_abs_err"] = abs_err
        rec[f"{k}_rel_err"] = rel_err
    records.append(rec)

# -------------  結果表示  -------------
df = pd.DataFrame(records)
pd.set_option("display.float_format", "{:11.4e}".format)
print(df.to_string(index=False))
