# -*- coding: utf-8 -*-
"""
シミュレーションの際、タイムステップ・セルごとにCP.PropsSIを重複呼び出しすると計算コストが大きいため、
事前によく使う値をテーブル化しておく。P固定、Tのみの1次元テーブルを生成する。
実行例:
    python build_prop_table.py \
        --t_min 280 --t_max 300 --t_step 0.25 \
        --p_fixed 101235 \
        --fluids co2 nitrogen \
        --props L V CPMASS D \
        --outfile prop_table.npz
"""

import argparse
from itertools import product
from pathlib import Path
import numpy as np
import CoolProp.CoolProp as CP


# -------------------------------------------------
# 関数定義
# -------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1-D (temperature-only) property table for CoolProp.")
    parser.add_argument("--t_min", type=float, default=260.0, help="Temperature min [K]")
    parser.add_argument("--t_max", type=float, default=370.0, help="Temperature max [K]")
    parser.add_argument("--t_step", type=float, default=0.1, help="Temperature step [K]")
    parser.add_argument("--p_fixed", type=float, default=101325.0, help="Fixed pressure [Pa]")
    parser.add_argument(
        "--fluids",
        nargs="+",
        default=["co2", "nitrogen"],
        help="List of fluids (CoolProp names, space-separated)",
    )
    parser.add_argument(
        "--props",
        nargs="+",
        default=["L", "V", "CPMASS", "D"],
        help="List of property keys (CoolProp)",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="prop_table.npz",
        help="Output *.npz file",
    )
    return parser.parse_args()


def build_table(
    t_min: float,
    t_max: float,
    t_step: float,
    p_fixed: float,
    fluids: list[str],
    props: list[str],
) -> dict[str, np.ndarray]:
    """Calculate property arrays and return dict ready for np.savez."""
    t_grid = np.arange(t_min, t_max + t_step, t_step, dtype=np.float32)
    out = {"T": t_grid, "P_FIXED": np.array(p_fixed, dtype=np.float32)}

    for fluid, prop in product(fluids, props):
        key = f"{fluid.lower()}_{prop}"
        out[key] = np.array(
            [CP.PropsSI(prop, "T", T, "P", p_fixed, fluid) for T in t_grid],
            dtype=np.float32,
        )

    return out


def save_table(arr_dict: dict[str, np.ndarray], outfile: str) -> None:
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outfile, **arr_dict)
    print(f"[DONE] Saved {outfile} " f"(T points: {arr_dict['T'].size}, arrays: {len(arr_dict)-1})")


def main() -> None:
    args = parse_args()
    table = build_table(
        t_min=args.t_min,
        t_max=args.t_max,
        t_step=args.t_step,
        p_fixed=args.p_fixed,
        fluids=args.fluids,
        props=args.props,
    )
    save_table(table, args.outfile)


if __name__ == "__main__":
    main()
