"""
熱伝達（層伝熱係数 u1・壁-層伝熱係数 hw1）を計算する補助モジュール
旧 base_models._heat_transfer_coef のロジックを小関数に分割して再構成。
"""

import math
from typing import Tuple, Dict

import CoolProp.CoolProp as CP

from state_variables import StateVariables
from sim_conditions import StreamConditions, TowerConditions


# ------------------------------------------------------------------
# 1. 物性値: 気体熱伝導率
# ------------------------------------------------------------------
def compute_gas_k(T_K: float, mf_co2: float, mf_n2: float) -> float:
    """
    温度 T_K [K]・モル分率から混合気体の熱伝導率 [W/m/K] を返す。
    CoolProp の単体値を線形補間（大気圧を仮定）。
    """
    P_ATM = 0.101325e6  # Pa
    k_co2 = CP.PropsSI("L", "T", T_K, "P", P_ATM, "co2")
    k_n2 = CP.PropsSI("L", "T", T_K, "P", P_ATM, "nitrogen")
    return (k_co2 * mf_co2 + k_n2 * mf_n2) / 1000


# ------------------------------------------------------------------
# 2. Yagi–Kunii 放射補正で静止充填層有効熱伝導率 ke0 を求める
# ------------------------------------------------------------------
def _yagi_kunii_radiation(
    T_K: float,
    kf: float,
    kp: float,
    epsilon: float,
    epsilon_p: float,
    dp: float,
) -> float:
    """
    放射補正を含む静止充填層有効熱伝導率 ke0 [W/m/K] を返す。
    """
    # --- Yagi–Kunii Φ ------------------------------------------------
    Phi_1, Phi_2 = 0.15, 0.07
    Phi = Phi_2 + (Phi_1 - Phi_2) * (epsilon - 0.26) / 0.26

    # --- hrv, hrp ----------------------------------------------------
    hrv = 0.227 / (1 + epsilon / 2 / (1 - epsilon) * (1 - epsilon_p) / epsilon_p) * (T_K / 100.0) ** 3
    hrp = 0.227 * epsilon_p / (2 - epsilon_p) * (T_K / 100.0) ** 3

    # --- ksi & ke0/kf -----------------------------------------------
    ksi = 1.0 / Phi + hrp * dp / kf
    ke0_kf = epsilon * (1 + hrv * dp / kf) + (1 - epsilon) / (1 / ksi + 2 * kf / 3 / kp)

    return kf * ke0_kf


# ------------------------------------------------------------------
# 3. 流速補正を含む層/壁-層伝熱係数を求める
# ------------------------------------------------------------------
def _axial_flow_correction(
    ke0: float,
    kf: float,
    dp: float,
    d1: float,
    Pr: float,
    Rep: float,
    epsilon: float,
    Lbed: float,
    num_sec: int,
) -> Tuple[float, float, float, float]:
    """
    ke0 から流速補正を行い、
    ke (有効熱伝導率)・habs (粒子-流体間)・dlat (粒子代表長さ)・hw1_raw を返す。
    """
    # --- ke (流速補正込み) ------------------------------------------
    psi_beta = 1.0985 * (dp / d1) ** 2 - 0.5192 * (dp / d1) + 0.1324  # 充填層有効熱伝導率1
    ke_kf = ke0 / kf + psi_beta * Pr * Rep  # 充填層有効熱伝導率2
    ke = ke_kf * kf  # 充填層有効熱伝導率3 [W/m/K]

    # --- 粒子-流体間伝熱率 -----------------------------------------
    Nup = 0.84 * Rep  # ヌッセルト数
    habs = Nup / dp * kf  # 粒子‐流体間熱伝達率 [W/m2/K]

    # --- 壁-層側 -----------------------------------------------------
    a = 2.0  # 隙間係数
    l0 = dp * a * 2.0 / 2**0.5  # 格子長さ[m]
    dlat = l0 * (1 - epsilon)  # 粒子代表長さ[m]

    c0 = 4.0  # 代表長さ（セクション全長）1
    Lambda_2 = c0 * Pr ** (1.0 / 3.0) * Rep**0.5  # 代表長さ（セクション全長）2
    knew = ke + 1.0 / (1.0 / (0.02 * Pr * Rep) + 2.0 / Lambda_2)  # 代表長さ（セクション全長）3
    Lambda_1 = 2.0 / (kf / ke - kf / knew)  # 代表長さ（セクション全長）4
    b0 = 0.5 * Lambda_1 * d1 / dp * kf / ke  # 代表長さ（セクション全長）5

    Phi_b = 0.0775 * math.log(b0) + 0.028  # 代表長さ（セクション全長）6
    a12 = 0.9107 * math.log(b0) + 2.2395  # 代表長さ（セクション全長）7

    Lp = Lbed / num_sec  # 代表長さ（セクション全長）8 [m]
    y0 = 4.0 * dp / d1 * Lp / d1 * ke_kf / (Pr * Rep)  # 粒子層-壁面伝熱ヌッセルト数 1
    Nupw = dp / d1 * ke_kf * (a12 + Phi_b / y0)  # 粒子層-壁面伝熱ヌッセルト数 1

    hw1_raw = Nupw / dp * kf  # 壁-層ヌッセルト数 → 係数（補正前）

    return ke, habs, dlat, hw1_raw


# ------------------------------------------------------------------
# 4. 外部公開関数: calc_heat_transfer_coef
# ------------------------------------------------------------------
def calc_heat_transfer_coef(
    sim_conds: TowerConditions,
    stream_conds: Dict[int, StreamConditions],
    stream: int,
    section: int,
    temp_now: float,
    mode: int,
    state_manager: StateVariables,
    tower_num: int,
    material_output: dict,
    vacuum_pumping_results: dict | None = None,
) -> tuple[float, float]:
    """層伝熱係数、壁-層伝熱係数を算出する

    Args:
        sim_conds(dict):
        stream_conds(dict):
        stream(int):
        section(int):
        temp_now(float):
        mode(int):
        material_output(dict):
        vacuum_pumping_results(dict | None):

    Returns:
        tuple[float, float]:
            hw1 (float): 壁-層伝熱係数 [W/m2/K]
            u1 (float): 層伝熱係数 [W/m2/K
    """
    tower = state_manager.towers[tower_num]
    T_K = temp_now + 273.15

    if mode == 0:  # 吸着
        mf_co2 = material_output["inflow_mf_co2"]
        mf_n2 = material_output["inflow_mf_n2"]
    elif mode == 2:  # 脱着
        mf_co2 = tower.mf_co2[stream - 1, section - 1]
        mf_n2 = tower.mf_n2[stream - 1, section - 1]
    else:  # その他はとりあえず吸着と同様
        mf_co2 = material_output["inflow_mf_co2"]
        mf_n2 = material_output["inflow_mf_n2"]

    # ---- 物性値 -----------------------------------------------------
    kf = compute_gas_k(T_K, mf_co2, mf_n2)  # 気体熱伝導率
    kp = sim_conds.packed_bed.thermal_conductivity

    epsilon = sim_conds.packed_bed.average_porosity
    epsilon_p = sim_conds.packed_bed.emissivity
    dp = sim_conds.packed_bed.average_particle_diameter
    Lbed = sim_conds.packed_bed.height
    num_sec = sim_conds.common.num_sections

    # ---- ke0 (放射補正) --------------------------------------------
    ke0 = _yagi_kunii_radiation(T_K, kf, kp, epsilon, epsilon_p, dp)

    # ---- 流量関係 ---------------------------------------------------
    # ストリーム換算直径 d1
    d1 = 2.0 * (stream_conds[stream].cross_section / math.pi) ** 0.5

    # NOTE: 気体粘度 μ, 比熱 cp は大気圧を仮定
    P_ATM = 0.101325e6
    mu = (
        CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * mf_co2
        + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * mf_n2
    )
    Pr = mu * 1000.0 * material_output["gas_cp"] / kf

    # 流入ガス体積流量 f0
    if mode == 0:
        f0 = (
            (material_output["inflow_fr_co2"] + material_output["inflow_fr_n2"])
            / 1e6
            / (sim_conds.common.calculation_step_time * 60.0)
        )
    elif mode == 2:  # 脱着時は排気ガス体積流量 [m3/s]
        f0 = vacuum_pumping_results["vacuum_rate_N"] / 60.0 * stream_conds[stream].area_fraction
    else:
        f0 = 0.0  # NOTE: この処理で正しいか確認

    vcol = f0 / stream_conds[stream].cross_section  # 空塔速度[m/s]
    nu = mu / material_output["gas_density"]  # 気体動粘度[m2/s]
    Rep = 1.0 if vcol == 0 else vcol * dp / nu  # 粒子レイノルズ数

    # ---- ke, habs, dlat, hw1_raw -----------------------------------
    ke, habs, dlat, hw1_raw = _axial_flow_correction(ke0, kf, dp, d1, Pr, Rep, epsilon, Lbed, num_sec)

    # ---- 壁-層伝熱係数補正 -----------------------------------------
    hw1 = hw1_raw * sim_conds.vessel.wall_to_bed_htc_correction_factor

    # ---- 層伝熱係数 u1 ---------------------------------------------
    u1 = 1.0 / (dlat / ke + 1.0 / habs)

    return hw1, u1
