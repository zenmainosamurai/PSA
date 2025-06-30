import numpy as np
from typing import Tuple, Dict, Optional
from numba import jit


class OptimizedCalculations:
    """NumPy配列を使用した高速化された計算関数群"""

    @staticmethod
    @jit(nopython=True)
    def calculate_equilibrium_adsorption_amount_vectorized(P_array: np.ndarray, T_array: np.ndarray) -> np.ndarray:
        """平衡吸着量をベクトル化して計算（JITコンパイル対応）

        Args:
            P_array: 圧力配列 [kPaA]
            T_array: 温度配列 [K]

        Returns:
            平衡吸着量配列
        """
        return (
            P_array
            * (252.0724 - 0.50989705 * T_array)
            / (P_array - 3554.54819062669 * (1 - 0.0655247236249063 * np.sqrt(T_array)) ** 3 + 1.7354268)
        )

    @staticmethod
    def calculate_mean_properties_fast(state_arrays: "TowerStateArrays") -> Dict[str, float]:
        """塔全体の平均プロパティを高速計算"""
        return {
            "mean_temp": np.mean(state_arrays.temp) + 273.15,  # [K]
            "mean_mf_co2": np.mean(state_arrays.mf_co2),
            "mean_mf_n2": np.mean(state_arrays.mf_n2),
            "mean_temp_celsius": np.mean(state_arrays.temp),  # [℃]
        }

    @staticmethod
    def calculate_pressure_after_vacuum_pumping_optimized(
        sim_conds: Dict, state_manager: "StateVariables", tower_num: int
    ) -> Dict:
        """最適化された排気後圧力計算"""

        tower = state_manager.towers[tower_num]

        # 平均値の高速計算
        mean_props = OptimizedCalculations.calculate_mean_properties_fast(tower)
        T_K = mean_props["mean_temp"]
        average_co2_mole_fraction = mean_props["mean_mf_co2"]
        average_n2_mole_fraction = mean_props["mean_mf_n2"]

        # 全圧 [PaA]
        P = tower.total_press * 1e6

        # 粘度と密度の計算（既存のコードと同じ）
        P_ATM = 0.101325 * 1e6
        import CoolProp.CoolProp as CP

        viscosity = (
            CP.PropsSI("V", "T", T_K, "P", P_ATM, "co2") * average_co2_mole_fraction
            + CP.PropsSI("V", "T", T_K, "P", P_ATM, "nitrogen") * average_n2_mole_fraction
        )
        rho = (
            CP.PropsSI("D", "T", T_K, "P", P, "co2") * average_co2_mole_fraction
            + CP.PropsSI("D", "T", T_K, "P", P, "nitrogen") * average_n2_mole_fraction
        )

        # 圧損計算（既存のロジックをそのまま使用）
        _max_iteration = 1000
        pressure_loss = 0
        tolerance = 1e-6

        for iter in range(_max_iteration):
            pressure_loss_old = pressure_loss
            P_PUMP = (tower.total_press - pressure_loss) * 1e6
            P_PUMP = max(0, P_PUMP)

            vacuum_rate = 25 * (sim_conds["VACUUM_PIPING_COND"]["diameter"] ** 4) * P_PUMP / 2
            vacuum_rate_N = vacuum_rate / 0.1013 * P_PUMP * 1e-6
            linear_velocity = vacuum_rate / sim_conds["VACUUM_PIPING_COND"]["cross_section"]
            Re = (
                rho * linear_velocity * sim_conds["VACUUM_PIPING_COND"]["diameter"] / viscosity if viscosity != 0 else 0
            )
            lambda_f = 64 / Re if Re != 0 else 0
            pressure_loss = (
                lambda_f
                * sim_conds["VACUUM_PIPING_COND"]["length"]
                / sim_conds["VACUUM_PIPING_COND"]["diameter"]
                * linear_velocity**2
                / (2 * 9.81)
            ) * 1e-6

            if abs(pressure_loss - pressure_loss_old) < tolerance:
                break

        # CO2回収濃度計算
        vacuum_rate_mol = 101325 * vacuum_rate_N / 8.314 / T_K
        moles_pumped = vacuum_rate_mol * sim_conds["COMMON_COND"]["calculation_step_time"]
        vacuum_amt_co2 = moles_pumped * average_co2_mole_fraction
        vacuum_amt_n2 = moles_pumped * average_n2_mole_fraction

        total_co2_recovered = tower.vacuum_amt_co2 + vacuum_amt_co2
        total_n2_recovered = tower.vacuum_amt_n2 + vacuum_amt_n2
        co2_recovery_concentration = (total_co2_recovered / (total_co2_recovered + total_n2_recovered)) * 100

        # 排気後圧力計算
        case_inner_mol_amt = (
            (P_PUMP + pressure_loss * 1e6) * sim_conds["VACUUM_PIPING_COND"]["space_volume"] / 8.314 / T_K
        )
        remaining_moles = max(0, case_inner_mol_amt - moles_pumped)
        final_pressure = remaining_moles * 8.314 * T_K / sim_conds["VACUUM_PIPING_COND"]["space_volume"] * 1e-6

        return {
            "pressure_loss": pressure_loss,
            "total_co2_recovered": total_co2_recovered,
            "total_n2_recovered": total_n2_recovered,
            "co2_recovery_concentration": co2_recovery_concentration,
            "vacuum_rate_N": vacuum_rate_N,
            "remaining_moles": remaining_moles,
            "final_pressure": final_pressure,
        }

    @staticmethod
    def batch_update_from_material_output(
        tower: "TowerStateArrays", material_output: Dict, num_streams: int, num_sections: int
    ):
        """マテリアルバランス出力から状態変数を一括更新"""
        # 事前にNumPy配列を確保
        outlet_co2_mole_fraction = np.zeros((num_streams, num_sections))
        outlet_n2_mole_fraction = np.zeros((num_streams, num_sections))
        updated_loading = np.zeros((num_streams, num_sections))
        outlet_co2_partial_pressure = np.zeros((num_streams, num_sections))

        # 一括代入
        for stream in range(1, num_streams + 1):
            for section in range(1, num_sections + 1):
                idx_stream = stream - 1
                idx_section = section - 1

                outlet_co2_mole_fraction[idx_stream, idx_section] = material_output[stream][section][
                    "outlet_co2_mole_fraction"
                ]
                outlet_n2_mole_fraction[idx_stream, idx_section] = material_output[stream][section][
                    "outlet_n2_mole_fraction"
                ]
                updated_loading[idx_stream, idx_section] = material_output[stream][section]["updated_loading"]
                outlet_co2_partial_pressure[idx_stream, idx_section] = material_output[stream][section][
                    "outlet_co2_partial_pressure"
                ]

        # NumPy配列を直接更新
        tower.mf_co2[:] = outlet_co2_mole_fraction
        tower.mf_n2[:] = outlet_n2_mole_fraction
        tower.adsorp_amt[:] = updated_loading
        tower.outlet_co2_partial_pressure[:] = outlet_co2_partial_pressure


# 既存の関数をラップして最適化版を提供
def calculate_mass_balance_for_adsorption_optimized(
    sim_conds,
    stream_conds,
    stream,
    section,
    state_manager,
    tower_num,
    inflow_gas=None,
    flow_amt_depress=None,
    stagnant_mode=None,
):
    """最適化されたマテリアルバランス計算

    既存の関数をラップし、状態変数へのアクセスを最適化
    """
    # StateManagerから必要なデータを効率的に取得
    tower = state_manager.towers[tower_num]

    # レガシー形式の変数を作成（既存の関数との互換性のため）
    variables = {"temp": {}, "adsorp_amt": {}, "total_pressure": tower.total_press, "outlet_co2_partial_pressure": {}}

    # 必要な部分のみレガシー形式に変換
    for s in range(1, state_manager.num_streams + 1):
        variables["temp"][s] = {}
        variables["adsorp_amt"][s] = {}
        variables["outlet_co2_partial_pressure"][s] = {}
        for sec in range(1, state_manager.num_sections + 1):
            variables["temp"][s][sec] = tower.temp[s - 1, sec - 1]
            variables["adsorp_amt"][s][sec] = tower.adsorp_amt[s - 1, sec - 1]
            variables["outlet_co2_partial_pressure"][s][sec] = tower.outlet_co2_partial_pressure[s - 1, sec - 1]

    # 既存の関数を呼び出し
    import adsorption_base_models

    result = adsorption_base_models.calculate_mass_balance_for_adsorption(
        sim_conds, stream_conds, stream, section, variables, inflow_gas, flow_amt_depress, stagnant_mode
    )

    return result
