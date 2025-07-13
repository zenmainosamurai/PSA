import numpy as np
from dataclasses import dataclass
from typing import Dict
from config.sim_conditions import SimulationConditions
from .results import (
    HeatBalanceResults,
    MassBalanceResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    MoleFractionResults,
    VacuumPumpingResult,
)


@dataclass
class TowerStateArrays:
    """各塔の状態変数を効率的に保持するためのデータクラス"""

    # 2D arrays (stream x section)
    temp: np.ndarray  # 温度
    thermocouple_temperature: np.ndarray  # 熱電対温度
    loading: np.ndarray  # 吸着量
    previous_loading: np.ndarray  # 前タイムステップの吸着量
    co2_mole_fraction: np.ndarray  # CO2モル分率
    n2_mole_fraction: np.ndarray  # N2モル分率
    wall_to_bed_heat_transfer_coef: np.ndarray  # 層伝熱係数
    bed_heat_transfer_coef: np.ndarray  # 壁-層伝熱係数
    outlet_co2_partial_pressure: np.ndarray  # 流出CO2分圧

    # 1D arrays (section only)
    temp_wall: np.ndarray  # 壁面温度

    # Scalars
    lid_temperature: float  # 上蓋温度
    bottom_temperature: float  # 下蓋温度
    total_press: float  # 全圧
    cumulative_co2_recovered: float  # 積算CO2回収量[Nm3]
    cumulative_n2_recovered: float  # 積算N2回収量[Nm3]


class StateVariables:
    """最適化された状態変数管理クラス"""

    def __init__(self, num_towers: int, num_streams: int, num_sections: int, sim_conds: SimulationConditions):
        self.num_towers = num_towers
        self.num_streams = num_streams
        self.num_sections = num_sections
        self.sim_conds = sim_conds

        # 各塔の状態変数を初期化
        self.towers: Dict[int, TowerStateArrays] = {}
        for tower_num in range(1, num_towers + 1):
            self.towers[tower_num] = self._init_tower_state(tower_num)

    def _init_tower_state(self, tower_num: int) -> TowerStateArrays:
        """各塔の状態変数を初期化"""
        tower_cond = self.sim_conds.get_tower(tower_num)

        ambient_temperature = tower_cond.vessel.ambient_temperature
        packed_bed_initial_temperature = tower_cond.packed_bed.initial_temperature

        # 2D配列の初期化
        temp_2d = np.full((self.num_streams, self.num_sections), packed_bed_initial_temperature, dtype=np.float64)

        # モル分率の初期化
        mf_co2_init = tower_cond.feed_gas.co2_mole_fraction
        mf_n2_init = tower_cond.feed_gas.n2_mole_fraction
        mf_co2_2d = np.full((self.num_streams, self.num_sections), mf_co2_init, dtype=np.float64)
        mf_n2_2d = np.full((self.num_streams, self.num_sections), mf_n2_init, dtype=np.float64)

        # 吸着量の初期化
        initial_loading = tower_cond.packed_bed.initial_adsorption_amount
        loading_2d = np.full((self.num_streams, self.num_sections), initial_loading, dtype=np.float64)

        # 伝熱係数の初期化
        heat_coef_2d = np.full((self.num_streams, self.num_sections), 1e-5, dtype=np.float64)
        heat_coef_wall_2d = np.full((self.num_streams, self.num_sections), 14.0, dtype=np.float64)

        # 流出CO2分圧の初期化
        total_press_init = tower_cond.feed_gas.total_pressure
        outflow_pco2_2d = np.full((self.num_streams, self.num_sections), total_press_init, dtype=np.float64)

        # 1D配列の初期化
        temp_wall_1d = np.full(self.num_sections, ambient_temperature, dtype=np.float64)

        return TowerStateArrays(
            temp=temp_2d.copy(),
            thermocouple_temperature=temp_2d.copy(),
            loading=loading_2d.copy(),
            previous_loading=loading_2d.copy(),
            co2_mole_fraction=mf_co2_2d.copy(),
            n2_mole_fraction=mf_n2_2d.copy(),
            wall_to_bed_heat_transfer_coef=heat_coef_2d.copy(),
            bed_heat_transfer_coef=heat_coef_wall_2d.copy(),
            outlet_co2_partial_pressure=outflow_pco2_2d.copy(),
            temp_wall=temp_wall_1d.copy(),
            lid_temperature=ambient_temperature,
            bottom_temperature=ambient_temperature,
            total_press=total_press_init,
            cumulative_co2_recovered=0.0,
            cumulative_n2_recovered=0.0,
        )

    def get_tower(self, tower_num: int) -> TowerStateArrays:
        """指定された塔の状態変数を取得"""
        return self.towers[tower_num]

    def update_from_calc_output(self, tower_num: int, mode: str, calc_output):
        """計算結果から状態変数を効率的に更新"""
        tower = self.towers[tower_num]

        # 前のタイムステップの吸着量
        tower.previous_loading[:] = tower.loading[:]

        heat_results: HeatBalanceResults = calc_output.heat  # HeatBalanceResults
        material_results: MassBalanceResults = calc_output.material  # MassBalanceResults

        # 各セル(stream, section)の結果を状態変数に反映
        for stream in range(1, self.num_streams + 1):
            for section in range(1, self.num_sections + 1):
                # マテリアルバランス結果の更新
                material_result = material_results.get_result(stream, section)
                tower.loading[stream - 1, section - 1] = material_result.adsorption_state.updated_loading
                tower.outlet_co2_partial_pressure[stream - 1, section - 1] = (
                    material_result.pressure_state.outlet_co2_partial_pressure
                )

                # 熱バランス結果の更新
                heat_result = heat_results.get_result(stream, section)
                tower.temp[stream - 1, section - 1] = heat_result.cell_temperatures.bed_temperature
                tower.thermocouple_temperature[stream - 1, section - 1] = (
                    heat_result.cell_temperatures.thermocouple_temperature
                )
                tower.wall_to_bed_heat_transfer_coef[stream - 1, section - 1] = (
                    heat_result.heat_transfer_coefficients.wall_to_bed
                )
                tower.bed_heat_transfer_coef[stream - 1, section - 1] = (
                    heat_result.heat_transfer_coefficients.bed_to_bed
                )

        # 壁面温度の更新
        heat_wall_results: Dict[int, WallHeatBalanceResult] = calc_output.heat_wall  # Dict[int, WallHeatBalanceResult]
        tower.temp_wall[:] = np.array(
            [heat_wall_results[section].temperature for section in range(1, self.num_sections + 1)],
            dtype=np.float64,
        )

        # 蓋温度の更新
        heat_lid_results: Dict[str, LidHeatBalanceResult] = calc_output.heat_lid  # Dict[str, LidHeatBalanceResult]
        tower.lid_temperature = heat_lid_results["up"].temperature
        tower.bottom_temperature = heat_lid_results["down"].temperature

        # モル分率の更新
        if mode == "停止":
            pass
        elif mode in [
            "初回ガス導入",
            "流通吸着_単独/上流",
            "バッチ吸着_上流",
            "バッチ吸着_上流（圧調弁あり）",
            "均圧_加圧",
            "均圧_減圧",
            "バッチ吸着_下流",
            "流通吸着_下流（圧調弁あり）",
            "流通吸着_下流",
        ]:
            for stream in range(1, self.num_streams + 1):
                for section in range(1, self.num_sections + 1):
                    material_result = material_results.get_result(stream, section)
                    tower.co2_mole_fraction[stream - 1, section - 1] = material_result.outlet_gas.co2_mole_fraction
                    tower.n2_mole_fraction[stream - 1, section - 1] = material_result.outlet_gas.n2_mole_fraction
        elif mode == "真空脱着":
            mol_frac_results: MoleFractionResults = calc_output.mol_fraction  # MoleFractionResults
            if mol_frac_results:  # mol_fractionがNoneでない場合のみ処理
                for stream in range(1, self.num_streams + 1):
                    for section in range(1, self.num_sections + 1):
                        mol_frac_result = mol_frac_results.get_result(stream, section)
                        tower.co2_mole_fraction[stream - 1, section - 1] = (
                            mol_frac_result.co2_mole_fraction_after_desorption
                        )
                        tower.n2_mole_fraction[stream - 1, section - 1] = (
                            mol_frac_result.n2_mole_fraction_after_desorption
                        )

        # 全圧の更新
        if mode == "停止":
            pass
        elif mode in [
            "初回ガス導入",
            "バッチ吸着_上流",
            "バッチ吸着_下流（圧調弁あり）",
            "均圧_加圧",
            "バッチ吸着_下流",
        ]:
            tower.total_press = calc_output.pressure_after_batch_adsorption
        elif mode in ["均圧_減圧", "流通吸着_単独/上流", "流通吸着_下流", "バッチ吸着_上流（圧調弁あり）"]:
            tower.total_press = calc_output.total_pressure
        elif mode == "真空脱着":
            tower.total_press = calc_output.pressure_after_vacuum_desorption

        # 回収量の更新
        if mode == "真空脱着":
            accum_vacuum_amt: VacuumPumpingResult = calc_output.accum_vacuum_amt
            tower.cumulative_co2_recovered = accum_vacuum_amt.cumulative_co2_recovered
            tower.cumulative_n2_recovered = accum_vacuum_amt.cumulative_n2_recovered
        else:
            tower.cumulative_co2_recovered = 0.0
            tower.cumulative_n2_recovered = 0.0
