from dataclasses import dataclass
from typing import Dict
import numpy as np
from config.sim_conditions import SimulationConditions
from common.enums import LidPosition
from .results import (
    HeatBalanceResults,
    MassBalanceResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
    MoleFractionResults,
    VacuumPumpingResult,
)


class CellAccessor:
    """セルアクセス用のヘルパークラス

    tower.cell(stream, section).temp のような自然な記法でアクセス可能
    
    内部的には0オリジンのインデックスを使用します。
    stream=0, section=0 が最初のセルです。
    """

    def __init__(self, tower_state: "TowerStateArrays", stream: int, section: int):
        self._tower_state = tower_state
        self._stream = stream
        self._section = section
        # 0オリジンをそのまま使用（内部処理は全て0オリジン）
        self._stream_idx = stream
        self._section_idx = section

    @property
    def temp(self) -> float:
        """温度を取得"""
        return self._tower_state.temp[self._stream_idx, self._section_idx]

    @temp.setter
    def temp(self, value: float) -> None:
        """温度を設定"""
        self._tower_state.temp[self._stream_idx, self._section_idx] = value

    @property
    def loading(self) -> float:
        """吸着量を取得"""
        return self._tower_state.loading[self._stream_idx, self._section_idx]

    @loading.setter
    def loading(self, value: float) -> None:
        """吸着量を設定"""
        self._tower_state.loading[self._stream_idx, self._section_idx] = value

    @property
    def co2_mole_fraction(self) -> float:
        """CO2モル分率を取得"""
        return self._tower_state.co2_mole_fraction[self._stream_idx, self._section_idx]

    @co2_mole_fraction.setter
    def co2_mole_fraction(self, value: float) -> None:
        """CO2モル分率を設定"""
        self._tower_state.co2_mole_fraction[self._stream_idx, self._section_idx] = value

    @property
    def n2_mole_fraction(self) -> float:
        """N2モル分率を取得"""
        return self._tower_state.n2_mole_fraction[self._stream_idx, self._section_idx]

    @n2_mole_fraction.setter
    def n2_mole_fraction(self, value: float) -> None:
        """N2モル分率を設定"""
        self._tower_state.n2_mole_fraction[self._stream_idx, self._section_idx] = value

    @property
    def thermocouple_temperature(self) -> float:
        """熱電対温度を取得"""
        return self._tower_state.thermocouple_temperature[self._stream_idx, self._section_idx]

    @thermocouple_temperature.setter
    def thermocouple_temperature(self, value: float) -> None:
        """熱電対温度を設定"""
        self._tower_state.thermocouple_temperature[self._stream_idx, self._section_idx] = value

    @property
    def outlet_co2_partial_pressure(self) -> float:
        """流出CO2分圧を取得"""
        return self._tower_state.outlet_co2_partial_pressure[self._stream_idx, self._section_idx]

    @outlet_co2_partial_pressure.setter
    def outlet_co2_partial_pressure(self, value: float) -> None:
        """流出CO2分圧を設定"""
        self._tower_state.outlet_co2_partial_pressure[self._stream_idx, self._section_idx] = value

    @property
    def wall_to_bed_heat_transfer_coef(self) -> float:
        """壁-層伝熱係数を取得"""
        return self._tower_state.wall_to_bed_heat_transfer_coef[self._stream_idx, self._section_idx]

    @wall_to_bed_heat_transfer_coef.setter
    def wall_to_bed_heat_transfer_coef(self, value: float) -> None:
        """壁-層伝熱係数を設定"""
        self._tower_state.wall_to_bed_heat_transfer_coef[self._stream_idx, self._section_idx] = value

    @property
    def bed_heat_transfer_coef(self) -> float:
        """層伝熱係数を取得"""
        return self._tower_state.bed_heat_transfer_coef[self._stream_idx, self._section_idx]

    @bed_heat_transfer_coef.setter
    def bed_heat_transfer_coef(self, value: float) -> None:
        """層伝熱係数を設定"""
        self._tower_state.bed_heat_transfer_coef[self._stream_idx, self._section_idx] = value

    @property
    def previous_loading(self) -> float:
        """前タイムステップの吸着量を取得"""
        return self._tower_state.previous_loading[self._stream_idx, self._section_idx]

    @previous_loading.setter
    def previous_loading(self, value: float) -> None:
        """前タイムステップの吸着量を設定"""
        self._tower_state.previous_loading[self._stream_idx, self._section_idx] = value


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
    top_temperature: float  # 上蓋温度
    bottom_temperature: float  # 下蓋温度
    total_press: float  # 全圧
    cumulative_co2_recovered: float  # 積算CO2回収量[Nm3]
    cumulative_n2_recovered: float  # 積算N2回収量[Nm3]

    def cell(self, stream: int, section: int) -> CellAccessor:
        """指定されたstream, sectionのセルアクセッサーを取得

        Usage:
            tower.cell(0, 0).temp  # stream=0, section=0の温度を取得
            tower.cell(1, 2).loading = 0.5  # stream=1, section=2の吸着量を設定

        Args:
            stream: ストリームインデックス (0オリジン)
            section: セクションインデックス (0オリジン)

        Returns:
            CellAccessor: セルアクセッサー
        """
        return CellAccessor(self, stream, section)


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
        mf_co2_init = tower_cond.packed_bed.initial_co2_mole_fraction
        mf_n2_init = tower_cond.packed_bed.initial_n2_mole_fraction
        mf_co2_2d = np.full((self.num_streams, self.num_sections), mf_co2_init, dtype=np.float64)
        mf_n2_2d = np.full((self.num_streams, self.num_sections), mf_n2_init, dtype=np.float64)

        # 吸着量の初期化
        initial_loading = tower_cond.packed_bed.initial_adsorption_amount
        loading_2d = np.full((self.num_streams, self.num_sections), initial_loading, dtype=np.float64)

        # 伝熱係数の初期化
        heat_coef_2d = np.full((self.num_streams, self.num_sections), 1e-5, dtype=np.float64)
        heat_coef_wall_2d = np.full((self.num_streams, self.num_sections), 14.0, dtype=np.float64)

        # 流出CO2分圧の初期化
        outflow_pco2_2d = np.full(
            (self.num_streams, self.num_sections), tower_cond.packed_bed.initial_co2_partial_pressure, dtype=np.float64
        )

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
            top_temperature=ambient_temperature,
            bottom_temperature=ambient_temperature,
            total_press=tower_cond.packed_bed.initial_internal_pressure,
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

        # 各セル(stream, section)の結果を状態変数に反映（0オリジン）
        for stream in range(self.num_streams):
            for section in range(self.num_sections):
                # マテリアルバランス結果の更新
                material_result = material_results.get_result(stream, section)
                tower.cell(stream, section).loading = material_result.adsorption_state.updated_loading
                tower.cell(stream, section).outlet_co2_partial_pressure = (
                    material_result.pressure_state.outlet_co2_partial_pressure
                )

                # 熱バランス結果の更新
                heat_result = heat_results.get_result(stream, section)
                tower.cell(stream, section).temp = heat_result.cell_temperatures.bed_temperature
                tower.cell(stream, section).thermocouple_temperature = (
                    heat_result.cell_temperatures.thermocouple_temperature
                )
                tower.cell(stream, section).wall_to_bed_heat_transfer_coef = (
                    heat_result.heat_transfer_coefficients.wall_to_bed
                )
                tower.cell(stream, section).bed_heat_transfer_coef = heat_result.heat_transfer_coefficients.bed_to_bed

        # 壁面温度の更新（0オリジン）
        heat_wall_results: Dict[int, WallHeatBalanceResult] = calc_output.heat_wall  # Dict[int, WallHeatBalanceResult]
        tower.temp_wall[:] = np.array(
            [heat_wall_results[section].temperature for section in range(self.num_sections)],
            dtype=np.float64,
        )

        # 蓋温度の更新
        heat_lid_results: Dict[LidPosition, LidHeatBalanceResult] = calc_output.heat_lid
        tower.top_temperature = heat_lid_results[LidPosition.TOP].temperature
        tower.bottom_temperature = heat_lid_results[LidPosition.BOTTOM].temperature

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
            for stream in range(self.num_streams):
                for section in range(self.num_sections):
                    material_result = material_results.get_result(stream, section)
                    tower.cell(stream, section).co2_mole_fraction = material_result.outlet_gas.co2_mole_fraction
                    tower.cell(stream, section).n2_mole_fraction = material_result.outlet_gas.n2_mole_fraction
        elif mode == "真空脱着":
            mol_frac_results: MoleFractionResults = calc_output.mol_fraction  # MoleFractionResults
            if mol_frac_results:  # mol_fractionがNoneでない場合のみ処理
                for stream in range(self.num_streams):
                    for section in range(self.num_sections):
                        mol_frac_result = mol_frac_results.get_result(stream, section)
                        tower.cell(stream, section).co2_mole_fraction = (
                            mol_frac_result.co2_mole_fraction_after_desorption
                        )
                        tower.cell(stream, section).n2_mole_fraction = mol_frac_result.n2_mole_fraction_after_desorption

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
