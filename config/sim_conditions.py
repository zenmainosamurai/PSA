from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import math


@dataclass
class CommonConditions:
    """共通条件"""

    calculation_step_time: float  # min
    num_streams: int
    num_sections: int

    def __post_init__(self):
        # 型変換と検証
        self.num_streams = int(self.num_streams)
        self.num_sections = int(self.num_sections)
        self.calculation_step_time = float(self.calculation_step_time)


@dataclass
class PackedBedConditions:
    """充填層条件"""

    diameter: float  # m
    radius: float  # m
    cross_section: float  # m^2
    height: float  # m
    volume: float  # m^3
    adsorbent_mass: float  # g
    adsorbent_bulk_density: float  # g/cm^3
    thermal_conductivity: float  # W/(m·K)
    emissivity: float  # -
    specific_heat_capacity: float  # J/(g·K)
    heat_capacity: float  # J/K
    average_porosity: float  # -
    average_particle_diameter: float  # m
    particle_shape_factor: float  # -
    initial_internal_pressure: float  # kPaA
    adsorption_mass_transfer_coef: float  # 1e-6/sec
    desorption_mass_transfer_coef: float  # 1e-6/sec
    void_volume: float  # m^3
    upstream_piping_volume: float  # m^3
    vessel_internal_void_volume: float  # m^3
    initial_adsorption_amount: float  # cm^3/g-abs
    initial_temperature: float  # degC

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class FeedGasConditions:
    """導入ガス条件"""

    co2_molecular_weight: float  # g/mol
    co2_flow_rate_normal: float  # Nm3/h
    co2_flow_rate: float  # L/min
    n2_molecular_weight: float  # g/mol
    n2_flow_rate_normal: float  # Nm3/h
    n2_flow_rate: float  # L/min
    total_flow_rate: float  # L/min
    total_pressure: float  # MPaA
    temperature: float  # degC
    co2_mole_fraction: float  # -
    n2_mole_fraction: float  # -
    co2_density: float  # kg/m^3
    n2_density: float  # kg/m^3
    average_density: float  # kg/m^3
    co2_thermal_conductivity: float  # W/(m·K)
    n2_thermal_conductivity: float  # W/(m·K)
    average_thermal_conductivity: float  # W/(m·K)
    co2_viscosity: float  # Pa·s
    n2_viscosity: float  # Pa·s
    average_viscosity: float  # Pa·s
    enthalpy: float  # kJ/kg
    co2_specific_heat_capacity: float  # kJ/(kg·K)
    n2_specific_heat_capacity: float  # kJ/(kg·K)
    average_specific_heat_capacity: float  # kJ/(kg·K)
    heat_capacity_per_hour: float  # kJ/K/h
    co2_adsorption_heat: float  # kJ/kg
    n2_adsorption_heat: float  # kJ/kg

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class VesselConditions:
    """容器条件"""

    diameter: float  # m
    radius: float  # m
    height: float  # m
    wall_thickness: float  # m
    wall_cross_section: float  # m^2
    wall_volume: float  # m^3
    wall_density: float  # g/cm^3
    wall_total_weight: float  # g
    wall_specific_heat_capacity: float  # J/(kg·K)
    wall_thermal_conductivity: float  # W/(m·K)
    lateral_surface_area: float  # m^2
    external_heat_transfer_coef: float  # W/(m^2·K)
    ambient_temperature: float  # degC
    wall_to_bed_htc_correction_factor: float  # -

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class EndCoverConditions:
    """終端カバー条件"""

    flange_diameter: float  # mm
    flange_thickness: float  # mm
    outer_flange_inner_diameter: float  # mm
    outer_flange_area: float  # m^2
    outer_flange_volume: float  # cm^3
    inner_flange_inner_diameter: float  # mm
    inner_flange_volume: float  # cm^3
    flange_total_weight: float  # g

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class PipingConditions:
    """配管条件"""

    length: float  # m
    diameter: float  # m
    cross_section: float  # m^2
    volume: float  # m^3

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class EqualizingPipingConditions(PipingConditions):
    """均圧配管条件"""

    flow_velocity_correction_factor: float  # -
    main_part_volume: float  # m^3
    isolated_equalizing_volume: float  # m^3


@dataclass
class VacuumPipingConditions(PipingConditions):
    """真空引き配管条件"""

    space_volume: float  # m^3
    vacuum_pumping_speed: float  # m^3/min


@dataclass
class ThermocoupleConditions:
    """熱電対条件"""

    specific_heat: float  # J/(g·K)
    weight: float  # g
    heat_transfer_correction_factor: float  # -

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class StreamConditions:
    """ストリーム条件"""

    inner_radius: float
    outer_radius: float
    cross_section: float
    area_fraction: float
    innter_perimeter: float
    inner_boundary_area: float
    outer_perimeter: float
    outer_boundary_area: float
    adsorbent_mass: float
    wall_weight: Optional[float] = field(default=None)


@dataclass
class TowerConditions:
    """塔ごとの条件"""

    common: CommonConditions
    packed_bed: PackedBedConditions
    feed_gas: FeedGasConditions
    vessel: VesselConditions
    lid: EndCoverConditions
    bottom: EndCoverConditions
    equalizing_piping: EqualizingPipingConditions
    vacuum_piping: VacuumPipingConditions
    thermocouple: ThermocoupleConditions

    stream_conditions: Dict[int, StreamConditions] = field(default_factory=dict)

    def initialize_stream_conditions(self):
        num_streams = self.common.num_streams
        dr = self.packed_bed.radius / num_streams
        for stream in range(1, num_streams + 1):
            inner_radius = (stream - 1) * dr
            outer_radius = stream * dr
            cross_section = math.pi * (outer_radius**2 - inner_radius**2)
            self.stream_conditions[stream] = StreamConditions(
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                cross_section=cross_section,
                area_fraction=cross_section / self.packed_bed.cross_section,
                innter_perimeter=2 * math.pi * inner_radius,
                inner_boundary_area=2 * math.pi * inner_radius * self.packed_bed.height,
                outer_perimeter=2 * math.pi * outer_radius,
                outer_boundary_area=2 * math.pi * outer_radius * self.packed_bed.height,
                adsorbent_mass=self.packed_bed.adsorbent_mass * (cross_section / self.packed_bed.cross_section),
            )
        # 壁面条件
        outermost_stream = self.stream_conditions[num_streams]
        inner_radius = outermost_stream.outer_radius
        outer_radius = self.vessel.radius
        cross_section = math.pi * (outer_radius**2 - inner_radius**2)
        self.stream_conditions[num_streams + 1] = StreamConditions(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            cross_section=cross_section,
            area_fraction=0,  # 壁面は面積分率なし
            innter_perimeter=2 * math.pi * inner_radius,
            inner_boundary_area=2 * math.pi * inner_radius * self.packed_bed.height,
            outer_perimeter=2 * math.pi * outer_radius,
            outer_boundary_area=2 * math.pi * outer_radius * self.packed_bed.height,
            adsorbent_mass=0,  # 壁面には吸着材なし
            wall_weight=self.vessel.wall_total_weight,
        )


class SimulationConditions:
    """シミュレーション条件全体を管理するクラス"""

    def __init__(self, cond_id: str):
        self.cond_id = cond_id
        self.towers: Dict[int, TowerConditions] = {}
        self._load_conditions()

    def _load_conditions(self):
        """Excelファイルから条件を読み込む"""
        filepath = f"conditions/{self.cond_id}/sim_conds.xlsx"

        sheets = pd.read_excel(
            filepath,
            sheet_name=[
                "共通",
                "触媒充填層条件",
                "導入ガス条件",
                "容器壁条件",
                "蓋条件",
                "底条件",
                "均圧配管条件",
                "真空引き配管条件",
                "熱電対条件",
            ],
            index_col=1,
        )

        # 各塔の条件を読み込み
        for tower_num in range(1, 4):
            tower = self._create_tower_conditions(sheets, tower_num)
            tower.initialize_stream_conditions()
            self.towers[tower_num] = tower

    def _create_tower_conditions(self, sheets: Dict[str, pd.DataFrame], tower_num: int) -> TowerConditions:
        """各塔の条件を作成"""
        col = f"塔{tower_num}"

        return TowerConditions(
            common=CommonConditions(**self._extract_params(sheets["共通"], col)),
            packed_bed=PackedBedConditions(**self._extract_params(sheets["触媒充填層条件"], col)),
            feed_gas=FeedGasConditions(**self._extract_params(sheets["導入ガス条件"], col)),
            vessel=VesselConditions(**self._extract_params(sheets["容器壁条件"], col)),
            lid=EndCoverConditions(**self._extract_params(sheets["蓋条件"], col)),
            bottom=EndCoverConditions(**self._extract_params(sheets["底条件"], col)),
            equalizing_piping=EqualizingPipingConditions(**self._extract_params(sheets["均圧配管条件"], col)),
            vacuum_piping=VacuumPipingConditions(**self._extract_params(sheets["真空引き配管条件"], col)),
            thermocouple=ThermocoupleConditions(**self._extract_params(sheets["熱電対条件"], col)),
        )

    def _extract_params(self, df: pd.DataFrame, col: str) -> Dict:
        """DataFrameから指定列のパラメータを辞書として抽出"""
        return {param: df.loc[param, col] for param in df.index if pd.notna(df.loc[param, col])}

    def get_tower(self, tower_num: int) -> TowerConditions:
        """指定した塔の条件を取得"""
        return self.towers[tower_num]

    @property
    def num_towers(self) -> int:
        """塔数を取得"""
        return len(self.towers)
