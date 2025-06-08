from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import math


@dataclass
class CommonConditions:
    """共通条件"""

    calculation_step_time: float
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

    diameter: float
    radius: float
    cross_section: float
    height: float
    volume: float
    adsorbent_mass: float
    adsorbent_bulk_density: float
    thermal_conductivity: float
    emissivity: float
    specific_heat_capacity: float
    heat_capacity: float
    average_porosity: float
    average_particle_diameter: float
    particle_shape_factor: float
    initial_internal_pressure: float
    adsorption_mass_transfer_coef: float
    desorption_mass_transfer_coef: float
    void_volume: float
    upstream_piping_volume: float
    vessel_internal_void_volume: float
    initial_adsorption_amount: float

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class FeedGasConditions:
    """導入ガス条件"""

    co2_molecular_weight: float
    co2_flow_rate: float
    n2_molecular_weight: float
    n2_flow_rate: float
    total_flow_rate: float
    total_pressure: float
    temperature: float
    co2_mole_fraction: float
    n2_mole_fraction: float
    co2_density: float
    n2_density: float
    average_density: float
    co2_thermal_conductivity: float
    n2_thermal_conductivity: float
    average_thermal_conductivity: float
    co2_viscosity: float
    n2_viscosity: float
    average_viscosity: float
    enthalpy: float
    co2_specific_heat_capacity: float
    n2_specific_heat_capacity: float
    average_specific_heat_capacity: float
    heat_capacity_per_hour: float
    co2_adsorption_heat: float
    n2_adsorption_heat: float

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class VesselConditions:
    """容器条件"""

    diameter: float
    radius: float
    height: float
    wall_thickness: float
    wall_cross_section: float
    wall_volume: float
    wall_density: float
    wall_total_weight: float
    wall_specific_heat_capacity: float
    wall_thermal_conductivity: float
    lateral_surface_area: float
    external_heat_transfer_coef: float
    ambient_temperature: float
    wall_to_bed_htc_correction_factor: float

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class EndCoverConditions:
    """終端カバー条件"""

    flange_diameter: float
    flange_thickness: float
    outer_flange_inner_diameter: float
    outer_flange_area: float
    outer_flange_volume: float
    inner_flange_inner_diameter: float
    inner_flange_volume: float
    flange_total_weight: float

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class PipingConditions:
    """配管条件"""

    length: float
    diameter: float
    cross_section: float
    volume: float

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


@dataclass
class EqualizingPipingConditions(PipingConditions):
    """均圧配管条件"""

    flow_velocity_correction_factor: float
    main_part_volume: float
    isolated_equalizing_volume: float


@dataclass
class VacuumPipingConditions(PipingConditions):
    """真空引き配管条件"""

    space_volume: float


@dataclass
class ThermocoupleConditions:
    """熱電対条件"""

    specific_heat: float
    weight: float
    heat_transfer_correction_factor: float

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, float(getattr(self, field_name)))


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


class SimulationConditions:
    """シミュレーション条件全体を管理するクラス"""

    def __init__(self, cond_id: str):
        self.cond_id = cond_id
        self.towers: Dict[int, TowerConditions] = {}
        self.stream_conditions: Dict[int, Dict[int, StreamConditions]] = {}
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
            self.towers[tower_num] = self._create_tower_conditions(sheets, tower_num)
            self._initialize_stream_conditions(tower_num)

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

    def _initialize_stream_conditions(self, tower_num: int):
        """ストリーム条件を初期化"""
        tower = self.towers[tower_num]
        self.stream_conditions[tower_num] = {}

        # 各ストリームの条件を計算
        for stream in range(1, tower.common.num_streams + 1):
            self.stream_conditions[tower_num][stream] = self._create_stream_conditions(tower, stream)

        # 壁面条件を追加
        self.stream_conditions[tower_num][tower.common.num_streams + 1] = self._create_wall_conditions(tower)

    def _create_stream_conditions(self, tower: TowerConditions, stream: int) -> StreamConditions:
        """ストリーム条件を作成"""
        dr = tower.packed_bed.radius / tower.common.num_streams
        inner_radius = (stream - 1) * dr
        outer_radius = stream * dr
        cross_section = math.pi * (outer_radius**2 - inner_radius**2)

        return StreamConditions(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            cross_section=cross_section,
            area_fraction=cross_section / tower.packed_bed.cross_section,
            innter_perimeter=2 * math.pi * inner_radius,
            inner_boundary_area=2 * math.pi * inner_radius * tower.packed_bed.height,
            outer_perimeter=2 * math.pi * outer_radius,
            outer_boundary_area=2 * math.pi * outer_radius * tower.packed_bed.height,
            adsorbent_mass=tower.packed_bed.adsorbent_mass * (cross_section / tower.packed_bed.cross_section),
        )

    def _create_wall_conditions(self, tower: TowerConditions) -> StreamConditions:
        """壁面（最外ストリーム）条件を作成"""
        second_outermost_stream = self.stream_conditions[1][tower.common.num_streams]
        inner_radius = second_outermost_stream.outer_radius
        outer_radius = tower.vessel.radius
        cross_section = math.pi * (outer_radius**2 - inner_radius**2)

        return StreamConditions(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            cross_section=cross_section,
            area_fraction=0,  # NOTE: inner_radius, outer_radiusの参照方法から最外ストリームのことだと思ったが違う？
            innter_perimeter=2 * math.pi * inner_radius,
            inner_boundary_area=2 * math.pi * inner_radius * tower.packed_bed.height,
            outer_perimeter=2 * math.pi * outer_radius,
            outer_boundary_area=2 * math.pi * outer_radius * tower.packed_bed.height,
            adsorbent_mass=0,  # 壁面には吸着材なし
            wall_weight=tower.vessel.wall_total_weight,
        )

    def get_tower(self, tower_num: int) -> TowerConditions:
        """指定した塔の条件を取得"""
        return self.towers[tower_num]

    def get_stream(self, tower_num: int, stream: int) -> StreamConditions:
        """指定した塔・ストリームの条件を取得"""
        return self.stream_conditions[tower_num][stream]

    @property
    def num_towers(self) -> int:
        """塔数を取得"""
        return len(self.towers)
