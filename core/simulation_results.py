"""シミュレーション結果を管理するデータクラス"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from .state import (
    MassBalanceResults,
    HeatBalanceResults,
    WallHeatBalanceResult,
    LidHeatBalanceResult,
)


@dataclass
class TimeSeriesData:
    """時系列データを格納するクラス"""

    timestamps: List[float] = field(default_factory=list)
    material: List[MassBalanceResults] = field(default_factory=list)
    heat: List[HeatBalanceResults] = field(default_factory=list)
    heat_wall: List[Dict[int, WallHeatBalanceResult]] = field(default_factory=list)
    heat_lid: List[Dict[str, LidHeatBalanceResult]] = field(default_factory=list)
    others: List[Dict[str, Any]] = field(default_factory=list)

    def append_record(
        self,
        timestamp: float,
        material: MassBalanceResults,
        heat: HeatBalanceResults,
        heat_wall: Dict[int, WallHeatBalanceResult],
        heat_lid: Dict[str, LidHeatBalanceResult],
        others: Dict[str, Any],
    ):
        """1つの計算結果を時系列データに追加"""
        self.timestamps.append(timestamp)
        self.material.append(material)
        self.heat.append(heat)
        self.heat_wall.append(heat_wall)
        self.heat_lid.append(heat_lid)
        self.others.append(others)


@dataclass
class TowerSimulationResults:
    """1つの塔のシミュレーション結果を管理するクラス"""

    tower_id: int
    time_series_data: TimeSeriesData = field(default_factory=TimeSeriesData)

    def add_calculation_result(
        self,
        timestamp: float,
        material: MassBalanceResults,
        heat: HeatBalanceResults,
        heat_wall: Dict[int, WallHeatBalanceResult],
        heat_lid: Dict[str, LidHeatBalanceResult],
        others: Dict[str, Any],
    ):
        """計算結果を追加"""
        self.time_series_data.append_record(timestamp, material, heat, heat_wall, heat_lid, others)


@dataclass
class SimulationResults:
    """全体のシミュレーション結果を管理するクラス"""

    tower_simulation_results: Dict[int, TowerSimulationResults] = field(default_factory=dict)

    def initialize_tower(self, tower_id: int):
        """塔の結果管理を初期化"""
        if tower_id not in self.tower_simulation_results:
            self.tower_simulation_results[tower_id] = TowerSimulationResults(tower_id=tower_id)

    def add_tower_result(
        self,
        tower_id: int,
        timestamp: float,
        material: MassBalanceResults,
        heat: HeatBalanceResults,
        heat_wall: Dict[int, WallHeatBalanceResult],
        heat_lid: Dict[str, LidHeatBalanceResult],
        others: Dict[str, Any],
    ):
        """塔の計算結果を追加"""
        if tower_id not in self.tower_simulation_results:
            self.initialize_tower(tower_id)

        self.tower_simulation_results[tower_id].add_calculation_result(
            timestamp, material, heat, heat_wall, heat_lid, others
        )
