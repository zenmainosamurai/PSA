"""計算結果データクラス（簡素化版）

PSA担当者向け説明:
シミュレーション計算の結果を格納するデータクラスです。

主要なクラス:
- GasFlow: ガスの流量とモル分率
- MaterialBalanceResult: 1セルの物質収支結果
- HeatBalanceResult: 1セルの熱収支結果
- TowerResults: 1塔の全セル結果をまとめたもの
- OperationResult: 運転モード計算の結果

使用例:
    # セクション温度を取得
    temp = tower_results.get_temperature(stream=1, section=3)
    
    # 物質収支結果を取得
    mb = tower_results.get_mass_balance(stream=1, section=3)
    print(f"吸着量: {mb.updated_loading}")
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any


# ============================================================
# 基本データクラス（ガス流量）
# ============================================================

@dataclass
class GasFlow:
    """
    ガス流量
    
    Attributes:
        co2_volume: CO2体積 [cm3]
        n2_volume: N2体積 [cm3]
        co2_mole_fraction: CO2モル分率 [-]
        n2_mole_fraction: N2モル分率 [-]
    """
    co2_volume: float
    n2_volume: float
    co2_mole_fraction: float
    n2_mole_fraction: float


@dataclass
class GasProperties:
    """
    ガス物性
    
    Attributes:
        density: ガス密度 [kg/m3]
        specific_heat: ガス比熱 [kJ/kg/K]
    """
    density: float
    specific_heat: float


# ============================================================
# 物質収支結果
# ============================================================

@dataclass
class AdsorptionState:
    """
    吸着状態
    
    Attributes:
        equilibrium_loading: 平衡吸着量 [cm3/g-abs]
        actual_uptake_volume: 実際の吸着量 [cm3]
        updated_loading: 更新後の吸着量 [cm3/g-abs]
        theoretical_loading_delta: 理論新規吸着量 [cm3/g-abs]
    """
    equilibrium_loading: float
    actual_uptake_volume: float
    updated_loading: float
    theoretical_loading_delta: float


@dataclass
class PressureState:
    """
    圧力状態
    
    Attributes:
        co2_partial_pressure: CO2分圧 [MPa]
        outlet_co2_partial_pressure: 流出CO2分圧 [MPa]
    """
    co2_partial_pressure: float
    outlet_co2_partial_pressure: float


@dataclass
class MaterialBalanceResult:
    """
    1セルの物質収支結果
    
    PSA担当者向け説明:
    各セル（ストリーム×セクション）における物質収支計算の結果です。
    流入・流出ガス量、吸着量、CO2分圧などを含みます。
    
    Attributes:
        inlet_gas: 流入ガス情報
        outlet_gas: 流出ガス情報
        gas_properties: ガス物性
        adsorption_state: 吸着状態
        pressure_state: 圧力状態
    """
    inlet_gas: GasFlow
    outlet_gas: GasFlow
    gas_properties: GasProperties
    adsorption_state: AdsorptionState
    pressure_state: PressureState
    
    def to_dict(self) -> dict:
        """辞書形式に変換（CSV出力等で使用）"""
        return {
            "inlet_co2_volume": self.inlet_gas.co2_volume,
            "inlet_n2_volume": self.inlet_gas.n2_volume,
            "inlet_co2_mole_fraction": self.inlet_gas.co2_mole_fraction,
            "inlet_n2_mole_fraction": self.inlet_gas.n2_mole_fraction,
            "gas_density": self.gas_properties.density,
            "gas_specific_heat": self.gas_properties.specific_heat,
            "equilibrium_loading": self.adsorption_state.equilibrium_loading,
            "actual_uptake_volume": self.adsorption_state.actual_uptake_volume,
            "updated_loading": self.adsorption_state.updated_loading,
            "theoretical_loading_delta": self.adsorption_state.theoretical_loading_delta,
            "outlet_co2_volume": self.outlet_gas.co2_volume,
            "outlet_n2_volume": self.outlet_gas.n2_volume,
            "outlet_co2_mole_fraction": self.outlet_gas.co2_mole_fraction,
            "outlet_n2_mole_fraction": self.outlet_gas.n2_mole_fraction,
            "co2_partial_pressure": self.pressure_state.co2_partial_pressure,
            "outlet_co2_partial_pressure": self.pressure_state.outlet_co2_partial_pressure,
        }


# ============================================================
# 熱収支結果
# ============================================================

@dataclass
class CellTemperatures:
    """
    セル温度
    
    Attributes:
        bed_temperature: 層温度 [℃]
        thermocouple_temperature: 熱電対温度 [℃]
    """
    bed_temperature: float
    thermocouple_temperature: float


@dataclass
class HeatTransferCoefficients:
    """
    伝熱係数
    
    Attributes:
        wall_to_bed: 壁-層伝熱係数 [W/m2/K]
        bed_to_bed: 層間伝熱係数 [W/m2/K]
    """
    wall_to_bed: float
    bed_to_bed: float


@dataclass
class HeatFlux:
    """
    熱流束
    
    Attributes:
        adsorption: 吸着熱 [J]
        from_inner_boundary: 内側境界からの熱流束 [J]
        to_outer_boundary: 外側境界への熱流束 [J]
        downstream: 下流への熱流束 [J]
        upstream: 上流への熱流束 [J]
    """
    adsorption: float
    from_inner_boundary: float
    to_outer_boundary: float
    downstream: float
    upstream: float


@dataclass
class HeatBalanceResult:
    """
    1セルの熱収支結果
    
    PSA担当者向け説明:
    各セル（ストリーム×セクション）における熱収支計算の結果です。
    温度、伝熱係数、熱流束などを含みます。
    
    Attributes:
        cell_temperatures: セル温度
        heat_transfer_coefficients: 伝熱係数
        heat_flux: 熱流束
    """
    cell_temperatures: CellTemperatures
    heat_transfer_coefficients: HeatTransferCoefficients
    heat_flux: HeatFlux
    
    def to_dict(self) -> dict:
        """辞書形式に変換（CSV出力等で使用）"""
        return {
            "temp_reached": self.cell_temperatures.bed_temperature,
            "temp_thermocouple_reached": self.cell_temperatures.thermocouple_temperature,
            "wall_to_bed_heat_transfer_coef": self.heat_transfer_coefficients.wall_to_bed,
            "bed_heat_transfer_coef": self.heat_transfer_coefficients.bed_to_bed,
            "adsorption_heat": self.heat_flux.adsorption,
            "heat_flux_from_inner_boundary": self.heat_flux.from_inner_boundary,
            "heat_flux_to_outer_boundary": self.heat_flux.to_outer_boundary,
            "downstream_heat_flux": self.heat_flux.downstream,
            "upstream_heat_flux": self.heat_flux.upstream,
        }


# ============================================================
# 壁面・蓋の熱収支結果
# ============================================================

@dataclass
class WallHeatFlux:
    """
    壁面熱流束
    
    Attributes:
        from_inner_boundary: 内側境界からの熱流束 [J]
        to_outer_boundary: 外側境界への熱流束 [J]
        downstream: 下流への熱流束 [J]
        upstream: 上流への熱流束 [J]
    """
    from_inner_boundary: float
    to_outer_boundary: float
    downstream: float
    upstream: float


@dataclass
class WallHeatBalanceResult:
    """
    壁面の熱収支結果
    
    Attributes:
        temperature: 壁温度 [℃]
        heat_flux: 熱流束
    """
    temperature: float
    heat_flux: WallHeatFlux
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "temp_reached": self.temperature,
            "downstream_heat_flux": self.heat_flux.downstream,
            "upstream_heat_flux": self.heat_flux.upstream,
            "heat_flux_from_inner_boundary": self.heat_flux.from_inner_boundary,
            "heat_flux_to_outer_boundary": self.heat_flux.to_outer_boundary,
        }


@dataclass
class LidHeatBalanceResult:
    """
    蓋の熱収支結果
    
    Attributes:
        temperature: 蓋温度 [℃]
    """
    temperature: float


# ============================================================
# 真空脱着関連
# ============================================================

@dataclass
class VacuumPumpingResult:
    """
    真空排気計算結果
    
    PSA担当者向け説明:
    真空ポンプによる排気計算の結果です。
    CO2回収量、回収濃度、排気後圧力などを含みます。
    
    Attributes:
        pressure_loss: 圧力損失 [MPa]
        cumulative_co2_recovered: 累積CO2回収量 [Nm3]
        cumulative_n2_recovered: 累積N2回収量 [Nm3]
        co2_recovery_concentration: CO2回収濃度 [%]
        volumetric_flow_rate: 体積流量 [m3/min]
        remaining_moles: 残存モル量 [mol]
        final_pressure: 最終圧力 [MPa]
    """
    pressure_loss: float
    cumulative_co2_recovered: float
    cumulative_n2_recovered: float
    co2_recovery_concentration: float
    volumetric_flow_rate: float
    remaining_moles: float
    final_pressure: float


@dataclass
class DesorptionMoleFractionResult:
    """
    脱着時のモル分率計算結果
    
    Attributes:
        co2_mole_fraction_after_desorption: 脱着後CO2モル分率 [-]
        n2_mole_fraction_after_desorption: 脱着後N2モル分率 [-]
        total_moles_after_desorption: 脱着後総モル量 [mol]
    """
    co2_mole_fraction_after_desorption: float
    n2_mole_fraction_after_desorption: float
    total_moles_after_desorption: float


# ============================================================
# 均圧関連
# ============================================================

@dataclass
class DepressurizationResult:
    """
    減圧計算結果
    
    Attributes:
        final_pressure: 最終圧力 [MPa]
        flow_rate: 流量 [L/min]
        pressure_differential: 圧力差 [Pa]
    """
    final_pressure: float
    flow_rate: float
    pressure_differential: float


@dataclass
class DownstreamFlowResult:
    """
    下流側流量計算結果
    
    Attributes:
        final_pressure: 最終圧力 [MPa]
        outlet_flows: 各ストリームの流出ガス
    """
    final_pressure: float
    outlet_flows: Dict[int, GasFlow]


# ============================================================
# 塔全体の結果（簡素化版）
# ============================================================

@dataclass
class TowerResults:
    """
    1塔の計算結果
    
    PSA担当者向け説明:
    1つの塔における全セルの計算結果をまとめたクラスです。
    get_temperature(stream, section) のように直感的にアクセスできます。
    
    使用例:
        # セクション温度を取得
        temp = results.get_temperature(stream=1, section=3)
        
        # 物質収支結果を取得
        mb = results.get_mass_balance(stream=1, section=3)
        print(f"吸着量: {mb.adsorption_state.updated_loading}")
        
        # 壁温度を取得
        wall_temp = results.get_wall_temperature(section=3)
    """
    # (stream, section) をキーとする辞書
    _mass_balance: Dict[Tuple[int, int], MaterialBalanceResult] = field(default_factory=dict)
    _heat_balance: Dict[Tuple[int, int], HeatBalanceResult] = field(default_factory=dict)
    
    # section のみをキーとする辞書
    _wall_heat: Dict[int, WallHeatBalanceResult] = field(default_factory=dict)
    
    # "up" / "down" をキーとする辞書
    _lid_heat: Dict[str, LidHeatBalanceResult] = field(default_factory=dict)
    
    # --- 物質収支 ---
    def get_mass_balance(self, stream: int, section: int) -> MaterialBalanceResult:
        """物質収支結果を取得"""
        return self._mass_balance[(stream, section)]
    
    def set_mass_balance(self, stream: int, section: int, result: MaterialBalanceResult):
        """物質収支結果を設定"""
        self._mass_balance[(stream, section)] = result
    
    def has_mass_balance(self, stream: int, section: int) -> bool:
        """物質収支結果が存在するか確認"""
        return (stream, section) in self._mass_balance
    
    # --- 熱収支 ---
    def get_heat_balance(self, stream: int, section: int) -> HeatBalanceResult:
        """熱収支結果を取得"""
        return self._heat_balance[(stream, section)]
    
    def set_heat_balance(self, stream: int, section: int, result: HeatBalanceResult):
        """熱収支結果を設定"""
        self._heat_balance[(stream, section)] = result
    
    # --- 便利メソッド（よく使う値への直接アクセス） ---
    def get_temperature(self, stream: int, section: int) -> float:
        """セクション温度を取得 [℃]"""
        return self._heat_balance[(stream, section)].cell_temperatures.bed_temperature
    
    def get_thermocouple_temperature(self, stream: int, section: int) -> float:
        """熱電対温度を取得 [℃]"""
        return self._heat_balance[(stream, section)].cell_temperatures.thermocouple_temperature
    
    def get_loading(self, stream: int, section: int) -> float:
        """吸着量を取得 [cm3/g-abs]"""
        return self._mass_balance[(stream, section)].adsorption_state.updated_loading
    
    def get_outlet_co2_flow(self, stream: int, section: int) -> float:
        """流出CO2流量を取得 [cm3]"""
        return self._mass_balance[(stream, section)].outlet_gas.co2_volume
    
    def get_outlet_n2_flow(self, stream: int, section: int) -> float:
        """流出N2流量を取得 [cm3]"""
        return self._mass_balance[(stream, section)].outlet_gas.n2_volume
    
    def get_co2_partial_pressure(self, stream: int, section: int) -> float:
        """CO2分圧を取得 [MPa]"""
        return self._mass_balance[(stream, section)].pressure_state.co2_partial_pressure
    
    # --- 壁面 ---
    def get_wall_heat(self, section: int) -> WallHeatBalanceResult:
        """壁面熱収支を取得"""
        return self._wall_heat[section]
    
    def set_wall_heat(self, section: int, result: WallHeatBalanceResult):
        """壁面熱収支を設定"""
        self._wall_heat[section] = result
    
    def get_wall_temperature(self, section: int) -> float:
        """壁温度を取得 [℃]"""
        return self._wall_heat[section].temperature
    
    # --- 蓋 ---
    def get_lid_heat(self, position: str) -> LidHeatBalanceResult:
        """蓋熱収支を取得 (position: "up" or "down")"""
        return self._lid_heat[position]
    
    def set_lid_heat(self, position: str, result: LidHeatBalanceResult):
        """蓋熱収支を設定"""
        self._lid_heat[position] = result
    
    def get_lid_temperature(self, position: str) -> float:
        """蓋温度を取得 [℃] (position: "up" or "down")"""
        return self._lid_heat[position].temperature


# ============================================================
# 運転モード計算の統一結果
# ============================================================

@dataclass
class OperationResult:
    """
    運転モード計算結果（全モード共通）
    
    PSA担当者向け説明:
    各運転モード（流通吸着、バッチ吸着、真空脱着など）の計算結果を
    統一的に格納するクラスです。
    
    - 全モードで共通: tower_results（物質収支・熱収支）
    - 圧力変化があるモード: pressure_after に値が入る
    - 真空脱着モード: vacuum_pumping, mole_fraction に値が入る
    - 均圧減圧モード: downstream_flow に値が入る
    
    Attributes:
        tower_results: 塔の計算結果
        pressure_after: 計算後の圧力 [MPa]（該当モードのみ）
        pressure_diff: 圧力差 [Pa]（均圧減圧モードのみ）
        downstream_flow: 下流流量情報（均圧減圧モードのみ）
        vacuum_pumping: 真空排気結果（真空脱着モードのみ）
        mole_fraction: モル分率結果（真空脱着モードのみ）
    """
    # 全モード共通
    tower_results: TowerResults
    
    # 圧力関連（バッチ吸着、流通吸着、真空脱着、均圧で使用）
    pressure_after: Optional[float] = None
    
    # 均圧減圧モード用
    pressure_diff: Optional[float] = None
    downstream_flow: Optional[DownstreamFlowResult] = None
    
    # 真空脱着モード用
    vacuum_pumping: Optional[VacuumPumpingResult] = None
    mole_fraction: Optional[Dict[Tuple[int, int], DesorptionMoleFractionResult]] = None
    
    def get_record_items(self) -> Dict[str, Any]:
        """
        記録用の項目を取得（既存コードとの互換性のため）
        
        Returns:
            material, heat, heat_wall, heat_lid を含む辞書
        """
        return {
            "material": self.tower_results._mass_balance,
            "heat": self.tower_results._heat_balance,
            "heat_wall": self.tower_results._wall_heat,
            "heat_lid": self.tower_results._lid_heat,
        }


# ============================================================
# 物質収支計算の統一結果（Strategyパターン廃止後の型）
# ============================================================

@dataclass
class MassBalanceCalculationResult:
    """
    物質収支計算の結果
    
    Attributes:
        material_balance: 物質収支結果
        mole_fraction_data: モル分率データ（脱着モードのみ）
    """
    material_balance: MaterialBalanceResult
    mole_fraction_data: Optional[DesorptionMoleFractionResult] = None
    
    def has_mole_fraction_data(self) -> bool:
        """モル分率データを持っているか"""
        return self.mole_fraction_data is not None
