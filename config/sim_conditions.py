from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import math
import logger as log

logger = log.logger.getChild(__name__)


# sim_conds.xlsx記載の各条件
@dataclass
class CommonConditions:
    """共通条件"""

    calculation_step_time: float  # min
    num_streams: int
    num_sections: int
    use_xlsx: int = 0  # xlsxファイル出力フラグ（0: 無効, 1: 有効）
    sections_for_graph: str = ""  # グラフ用セクション番号（セミコロン区切り）

    def __post_init__(self):
        # 型変換と検証
        self.num_streams = int(self.num_streams)
        self.num_sections = int(self.num_sections)
        self.calculation_step_time = float(self.calculation_step_time)
        self.use_xlsx = int(self.use_xlsx)
        self.sections_for_graph = str(self.sections_for_graph)

    def get_sections_for_graph(self) -> list:
        """
        グラフ用セクション番号を整数のリストとして取得

        Returns:
            list: セクション番号のリスト（例: [3, 5, 8]）
        """
        if not self.sections_for_graph:
            return [2, 10, 18]

        try:
            sections = [int(section.strip()) for section in self.sections_for_graph.split(";") if section.strip()]
            return sections
        except (ValueError, AttributeError):
            logger.warning(
                f"グラフ用セクション番号の解析に失敗しました: {self.sections_for_graph}. デフォルト値を使用します。"
            )
            return [2, 10, 18]


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
    initial_internal_pressure: float  # MPaA
    adsorption_mass_transfer_coef: float  # 1e-6/sec
    desorption_mass_transfer_coef: float  # 1e-6/sec
    void_volume: float  # m^3
    upstream_piping_volume: float  # m^3
    vessel_internal_void_volume: float  # m^3
    initial_adsorption_amount: float  # cm^3/g-abs
    initial_temperature: float  # degC
    initial_co2_partial_pressure: float  # MPaA
    initial_n2_partial_pressure: float  # MPaA
    initial_co2_mole_fraction: float  # -
    initial_n2_mole_fraction: float  # -

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
    pipe_correction_factor: float  # -


@dataclass
class VacuumPipingConditions(PipingConditions):
    """真空引き配管条件"""

    space_volume: float  # m^3
    vacuum_pumping_speed: float  # m^3/min
    pump_correction_factor_1: float  # -
    pump_correction_factor_2: float  # -


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
    """1つの塔の全条件をまとめたクラス"""

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

    def _validate_sheet_existence(self, filepath: str, required_sheets: List[str]) -> bool:
        """
        必要なシートが存在するかチェック

        Args:
            filepath (str): Excelファイルのパス
            required_sheets (List[str]): 必要なシート名のリスト

        Returns:
            bool: 全てのシートが存在する場合True

        Raises:
            Exception: 必要なシートが不足している場合
        """
        try:
            excel_file = pd.ExcelFile(filepath)
            existing_sheets_set = set(excel_file.sheet_names)
            required_sheets_set = set(required_sheets)

            if required_sheets_set.issubset(existing_sheets_set):
                return True
            else:
                missing_sheets = required_sheets_set - existing_sheets_set
                error_msg = f"条件ファイル({filepath})に必要なシートが不足しています: {list(missing_sheets)}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except FileNotFoundError:
            error_msg = f"条件ファイルが見つかりません: {filepath}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"条件ファイルの読み込み中に予期せぬエラーが発生: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def _validate_dataclass_fields(
        self, dataclass_type, sheet_name: str, tower_col: str, extracted_params: Dict
    ) -> None:
        """
        データクラスのフィールドに対してバリデーションを実行

        Args:
            dataclass_type: バリデーション対象のデータクラス
            sheet_name (str): シート名
            tower_col (str): 塔カラム名
            extracted_params (Dict): 抽出されたパラメータ

        Raises:
            Exception: バリデーションエラーが発生した場合
        """
        expected_fields = set(dataclass_type.__dataclass_fields__.keys())
        actual_fields = set(extracted_params.keys())

        # 余分なフィールドのチェック
        extra_fields = actual_fields - expected_fields
        if extra_fields:
            error_msg = (
                f"シート'{sheet_name}'の{tower_col}に予期しない変数（指定外の変数）があります: {list(extra_fields)}"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        # 不足フィールドのチェック
        missing_fields = expected_fields - actual_fields
        if missing_fields:
            error_msg = f"シート'{sheet_name}'の{tower_col}に必要な変数が不足しています: {list(missing_fields)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # 型チェック
        for field_name, field_value in extracted_params.items():
            if field_name in expected_fields:
                field_info = dataclass_type.__dataclass_fields__[field_name]
                expected_type = field_info.type

                # デフォルト値がある場合はスキップ
                if field_info.default != field_info.default_factory and pd.isna(field_value):
                    continue

                # floatまたはintまたはstrの型チェック
                if expected_type == str:
                    # 文字列型の場合はそのまま通す
                    pass
                elif expected_type == float:
                    try:
                        float(field_value)
                    except (ValueError, TypeError):
                        error_msg = f"シート'{sheet_name}'の{tower_col}、パラメータ'{field_name}': {field_value} (数値である必要があります)"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                elif expected_type == int:
                    try:
                        int(field_value)
                    except (ValueError, TypeError):
                        error_msg = f"シート'{sheet_name}'の{tower_col}、パラメータ'{field_name}': {field_value} (整数である必要があります)"
                        logger.error(error_msg)
                        raise Exception(error_msg)

        logger.debug(f"シート'{sheet_name}'の{tower_col}のバリデーションが成功しました")

    def _validate_sheet_data(self, df: pd.DataFrame, sheet_name: str) -> None:
        """
        シートデータの基本バリデーション

        Args:
            df (pd.DataFrame): 検証するDataFrame
            sheet_name (str): シート名

        Raises:
            Exception: バリデーションエラーが発生した場合
        """
        # 塔1〜3の列が存在するかチェック
        required_columns = ["塔1", "塔2", "塔3"]
        missing_columns = []

        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)

        if missing_columns:
            error_msg = (
                f"シート'{sheet_name}'に必要な列が不足しています: {missing_columns} (既存の列: {list(df.columns)})"
            )
            logger.error(error_msg)
            raise Exception(error_msg)

        # 各塔の列について値の型チェック
        for col in required_columns:
            logger.debug(f"シート'{sheet_name}'の列'{col}'の値を検証中...")

            # 各行について値をチェック
            for param_name in df.index:
                value = df.loc[param_name, col]

                # NaN値はスキップ（後でデフォルト値チェックを行う）
                if pd.isna(value):
                    continue

                # sections_for_graph は文字列なので数値チェックをスキップ
                if param_name == "sections_for_graph":
                    continue

                # 数値型であることをチェック
                if not self._is_numeric_value(value):
                    error_msg = f"シート'{sheet_name}'の列'{col}'、パラメータ'{param_name}'の値が不正です: {repr(value)} (数値である必要があります)"
                    logger.error(error_msg)
                    raise Exception(error_msg)

            logger.debug(f"シート'{sheet_name}'の列'{col}'の型チェックが成功しました")

    def _is_numeric_value(self, value) -> bool:
        """
        値が数値として変換可能かチェック

        Args:
            value: チェック対象の値

        Returns:
            bool: 数値として変換可能な場合True
        """
        try:
            # まずfloatに変換を試行
            float(value)
            return True
        except (ValueError, TypeError):
            # 文字列の場合、数値文字列かチェック
            if isinstance(value, str):
                # 空白を除去してから再試行
                cleaned_value = value.strip()
                if cleaned_value == "":
                    return False
                try:
                    float(cleaned_value)
                    return True
                except (ValueError, TypeError):
                    return False
            return False

    def _validate_sheet_data_with_types(self, df: pd.DataFrame, sheet_name: str, dataclass_type) -> None:
        """
        データクラスの型情報を使用したシートデータの詳細バリデーション

        Args:
            df (pd.DataFrame): 検証するDataFrame
            sheet_name (str): シート名
            dataclass_type: データクラスの型

        Raises:
            Exception: バリデーションエラーが発生した場合
        """
        required_columns = ["塔1", "塔2", "塔3"]

        # データクラスのフィールド情報を取得
        expected_fields = dataclass_type.__dataclass_fields__

        # 指定外のパラメータ（行）をチェック
        actual_params = set(df.index)
        expected_params = set(expected_fields.keys())
        extra_params = actual_params - expected_params

        if extra_params:
            error_msg = f"シート'{sheet_name}'に予期しないパラメータ（指定外の行）があります: {list(extra_params)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        for col in required_columns:
            if col not in df.columns:
                continue

            logger.debug(f"シート'{sheet_name}'の列'{col}'について詳細な型チェックを実行中...")

            for param_name in df.index:
                if param_name not in expected_fields:
                    continue

                value = df.loc[param_name, col]
                field_info = expected_fields[param_name]
                expected_type = field_info.type

                # NaN値の場合、デフォルト値があるかチェック
                if pd.isna(value):
                    if field_info.default == field_info.default_factory:  # デフォルト値なし
                        error_msg = f"シート'{sheet_name}'の列'{col}'、パラメータ'{param_name}': 値が設定されていません (必須パラメータ)"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    continue

                if expected_type == str:
                    # 文字列型の場合はそのまま通す
                    pass
                elif expected_type == float:
                    try:
                        float_val = float(value)
                        # 特定の範囲チェック（必要に応じて）
                        if param_name.endswith("_fraction") and not (0 <= float_val <= 1):
                            error_msg = f"シート'{sheet_name}'の列'{col}'、パラメータ'{param_name}': {value} (分率は0〜1の範囲である必要があります)"
                            logger.error(error_msg)
                            raise Exception(error_msg)
                    except (ValueError, TypeError):
                        error_msg = f"シート'{sheet_name}'の列'{col}'、パラメータ'{param_name}': {repr(value)} (浮動小数点数である必要があります)"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                elif expected_type == int:
                    try:
                        int_val = int(value)
                        # 正の整数チェック（必要に応じて）
                        if param_name in ["num_streams", "num_sections"] and int_val <= 0:
                            error_msg = f"シート'{sheet_name}'の列'{col}'、パラメータ'{param_name}': {value} (正の整数である必要があります)"
                            logger.error(error_msg)
                            raise Exception(error_msg)
                    except (ValueError, TypeError):
                        error_msg = f"シート'{sheet_name}'の列'{col}'、パラメータ'{param_name}': {repr(value)} (整数である必要があります)"
                        logger.error(error_msg)
                        raise Exception(error_msg)

            logger.debug(f"シート'{sheet_name}'の列'{col}'の詳細型チェックが成功しました")

    def _load_conditions(self):
        """Excelファイルから条件を読み込む"""
        filepath = f"conditions/{self.cond_id}/sim_conds.xlsx"

        required_sheets = [
            "共通",
            "触媒充填層条件",
            "導入ガス条件",
            "容器壁条件",
            "蓋条件",
            "底条件",
            "均圧配管条件",
            "真空引き配管条件",
            "熱電対条件",
        ]

        # シート存在チェック（例外が発生すれば即座に終了）
        self._validate_sheet_existence(filepath, required_sheets)

        try:
            logger.info(f"条件ファイル({filepath})読み込み開始")

            sheets = pd.read_excel(
                filepath,
                sheet_name=required_sheets,
                index_col=1,
            )

            dataclass_mappings = [
                (CommonConditions, "共通"),
                (PackedBedConditions, "触媒充填層条件"),
                (FeedGasConditions, "導入ガス条件"),
                (VesselConditions, "容器壁条件"),
                (EndCoverConditions, "蓋条件"),
                (EndCoverConditions, "底条件"),
                (EqualizingPipingConditions, "均圧配管条件"),
                (VacuumPipingConditions, "真空引き配管条件"),
                (ThermocoupleConditions, "熱電対条件"),
            ]

            sheet_dataclass_mapping = {sheet_name: dataclass_type for dataclass_type, sheet_name in dataclass_mappings}

            for sheet_name, df in sheets.items():
                # 基本バリデーション（列の存在チェックと基本的な型チェック）
                self._validate_sheet_data(df, sheet_name)

                # 詳細バリデーション（データクラスの型情報を使用）
                if sheet_name in sheet_dataclass_mapping:
                    dataclass_type = sheet_dataclass_mapping[sheet_name]
                    self._validate_sheet_data_with_types(df, sheet_name, dataclass_type)

            # 各塔のデータクラスフィールドバリデーション
            for tower_num in range(1, 4):
                col = f"塔{tower_num}"
                for dataclass_type, sheet_name in dataclass_mappings:
                    params = self._extract_params(sheets[sheet_name], col)
                    self._validate_dataclass_fields(dataclass_type, sheet_name, col, params)

            # 各塔の条件を読み込み
            for tower_num in range(1, 4):
                tower = self._create_tower_conditions(sheets, tower_num)
                tower.initialize_stream_conditions()
                self.towers[tower_num] = tower
            logger.info(f"条件ファイル({filepath})の読み込み完了")

        except Exception as e:
            logger.error(f"条件ファイル読み込み中にエラーが発生: {str(e)}")
            raise Exception(f"条件ファイル({filepath})の読み込みに失敗しました: {str(e)}")

    def _create_tower_conditions(self, sheets: Dict[str, pd.DataFrame], tower_num: int) -> TowerConditions:
        """各塔の条件を作成"""
        col = f"塔{tower_num}"

        extracted_data = {
            "共通": self._extract_params(sheets["共通"], col),
            "触媒充填層条件": self._extract_params(sheets["触媒充填層条件"], col),
            "導入ガス条件": self._extract_params(sheets["導入ガス条件"], col),
            "容器壁条件": self._extract_params(sheets["容器壁条件"], col),
            "蓋条件": self._extract_params(sheets["蓋条件"], col),
            "底条件": self._extract_params(sheets["底条件"], col),
            "均圧配管条件": self._extract_params(sheets["均圧配管条件"], col),
            "真空引き配管条件": self._extract_params(sheets["真空引き配管条件"], col),
            "熱電対条件": self._extract_params(sheets["熱電対条件"], col),
        }

        return TowerConditions(
            common=CommonConditions(**extracted_data["共通"]),
            packed_bed=PackedBedConditions(**extracted_data["触媒充填層条件"]),
            feed_gas=FeedGasConditions(**extracted_data["導入ガス条件"]),
            vessel=VesselConditions(**extracted_data["容器壁条件"]),
            lid=EndCoverConditions(**extracted_data["蓋条件"]),
            bottom=EndCoverConditions(**extracted_data["底条件"]),
            equalizing_piping=EqualizingPipingConditions(**extracted_data["均圧配管条件"]),
            vacuum_piping=VacuumPipingConditions(**extracted_data["真空引き配管条件"]),
            thermocouple=ThermocoupleConditions(**extracted_data["熱電対条件"]),
        )

    def _extract_params(self, df: pd.DataFrame, col: str) -> Dict:
        """DataFrameから指定列のパラメータを辞書として抽出"""
        return {param: df.loc[param, col] for param in df.index}

    def get_tower(self, tower_num: int) -> TowerConditions:
        """指定した塔の条件を取得"""
        return self.towers[tower_num]

    @property
    def num_towers(self) -> int:
        """塔数を取得"""
        return len(self.towers)
