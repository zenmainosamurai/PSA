"""共通ユーティリティ

シミュレーション全体で使用する定数、パス設定、単位変換などを提供します。

主要なエクスポート:
- 物理定数: GAS_CONSTANT, STANDARD_PRESSURE など
- パス設定: CONDITIONS_DIR, OUTPUT_DIR など
- 単位変換: convert_cm3_to_nm3 など
- 翻訳辞書: TRANSLATION, UNIT
"""

from .constants import (
    # 物理定数
    GAS_CONSTANT,
    STANDARD_MOLAR_VOLUME,
    STANDARD_PRESSURE,
    GRAVITY_ACCELERATION,
    CELSIUS_TO_KELVIN_OFFSET,
    
    # 単位変換係数
    PA_TO_MPA,
    MPA_TO_PA,
    MPA_TO_KPA,
    MINUTE_TO_SECOND,
    M3_TO_L,
    L_TO_M3,
    CM3_TO_L,
    L_TO_CM3,
    J_TO_KJ,
    
    # 計算用下限値
    MINIMUM_EQUILIBRIUM_LOADING,
    MINIMUM_CO2_PARTIAL_PRESSURE,
)

from .paths import (
    CONDITIONS_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    SIM_CONDITIONS_FILENAME,
    OPERATION_SCHEDULE_FILENAME,
    OBSERVATION_DATA_FILENAME,
    get_condition_dir,
    get_sim_conditions_path,
    get_operation_schedule_path,
    get_output_dir,
    get_observation_data_path,
)

from .translations import (
    TRANSLATION,
    UNIT,
    translate,
    get_unit,
    get_label_with_unit,
)

from .unit_conversion import (
    convert_cm3_to_nm3,
    convert_nm3_to_cm3,
    convert_l_per_min_to_m3_per_min,
    convert_m3_per_min_to_l_per_min,
    convert_celsius_to_kelvin,
    convert_kelvin_to_celsius,
    convert_mpa_to_pa,
    convert_pa_to_mpa,
)

__all__ = [
    # 物理定数
    "GAS_CONSTANT",
    "STANDARD_MOLAR_VOLUME",
    "STANDARD_PRESSURE",
    "GRAVITY_ACCELERATION",
    "CELSIUS_TO_KELVIN_OFFSET",
    
    # 単位変換係数
    "PA_TO_MPA",
    "MPA_TO_PA",
    "MPA_TO_KPA",
    "MINUTE_TO_SECOND",
    "M3_TO_L",
    "L_TO_M3",
    "CM3_TO_L",
    "L_TO_CM3",
    "J_TO_KJ",
    
    # 計算用下限値
    "MINIMUM_EQUILIBRIUM_LOADING",
    "MINIMUM_CO2_PARTIAL_PRESSURE",
    
    # パス設定
    "CONDITIONS_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "LOG_DIR",
    "SIM_CONDITIONS_FILENAME",
    "OPERATION_SCHEDULE_FILENAME",
    "OBSERVATION_DATA_FILENAME",
    "get_condition_dir",
    "get_sim_conditions_path",
    "get_operation_schedule_path",
    "get_output_dir",
    "get_observation_data_path",
    
    # 翻訳辞書
    "TRANSLATION",
    "UNIT",
    "translate",
    "get_unit",
    "get_label_with_unit",
    
    # 単位変換関数
    "convert_cm3_to_nm3",
    "convert_nm3_to_cm3",
    "convert_l_per_min_to_m3_per_min",
    "convert_m3_per_min_to_l_per_min",
    "convert_celsius_to_kelvin",
    "convert_kelvin_to_celsius",
    "convert_mpa_to_pa",
    "convert_pa_to_mpa",
]
