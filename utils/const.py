"""定数定義ファイル（後方互換性のためのエイリアス）

このファイルは後方互換性のために残されています。
新しいコードでは以下のモジュールを直接使用してください:
- common.constants: 物理定数・単位変換係数
- common.paths: ディレクトリパス
- common.translations: 日本語翻訳・単位辞書
"""

# 物理定数・単位変換係数（common/constants.py から再エクスポート）
from common.constants import (
    CELSIUS_TO_KELVIN_OFFSET,
    STANDARD_MOLAR_VOLUME,
    GAS_CONSTANT,
    STANDARD_PRESSURE,
    PA_TO_MPA,
    MPA_TO_PA,
    MPA_TO_KPA,
    MINUTE_TO_SECOND,
    GRAVITY_ACCELERATION,
    M3_TO_L,
    CM3_TO_L,
    L_TO_CM3,
    L_TO_M3,
    J_TO_KJ,
    MINIMUM_EQUILIBRIUM_LOADING,
    MINIMUM_CO2_PARTIAL_PRESSURE,
)

# ディレクトリパス（common/paths.py から再エクスポート）
from common.paths import (
    CONDITIONS_DIR,
    DATA_DIR,
    OUTPUT_DIR,
)

# 翻訳辞書・単位辞書（common/translations.py から再エクスポート）
from common.translations import (
    TRANSLATION,
    UNIT,
)

# 稼働モード（このファイル固有、operation_modes/mode_types.py と重複するが互換性のため残す）
OPERATION_MODE = {
    1: "初回ガス導入",
    2: "停止",
    3: "流通吸着_単独/上流",
    4: "バッチ吸着_上流",
    5: "均圧_減圧",
    6: "真空脱着",
    7: "均圧_加圧",
    8: "バッチ吸着_下流",
    9: "流通吸着_下流",
}
