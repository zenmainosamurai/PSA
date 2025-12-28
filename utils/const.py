"""定数定義ファイル（後方互換性のためのエイリアス）

このファイルは後方互換性のために残されています。
新しいコードでは以下のモジュールを直接使用してください:
- common.constants: 物理定数・単位変換係数
- common.paths: ディレクトリパス
- common.translations: 日本語翻訳・単位辞書
"""

# 翻訳辞書・単位辞書（common/translations.py から再エクスポート）
from common.translations import (
    TRANSLATION,
    UNIT,
)

# 稼働モード
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
