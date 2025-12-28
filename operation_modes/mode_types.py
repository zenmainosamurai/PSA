"""運転モード定義

PSA担当者向け説明:
各運転モードを定義しています。稼働工程表の「塔1」「塔2」「塔3」列に
記載されるモード名と対応しています。

使用例:
    from operation_modes.mode_types import OperationMode
    
    mode = OperationMode.from_japanese("流通吸着_単独/上流")
    if mode in ADSORPTION_MODES:
        # 吸着計算を実行
        ...
"""

from enum import Enum, IntEnum
from typing import Set


class HeatCalculationMode(IntEnum):
    """
    熱収支計算用のモード
    
    PSA担当者向け説明:
    物理計算（熱収支）で使用する内部モード番号です。
    運転モードとの対応は OperationMode.heat_mode プロパティで取得できます。
    """
    ADSORPTION = 0      # 吸着（ガス流通あり）
    VALVE_CLOSED = 1    # 停止（弁閉止、ガス流通なし）
    DESORPTION = 2      # 脱着（真空排気）


class OperationMode(Enum):
    """
    運転モード
    
    各モードは稼働工程表に記載される日本語名をvalueとして持ちます。
    """
    # 基本モード
    INITIAL_GAS_INTRODUCTION = "初回ガス導入"
    STOP = "停止"
    
    # 流通吸着
    FLOW_ADSORPTION_UPSTREAM = "流通吸着_単独/上流"
    FLOW_ADSORPTION_DOWNSTREAM = "流通吸着_下流"
    
    # バッチ吸着
    BATCH_ADSORPTION_UPSTREAM = "バッチ吸着_上流"
    BATCH_ADSORPTION_DOWNSTREAM = "バッチ吸着_下流"
    BATCH_ADSORPTION_UPSTREAM_WITH_VALVE = "バッチ吸着_上流（圧調弁あり）"
    BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE = "バッチ吸着_下流（圧調弁あり）"
    
    # 均圧
    EQUALIZATION_DEPRESSURIZATION = "均圧_減圧"
    EQUALIZATION_PRESSURIZATION = "均圧_加圧"
    
    # 真空脱着
    VACUUM_DESORPTION = "真空脱着"
    
    @classmethod
    def from_japanese(cls, name: str) -> "OperationMode":
        """
        日本語名からEnumに変換
        
        Args:
            name: 稼働工程表に記載される日本語のモード名
            
        Returns:
            対応するOperationMode
            
        Raises:
            ValueError: 未対応のモード名の場合
        """
        for mode in cls:
            if mode.value == name:
                return mode
        raise ValueError(f"未対応の運転モード: {name}")
    
    @property
    def japanese_name(self) -> str:
        """日本語名を取得"""
        return self.value
    
    def is_adsorption_mode(self) -> bool:
        """吸着計算を行うモードかどうか"""
        return self in ADSORPTION_MODES
    
    def is_upstream_mode(self) -> bool:
        """上流モードかどうか"""
        return self in UPSTREAM_MODES
    
    def is_downstream_mode(self) -> bool:
        """下流モードかどうか"""
        return self in DOWNSTREAM_MODES
    
    def requires_pressure_update(self) -> bool:
        """圧力更新が必要なモードかどうか"""
        return self in PRESSURE_UPDATE_MODES
    
    def requires_mole_fraction_update(self) -> bool:
        """モル分率更新が必要なモードかどうか"""
        return self in MOLE_FRACTION_UPDATE_MODES
    
    @property
    def heat_mode(self) -> HeatCalculationMode:
        """
        熱収支計算用のモード番号を取得
        
        PSA担当者向け説明:
        運転モードに対応する熱収支計算用のモード（整数）を返します。
        - 停止モード: VALVE_CLOSED (1) - 弁閉止状態
        - 真空脱着モード: DESORPTION (2) - 脱着計算
        - その他: ADSORPTION (0) - 吸着計算
        
        Returns:
            HeatCalculationMode: 熱収支計算用モード
        """
        if self == OperationMode.STOP:
            return HeatCalculationMode.VALVE_CLOSED
        elif self == OperationMode.VACUUM_DESORPTION:
            return HeatCalculationMode.DESORPTION
        return HeatCalculationMode.ADSORPTION


# ============================================================
# モードのグループ分け（計算ロジックで使用）
# ============================================================

# 吸着計算を行うモード（物質収支で吸着計算を使用）
ADSORPTION_MODES: Set[OperationMode] = {
    OperationMode.INITIAL_GAS_INTRODUCTION,
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
    OperationMode.EQUALIZATION_DEPRESSURIZATION,
    OperationMode.EQUALIZATION_PRESSURIZATION,
}

# 上流モード（フィードガスが直接流入）
UPSTREAM_MODES: Set[OperationMode] = {
    OperationMode.INITIAL_GAS_INTRODUCTION,
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
}

# 下流モード（他塔からのガスが流入）
DOWNSTREAM_MODES: Set[OperationMode] = {
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
}

# 上流・下流ペアの定義（塔間連携計算で使用）
UPSTREAM_DOWNSTREAM_PAIRS = [
    (OperationMode.FLOW_ADSORPTION_UPSTREAM, OperationMode.FLOW_ADSORPTION_DOWNSTREAM),
    (OperationMode.BATCH_ADSORPTION_UPSTREAM, OperationMode.BATCH_ADSORPTION_DOWNSTREAM),
    (OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE, OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE),
]

# 均圧モード
EQUALIZATION_MODES: Set[OperationMode] = {
    OperationMode.EQUALIZATION_DEPRESSURIZATION,
    OperationMode.EQUALIZATION_PRESSURIZATION,
}

# 圧力更新が必要なモード
PRESSURE_UPDATE_MODES: Set[OperationMode] = {
    OperationMode.INITIAL_GAS_INTRODUCTION,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
    OperationMode.EQUALIZATION_PRESSURIZATION,
    OperationMode.EQUALIZATION_DEPRESSURIZATION,
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
    OperationMode.VACUUM_DESORPTION,
}

# モル分率更新が必要なモード
MOLE_FRACTION_UPDATE_MODES: Set[OperationMode] = {
    OperationMode.INITIAL_GAS_INTRODUCTION,
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
    OperationMode.EQUALIZATION_PRESSURIZATION,
    OperationMode.EQUALIZATION_DEPRESSURIZATION,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
    OperationMode.VACUUM_DESORPTION,
}

# バッチ吸着後の圧力計算が必要なモード
BATCH_PRESSURE_CALCULATION_MODES: Set[OperationMode] = {
    OperationMode.INITIAL_GAS_INTRODUCTION,
    OperationMode.BATCH_ADSORPTION_UPSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_DOWNSTREAM_WITH_VALVE,
    OperationMode.EQUALIZATION_PRESSURIZATION,
}

# 流通吸着後の圧力設定が必要なモード（フィードガス圧力を使用）
FLOW_PRESSURE_MODES: Set[OperationMode] = {
    OperationMode.FLOW_ADSORPTION_UPSTREAM,
    OperationMode.FLOW_ADSORPTION_DOWNSTREAM,
    OperationMode.BATCH_ADSORPTION_UPSTREAM_WITH_VALVE,
}


def get_mode_category(mode: OperationMode) -> str:
    """
    モードのカテゴリを取得（ログ出力等で使用）
    
    Args:
        mode: 運転モード
        
    Returns:
        カテゴリ名（日本語）
    """
    if mode == OperationMode.STOP:
        return "停止"
    elif mode == OperationMode.VACUUM_DESORPTION:
        return "真空脱着"
    elif mode in EQUALIZATION_MODES:
        return "均圧"
    elif mode in UPSTREAM_MODES:
        return "吸着（上流）"
    elif mode in DOWNSTREAM_MODES:
        return "吸着（下流）"
    else:
        return "その他"
