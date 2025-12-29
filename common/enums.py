"""共通Enum定義

循環インポートを避けるため、複数モジュールで共有するEnumをここに定義します。
"""

from enum import Enum


class LidPosition(Enum):
    """
    蓋の位置
    
    容器の上蓋・下蓋を区別するために使用します。
    """
    TOP = "top"         # 上蓋（top_temperature）
    BOTTOM = "bottom"   # 下蓋（bottom_temperature）
