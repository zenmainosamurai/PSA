"""終了条件判定

PSA担当者向け説明:
工程の終了条件を判定するモジュールです。
稼働工程表の「終了条件」列に記載された条件を解析し、
シミュレーションの各ステップで終了判定を行います。

終了条件の種類:
- 圧力到達: 指定塔の圧力が目標値に到達
- 温度到達: 指定塔・セクションの温度が目標値に到達
- 時間経過: プロセス開始からの経過時間
- 時間到達: シミュレーション開始からの絶対時間

書式例:
- "圧力到達_塔1_0.3" → 塔1の圧力が0.3MPaAに到達
- "温度到達_塔2_50" → 塔2の温度が50℃に到達
- "時間経過_5_min" → 5分経過
- "時間到達_30" → シミュレーション開始から30分到達
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

from state import StateVariables


class TerminationConditionType(Enum):
    """終了条件の種類"""
    PRESSURE_REACHED = "圧力到達"
    TEMPERATURE_REACHED = "温度到達"
    TIME_ELAPSED = "時間経過"
    TIME_REACHED = "時間到達"


@dataclass
class TerminationCondition:
    """
    終了条件のデータクラス
    
    PSA担当者向け説明:
    稼働工程表の「終了条件」列を解析した結果を保持します。
    """
    condition_type: TerminationConditionType
    tower_num: Optional[int] = None  # 塔番号（圧力/温度到達の場合）
    target_value: Optional[float] = None  # 目標値
    unit: Optional[str] = None  # 単位（時間経過の場合）


def parse_termination_condition(condition_str: str) -> TerminationCondition:
    """
    終了条件文字列を解析
    
    PSA担当者向け説明:
    稼働工程表に記載された終了条件文字列を解析します。
    
    Args:
        condition_str: 終了条件文字列（例: "圧力到達_塔1_0.3"）
    
    Returns:
        TerminationCondition: 解析結果
    
    Raises:
        ValueError: 未対応の終了条件形式の場合
    
    使用例:
        cond = parse_termination_condition("圧力到達_塔1_0.3")
        # cond.condition_type == TerminationConditionType.PRESSURE_REACHED
        # cond.tower_num == 1
        # cond.target_value == 0.3
    """
    parts = condition_str.split("_")
    cond_type_str = parts[0]
    
    if cond_type_str == "圧力到達":
        tower_num = int(parts[1][-1])  # "塔1" → 1
        target_value = float(parts[2])
        return TerminationCondition(
            condition_type=TerminationConditionType.PRESSURE_REACHED,
            tower_num=tower_num,
            target_value=target_value,
        )
    
    elif cond_type_str == "温度到達":
        tower_num = int(parts[1][-1])
        target_value = float(parts[2])
        return TerminationCondition(
            condition_type=TerminationConditionType.TEMPERATURE_REACHED,
            tower_num=tower_num,
            target_value=target_value,
        )
    
    elif cond_type_str == "時間経過":
        target_value = float(parts[1])
        unit = parts[2] if len(parts) > 2 else "min"
        return TerminationCondition(
            condition_type=TerminationConditionType.TIME_ELAPSED,
            target_value=target_value,
            unit=unit,
        )
    
    elif cond_type_str == "時間到達":
        target_value = float(parts[1])
        return TerminationCondition(
            condition_type=TerminationConditionType.TIME_REACHED,
            target_value=target_value,
        )
    
    else:
        raise ValueError(f"未対応の終了条件: {condition_str}")


def check_termination_condition(
    condition: TerminationCondition,
    state_manager: StateVariables,
    timestamp: float,
    timestamp_in_process: float,
    num_sections: int,
) -> bool:
    """
    終了条件を判定
    
    PSA担当者向け説明:
    現在の状態が終了条件を満たしているか判定します。
    Falseが返ると工程が終了、Trueが返ると継続します。
    
    Args:
        condition: 終了条件
        state_manager: 状態変数管理
        timestamp: シミュレーション開始からの経過時間 [min]
        timestamp_in_process: 現工程開始からの経過時間 [min]
        num_sections: セクション数
    
    Returns:
        bool: True=継続、False=終了
    
    使用例:
        cond = parse_termination_condition("圧力到達_塔1_0.3")
        should_continue = check_termination_condition(
            cond, state_manager, timestamp=10, timestamp_in_process=2, num_sections=5
        )
        if not should_continue:
            print("工程終了")
    """
    if condition.condition_type == TerminationConditionType.PRESSURE_REACHED:
        current_pressure = state_manager.towers[condition.tower_num].total_press
        # 圧力が目標値未満なら継続
        return current_pressure < condition.target_value
    
    elif condition.condition_type == TerminationConditionType.TEMPERATURE_REACHED:
        # 最下流セクションの平均温度を使用
        tower_state = state_manager.towers[condition.tower_num]
        current_temp = np.mean(tower_state.temp[:, num_sections - 1])
        # 温度が目標値未満なら継続
        return current_temp < condition.target_value
    
    elif condition.condition_type == TerminationConditionType.TIME_ELAPSED:
        # 単位変換（分に統一）
        target_time = condition.target_value
        if condition.unit == "s":
            target_time /= 60
        # 経過時間が目標未満なら継続
        return timestamp_in_process < target_time
    
    elif condition.condition_type == TerminationConditionType.TIME_REACHED:
        # 絶対時間が目標未満なら継続
        return timestamp + timestamp_in_process < condition.target_value
    
    return False


def should_continue_process(
    termination_cond_str: str,
    state_manager: StateVariables,
    timestamp: float,
    timestamp_in_process: float,
    num_sections: int,
) -> bool:
    """
    工程を継続すべきか判定（簡易インターフェース）
    
    PSA担当者向け説明:
    稼働工程表の終了条件文字列から直接判定を行う簡易関数です。
    
    Args:
        termination_cond_str: 終了条件文字列
        state_manager: 状態変数管理
        timestamp: シミュレーション開始からの経過時間 [min]
        timestamp_in_process: 現工程開始からの経過時間 [min]
        num_sections: セクション数
    
    Returns:
        bool: True=継続、False=終了
    
    使用例:
        while should_continue_process("時間経過_5_min", state_manager, timestamp, t_p, 5):
            # 計算を実行
            ...
    """
    condition = parse_termination_condition(termination_cond_str)
    return check_termination_condition(
        condition=condition,
        state_manager=state_manager,
        timestamp=timestamp,
        timestamp_in_process=timestamp_in_process,
        num_sections=num_sections,
    )
