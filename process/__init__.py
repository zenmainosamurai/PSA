"""プロセス制御モジュール

PSA担当者向け説明:
PSA工程（稼働工程表）に従ってシミュレーションを進行させる部分です。

主要なモジュール（今後作成予定）:
- simulator.py: シミュレーター本体（軽量化版）
- process_executor.py: 工程実行ロジック
- termination_conditions.py: 終了条件判定（圧力到達・温度到達・時間経過など）

現在は旧コード（core/simulator.py）からの移行準備段階です。
"""

__all__ = []
