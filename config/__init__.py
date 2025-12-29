"""シミュレーション条件の設定モジュール

Excelファイル（sim_conds.xlsx）から塔の物理条件を読み込み、
シミュレーションで使用するデータクラスに変換します。

主なクラス:
    SimulationConditions: 全塔の条件を管理
    TowerConditions: 1塔分の条件（充填層、容器、配管など）
"""
