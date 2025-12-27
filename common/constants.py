"""物理定数

PSA担当者向け説明:
シミュレーションで使用する物理定数を定義しています。
これらの値は普遍的な物理定数であり、変更することはありません。

使用例:
    from common.constants import GAS_CONSTANT, STANDARD_PRESSURE
    
    # 理想気体の状態方程式: PV = nRT
    n = P * V / (GAS_CONSTANT * T)
"""

# ============================================================
# 基本物理定数
# ============================================================

# 気体定数 [J/(mol·K)]
GAS_CONSTANT = 8.314

# 標準状態（0℃, 1atm）での気体のモル体積 [L/mol]
STANDARD_MOLAR_VOLUME = 22.4

# 標準大気圧 [Pa]
STANDARD_PRESSURE = 101325

# 重力加速度 [m/s²]
GRAVITY_ACCELERATION = 9.81

# 摂氏から絶対温度への変換オフセット [K]
CELSIUS_TO_KELVIN_OFFSET = 273.15


# ============================================================
# 単位変換係数
# ============================================================

# 圧力変換
PA_TO_MPA = 1e-6      # Pa → MPa
MPA_TO_PA = 1e6       # MPa → Pa
MPA_TO_KPA = 1e3      # MPa → kPa

# 時間変換
MINUTE_TO_SECOND = 60  # min → s

# 体積変換
M3_TO_L = 1e3         # m³ → L
L_TO_M3 = 1e-3        # L → m³
CM3_TO_L = 1e-3       # cm³ → L
L_TO_CM3 = 1e3        # L → cm³

# エネルギー変換
J_TO_KJ = 1e-3        # J → kJ


# ============================================================
# 計算用の下限値（ゼロ除算防止など）
# ============================================================

# 平衡吸着量の最小値 [cm³/g-abs]
# これより小さい場合はこの値に置き換え
MINIMUM_EQUILIBRIUM_LOADING = 0.1

# CO2分圧の最小値 [MPa]
# これより小さい場合はこの値に置き換え
MINIMUM_CO2_PARTIAL_PRESSURE = 2.5e-3
