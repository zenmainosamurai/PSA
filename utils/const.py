# 定数定義ファイル

# 物理モデルの実験条件ファイル
CONDITIONS_DIR = "conditions/"

# 観測値ファイル
DATA_DIR = "data/"

# 出力先ファイル
OUTPUT_DIR = "output/"

# 英語→日本語
TRANSLATION = {
    # 出力値
    "temp_reached": "セクション到達温度",
    "hw1": "壁-層伝熱係数",
    "Hbb": "下流セルへの熱流束",
    "inflow_fr_co2": "流入CO2流量",
    "inflow_fr_n2": "流入N2流量",
    "inflow_mf_co2": "流入CO2分率",
    "inflow_mf_n2": "流入N2分率",
    "gas_density": "ガス密度",
    "gas_cp": "ガス比熱",
    "adsorp_amt_estimate": "実際のセクション新規吸着量",
    "accum_adsorp_amt": "時間経過後吸着量",
    "outflow_fr_co2": "下流流出CO2流量",
    "outflow_fr_n2": "下流流出N2流量",
    "outflow_mf_co2": "下流流出CO2分率",
    "outflow_mf_n2": "下流流出N2分率",
    "Habs": "発生する吸着熱",
    "Hgas": "流入ガスが受け取る熱",
    "Hroof": "上流セルへの放熱",
    "Hwin": "内側境界からの熱流束",
    "Hwout": "外側境界への熱流束",
    "u1": "層伝熱係数",
    "Mabs": "セクション吸着材量",
    "adsorp_amt_estimate_abs": "セクション理論新規吸着量",
}
TRANSLATION_PARAMS = {
    # パラメータ
    "mw_co2": "CO2分子量 [g/mol]",
    "fr_co2": "CO2流量 [L/min]",
    "mw_n2": "N2分子量 [g/mol]",
    "fr_n2": "N2流量 [L/min]",
    "fr_all": "全流量 [L/min]",
    "press": "圧力 [MPaA]",
    "temp": "温度 [degC]",
    "mf_co2": "CO2分率 [co2]",
    "mf_n2": "CO2分率 [nitrogen]",
    "dense_co2": "CO2密度 [kg/m3]",
    "dense_n2": "N2密度 [kg/m3]",
    "dense_mean": "平均密度 [kg/m3]",
    "c_co2": "CO2熱伝導率 [W/m/K]",
    "c_n2": "N2熱伝導率 [W/m/K]",
    "c_mean": "平均熱伝導率 [W/m/K]",
    "vi_co2": "CO2粘度 [Pas]",
    "vi_n2": "N2粘度 [Pas]",
    "vi_mean": "平均粘度 [Pas]",
    "enthalpy": "エンタルピー [kJ/kg]",
    "cp_co2": "co2比熱 [kJ/kg/K]",
    "cp_n2": "n2比熱 [kJ/kg/K]",
    "cp_mean": "平均比熱 [kJ/kg/K]",
    "C_per_hour": "1h熱容量 [kJ/K/h]",
    "adsorp_heat_co2": "CO2吸着熱 [kJ/kg]",
    "adsorp_heat_n2": "N2吸着熱 [kJ/kg]",
}

UNIT = {
    "セクション到達温度": "[℃]",
    "壁-層伝熱係数": "[W/m2/K]",
    "下流セルへの熱流束": "[J]",
    "流入CO2流量": "[cm3]",
    "流入N2流量": "[cm3]",
    "流入CO2分率": "",
    "流入N2分率": "",
    "ガス密度": "[kg/m3]",
    "ガス比熱": "[kJ/kg/K]",
    "実際のセクション新規吸着量": "[cm3]",
    "時間経過後吸着量": "[cm3/g-abs]",
    "下流流出CO2流量": "[cm3]",
    "下流流出N2流量": "[cm3]",
    "下流流出CO2分率": "",
    "下流流出N2分率": "",
    "発生する吸着熱": "[J]",
    "流入ガスが受け取る熱": "[J]",
    "上流セルへの放熱": "[J]",
    "内側境界からの熱流束": "[J]",
    "外側境界への熱流束": "[J]",
    "層伝熱係数": "[W/m2/K]",
    "セクション吸着材量": "[g]",
    "セクション理論新規吸着量": "[g]",
}