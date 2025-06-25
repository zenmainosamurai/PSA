# 定数定義ファイル

# 物理モデルの実験条件ファイル
CONDITIONS_DIR = "conditions/"

# 観測値ファイル
DATA_DIR = "data/"

# 出力先ファイル
OUTPUT_DIR = "output/"

# 稼働モード
# NOTE: 稼働工程表.xlsxの「稼働モード一覧」と合わせること
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

# 英語→日本語
TRANSLATION = {
    # 出力値
    "temp_reached": "セクション到達温度",
    "temp_thermocouple_reached": "熱電対温度",
    "hw1": "壁-層伝熱係数",
    "Hbb": "下流セルへの熱流束",
    "inlet_co2_volume": "流入CO2流量",
    "inlet_n2_volume": "流入N2流量",
    "inlet_co2_mole_fraction": "流入CO2分率",
    "inlet_n2_mole_fraction": "流入N2分率",
    "gas_density": "ガス密度",
    "gas_specific_heat": "ガス比熱",
    "equilibrium_loading": "平衡吸着量",
    "actual_uptake_volume": "実際のセクション新規吸着量",
    "updated_loading": "時間経過後吸着量",
    "outlet_co2_volume": "下流流出CO2流量",
    "outlet_n2_volume": "下流流出N2流量",
    "outlet_co2_mole_fraction": "下流流出CO2分率",
    "outlet_n2_mole_fraction": "下流流出N2分率",
    "Habs": "発生する吸着熱",
    "Hgas": "流入ガスが受け取る熱",
    "Hroof": "上流セルへの放熱",
    "Hwin": "内側境界からの熱流束",
    "Hwout": "外側境界への熱流束",
    "u1": "層伝熱係数",
    "adsorbent_mass": "セクション吸着材量",
    "theoretical_loading_delta ": "セクション理論新規吸着量",
    "desorp_mw_co2": "気相放出CO2量",
    "case_inner_amount_after_vaccume": "排気後容器内部空間物質量",
    "total_press_after_vaccume": "排気後圧力",
    "mw_co2_after_vaccume": "排気後CO2モル量",
    "mw_n2_after_vaccume": "排気後N2モル量",
    "vacuum_rate": "真空ポンプの排気速度",
    "sum_desorp_mw": "脱着気相放出CO2モル量",
    "mf_co2_after_vaccume": "気相CO2モル分率",
    "mf_n2_after_vaccume": "気相N2モル分率",
    "total_pressure": "全圧",
    "co2_mole_fraction": "気相CO2モル分率",
    "n2_mole_fraction": "気相N2モル分率",
    "co2_partial_pressure": "CO2分圧",
    "outlet_co2_partial_pressure": "流出CO2分圧",
}

UNIT = {
    "セクション到達温度": "[℃]",
    "熱電対温度": "[℃]",
    "壁-層伝熱係数": "[W/m2/K]",
    "下流セルへの熱流束": "[J]",
    "流入CO2流量": "[cm3]",
    "流入N2流量": "[cm3]",
    "流入CO2分率": "",
    "流入N2分率": "",
    "ガス密度": "[kg/m3]",
    "ガス比熱": "[kJ/kg/K]",
    "平衡吸着量": "[cm3/g-abs]",
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
    "気相放出CO2量": "[mol]",
    "排気後容器内部空間物質量": "[mol]",
    "排気後圧力": "[MPaA]",
    "排気後CO2モル量": "[mol]",
    "排気後N2モル量": "[mol]",
    "真空ポンプの排気速度": "[mol]",
    "脱着気相放出CO2モル量": "[mol]",
    "全圧": "[MPaA]",
    "気相CO2モル分率": "",
    "気相N2モル分率": "",
    "CO2分圧": "[MPaA]",
    "流出CO2分圧": "[MPaA]",
}

TRANSLATION_PARAMS = {
    # パラメータ
    "co2_molecular_weight": "CO2分子量 [g/mol]",
    "co2_flow_rate": "CO2流量 [L/min]",
    "n2_molecular_weight": "N2分子量 [g/mol]",
    "n2_flow_rate": "N2流量 [L/min]",
    "total_flow_rate": "全流量 [L/min]",
    "total_pressure": "圧力 [MPaA]",
    "temp": "温度 [degC]",
    "co2_mole_fraction": "CO2分率 [co2]",
    "n2_mole_fraction": "CO2分率 [nitrogen]",
    "co2_density": "CO2密度 [kg/m3]",
    "n2_density": "N2密度 [kg/m3]",
    "average_density": "平均密度 [kg/m3]",
    "co2_thermal_conductivity": "CO2熱伝導率 [W/m/K]",
    "n2_thermal_conductivity": "N2熱伝導率 [W/m/K]",
    "c_mean": "平均熱伝導率 [W/m/K]",
    "vi_co2": "CO2粘度 [Pas]",
    "vi_n2": "N2粘度 [Pas]",
    "vi_mean": "平均粘度 [Pas]",
    "enthalpy": "エンタルピー [kJ/kg]",
    "co2_specific_heat_capacity": "co2比熱 [kJ/kg/K]",
    "n2_specific_heat_capacity": "n2比熱 [kJ/kg/K]",
    "cp_mean": "平均比熱 [kJ/kg/K]",
    "C_per_hour": "1h熱容量 [kJ/K/h]",
    "co2_adsorption_heat": "CO2吸着熱 [kJ/kg]",
    "n2_adsorption_heat": "N2吸着熱 [kJ/kg]",
}
