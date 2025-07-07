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
    "wall_to_bed_heat_transfer_coef": "壁-層伝熱係数",
    "downstream_heat_flux": "下流セルへの熱流束",
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
    "adsorption_heat": "発生する吸着熱",
    "Hgas": "流入ガスが受け取る熱",
    "upstream_heat_flux": "上流セルへの放熱",
    "heat_flux_from_inner_boundary": "内側境界からの熱流束",
    "heat_flux_to_outer_boundary": "外側境界への熱流束",
    "bed_heat_transfer_coef": "層伝熱係数",
    "adsorbent_mass": "セクション吸着材量",
    "theoretical_loading_delta": "セクション理論新規吸着量",
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
    "vacuum_rate_co2": "CO2回収率",
    "cumulative_co2_recovered": "累積CO2回収量",
    "cumulative_n2_recovered": "累積N2回収量",
}

UNIT = {
    "セクション到達温度": "[℃]",
    "熱電対温度": "[℃]",
    "壁-層伝熱係数": "[W/m2/K]",
    "下流セルへの熱流束": "[J]",
    "流入CO2流量": "[Nm3]",
    "流入N2流量": "[Nm3]",
    "流入CO2分率": "[-]",
    "流入N2分率": "[-]",
    "ガス密度": "[kg/m3]",
    "ガス比熱": "[kJ/kg/K]",
    "平衡吸着量": "[Ncm3/g-abs]",
    "実際のセクション新規吸着量": "[Ncm3]",
    "時間経過後吸着量": "[Ncm3/g-abs]",
    "下流流出CO2流量": "[Nm3]",
    "下流流出N2流量": "[Nm3]",
    "下流流出CO2分率": "[-]",
    "下流流出N2分率": "[-]",
    "発生する吸着熱": "[J]",
    "流入ガスが受け取る熱": "[J]",
    "上流セルへの放熱": "[J]",
    "内側境界からの熱流束": "[J]",
    "外側境界への熱流束": "[J]",
    "層伝熱係数": "[W/m2/K]",
    "セクション吸着材量": "[g]",
    "セクション理論新規吸着量": "[Ncm3/g-abs]",
    "気相放出CO2量": "[mol]",
    "排気後容器内部空間物質量": "[mol]",
    "排気後圧力": "[MPaA]",
    "排気後CO2モル量": "[mol]",
    "排気後N2モル量": "[mol]",
    "真空ポンプの排気速度": "[mol]",
    "脱着気相放出CO2モル量": "[mol]",
    "全圧": "[MPaA]",
    "気相CO2モル分率": "[-]",
    "気相N2モル分率": "[-]",
    "CO2分圧": "[MPaA]",
    "流出CO2分圧": "[MPaA]",
    "CO2回収率": "[%]",
    "累積CO2回収量": "[Nm3]",
    "累積N2回収量": "[Nm3]",
}
