"""日本語表示用辞書

PSA担当者向け説明:
英語の変数名を日本語表示名に変換するための辞書です。
CSV出力やグラフのラベル表示で使用します。

新しい出力項目を追加する場合は、ここに対応する日本語名と単位を追加してください。
"""

# ============================================================
# 英語→日本語の変換辞書
# ============================================================

TRANSLATION = {
    # 温度関連
    "temp_reached": "セクション到達温度",
    "temp_thermocouple_reached": "熱電対温度",
    
    # 伝熱係数
    "wall_to_bed_heat_transfer_coef": "壁-層伝熱係数",
    "bed_heat_transfer_coef": "層伝熱係数",
    
    # 熱流束
    "downstream_heat_flux": "下流セルへの熱流束",
    "upstream_heat_flux": "上流セルへの放熱",
    "heat_flux_from_inner_boundary": "内側境界からの熱流束",
    "heat_flux_to_outer_boundary": "外側境界への熱流束",
    "adsorption_heat": "発生する吸着熱",
    "Hgas": "流入ガスが受け取る熱",
    
    # 流量（入口）
    "inlet_co2_volume": "流入CO2流量",
    "inlet_n2_volume": "流入N2流量",
    "inlet_co2_mole_fraction": "流入CO2分率",
    "inlet_n2_mole_fraction": "流入N2分率",
    
    # 流量（出口）
    "outlet_co2_volume": "下流流出CO2流量",
    "outlet_n2_volume": "下流流出N2流量",
    "outlet_co2_mole_fraction": "下流流出CO2分率",
    "outlet_n2_mole_fraction": "下流流出N2分率",
    
    # ガス物性
    "gas_density": "ガス密度",
    "gas_specific_heat": "ガス比熱",
    
    # 吸着関連
    "equilibrium_loading": "平衡吸着量",
    "actual_uptake_volume": "実際のセクション新規吸着量",
    "updated_loading": "時間経過後吸着量",
    "adsorbent_mass": "セクション吸着材量",
    "theoretical_loading_delta": "セクション理論新規吸着量",
    
    # 圧力関連
    "total_pressure": "全圧",
    "co2_partial_pressure": "CO2分圧",
    "outlet_co2_partial_pressure": "流出CO2分圧",
    
    # モル分率
    "co2_mole_fraction": "気相CO2モル分率",
    "n2_mole_fraction": "気相N2モル分率",
    
    # 脱着関連
    "desorp_mw_co2": "気相放出CO2量",
    "case_inner_amount_after_vaccume": "排気後容器内部空間物質量",
    "total_press_after_vaccume": "排気後圧力",
    "mw_co2_after_vaccume": "排気後CO2モル量",
    "mw_n2_after_vaccume": "排気後N2モル量",
    "vacuum_rate": "真空ポンプの排気速度",
    "sum_desorp_mw": "脱着気相放出CO2モル量",
    "mf_co2_after_vaccume": "気相CO2モル分率",
    "mf_n2_after_vaccume": "気相N2モル分率",
    
    # 回収量
    "vacuum_rate_co2": "CO2回収率",
    "cumulative_co2_recovered": "累積CO2回収量",
    "cumulative_n2_recovered": "累積N2回収量",
}


# ============================================================
# 単位辞書
# ============================================================

UNIT = {
    # 温度
    "セクション到達温度": "[℃]",
    "熱電対温度": "[℃]",
    "蓋温度": "[℃]",
    
    # 伝熱係数
    "壁-層伝熱係数": "[W/m2/K]",
    "層伝熱係数": "[W/m2/K]",
    
    # 熱流束・熱量
    "下流セルへの熱流束": "[J]",
    "上流セルへの放熱": "[J]",
    "内側境界からの熱流束": "[J]",
    "外側境界への熱流束": "[J]",
    "発生する吸着熱": "[J]",
    "流入ガスが受け取る熱": "[J]",
    
    # 流量
    "流入CO2流量": "[Nm3]",
    "流入N2流量": "[Nm3]",
    "下流流出CO2流量": "[Nm3]",
    "下流流出N2流量": "[Nm3]",
    
    # 分率
    "流入CO2分率": "[-]",
    "流入N2分率": "[-]",
    "下流流出CO2分率": "[-]",
    "下流流出N2分率": "[-]",
    "気相CO2モル分率": "[-]",
    "気相N2モル分率": "[-]",
    
    # ガス物性
    "ガス密度": "[kg/m3]",
    "ガス比熱": "[kJ/kg/K]",
    
    # 吸着量
    "平衡吸着量": "[Ncm3/g-abs]",
    "実際のセクション新規吸着量": "[Ncm3]",
    "時間経過後吸着量": "[Ncm3/g-abs]",
    "セクション吸着材量": "[g]",
    "セクション理論新規吸着量": "[Ncm3/g-abs]",
    
    # 圧力
    "全圧": "[MPaA]",
    "CO2分圧": "[MPaA]",
    "流出CO2分圧": "[MPaA]",
    "排気後圧力": "[MPaA]",
    
    # モル量
    "気相放出CO2量": "[mol]",
    "排気後容器内部空間物質量": "[mol]",
    "排気後CO2モル量": "[mol]",
    "排気後N2モル量": "[mol]",
    "真空ポンプの排気速度": "[mol]",
    "脱着気相放出CO2モル量": "[mol]",
    
    # 回収量
    "CO2回収率": "[%]",
    "累積CO2回収量": "[Nm3]",
    "累積N2回収量": "[Nm3]",
}


# ============================================================
# ヘルパー関数
# ============================================================

def translate(english_name: str) -> str:
    """
    英語名を日本語名に変換
    
    Args:
        english_name: 英語の変数名
        
    Returns:
        日本語名（辞書にない場合は英語名をそのまま返す）
    """
    return TRANSLATION.get(english_name, english_name)


def get_unit(japanese_name: str) -> str:
    """
    日本語名に対応する単位を取得
    
    Args:
        japanese_name: 日本語の変数名
        
    Returns:
        単位（辞書にない場合は空文字列）
    """
    return UNIT.get(japanese_name, "")


def get_label_with_unit(english_name: str) -> str:
    """
    英語名から「日本語名 [単位]」形式のラベルを生成
    
    Args:
        english_name: 英語の変数名
        
    Returns:
        「日本語名 [単位]」形式の文字列
    """
    japanese_name = translate(english_name)
    unit = get_unit(japanese_name)
    if unit:
        return f"{japanese_name} {unit}"
    return japanese_name
