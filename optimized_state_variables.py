import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TowerStateArrays:
    """各塔の状態変数を効率的に保持するためのデータクラス"""
    # 2D arrays (stream x section)
    temp: np.ndarray  # 温度
    temp_thermo: np.ndarray  # 熱電対温度
    adsorp_amt: np.ndarray  # 吸着量
    mf_co2: np.ndarray  # CO2モル分率
    mf_n2: np.ndarray  # N2モル分率
    heat_t_coef: np.ndarray  # 層伝熱係数
    heat_t_coef_wall: np.ndarray  # 壁-層伝熱係数
    outflow_pco2: np.ndarray  # 流出CO2分圧
    
    # 1D arrays (section only)
    temp_wall: np.ndarray  # 壁面温度
    
    # Scalars
    temp_lid_up: float  # 上蓋温度
    temp_lid_down: float  # 下蓋温度
    total_press: float  # 全圧
    vacuum_amt_co2: float  # CO2回収量
    vacuum_amt_n2: float  # N2回収量


class OptimizedStateVariables:
    """最適化された状態変数管理クラス"""
    
    def __init__(self, num_towers: int, num_streams: int, num_sections: int, sim_conds: Dict):
        self.num_towers = num_towers
        self.num_streams = num_streams
        self.num_sections = num_sections
        self.sim_conds = sim_conds
        
        # 各塔の状態変数を初期化
        self.towers: Dict[int, TowerStateArrays] = {}
        for tower_num in range(1, num_towers + 1):
            self.towers[tower_num] = self._init_tower_state(tower_num)
    
    def _init_tower_state(self, tower_num: int) -> TowerStateArrays:
        """各塔の状態変数を初期化"""
        # 外気温度
        temp_outside = self.sim_conds[tower_num]["DRUM_WALL_COND"]["temp_outside"]
        
        # 2D配列の初期化
        temp_2d = np.full((self.num_streams, self.num_sections), temp_outside, dtype=np.float64)
        
        # モル分率の初期化
        mf_co2_init = self.sim_conds[tower_num]["INFLOW_GAS_COND"]["mf_co2"]
        mf_n2_init = self.sim_conds[tower_num]["INFLOW_GAS_COND"]["mf_n2"]
        mf_co2_2d = np.full((self.num_streams, self.num_sections), mf_co2_init, dtype=np.float64)
        mf_n2_2d = np.full((self.num_streams, self.num_sections), mf_n2_init, dtype=np.float64)
        
        # 吸着量の初期化
        init_adsorp = self.sim_conds[tower_num]["PACKED_BED_COND"]["init_adsorp_amt"]
        adsorp_2d = np.full((self.num_streams, self.num_sections), init_adsorp, dtype=np.float64)
        
        # 伝熱係数の初期化
        heat_coef_2d = np.full((self.num_streams, self.num_sections), 1e-5, dtype=np.float64)
        heat_coef_wall_2d = np.full((self.num_streams, self.num_sections), 14.0, dtype=np.float64)
        
        # 流出CO2分圧の初期化
        total_press_init = self.sim_conds[tower_num]["INFLOW_GAS_COND"]["total_press"]
        outflow_pco2_2d = np.full((self.num_streams, self.num_sections), total_press_init, dtype=np.float64)
        
        # 1D配列の初期化
        temp_wall_1d = np.full(self.num_sections, temp_outside, dtype=np.float64)
        
        return TowerStateArrays(
            temp=temp_2d.copy(),
            temp_thermo=temp_2d.copy(),
            adsorp_amt=adsorp_2d,
            mf_co2=mf_co2_2d,
            mf_n2=mf_n2_2d,
            heat_t_coef=heat_coef_2d,
            heat_t_coef_wall=heat_coef_wall_2d,
            outflow_pco2=outflow_pco2_2d,
            temp_wall=temp_wall_1d,
            temp_lid_up=temp_outside,
            temp_lid_down=temp_outside,
            total_press=total_press_init,
            vacuum_amt_co2=0.0,
            vacuum_amt_n2=0.0
        )
    
    def get_tower(self, tower_num: int) -> TowerStateArrays:
        """指定された塔の状態変数を取得"""
        return self.towers[tower_num]
    
    def to_legacy_format(self) -> Dict:
        """既存のコードとの互換性のため、レガシー形式の辞書に変換"""
        variables_tower = {}
        
        for tower_num in range(1, self.num_towers + 1):
            tower = self.towers[tower_num]
            variables_tower[tower_num] = {}
            
            # 2D配列データ
            for var_name in ['temp', 'temp_thermo', 'adsorp_amt', 'mf_co2', 'mf_n2', 
                           'heat_t_coef', 'heat_t_coef_wall', 'outflow_pco2']:
                variables_tower[tower_num][var_name] = {}
                array = getattr(tower, var_name)
                for stream in range(1, self.num_streams + 1):
                    variables_tower[tower_num][var_name][stream] = {}
                    for section in range(1, self.num_sections + 1):
                        variables_tower[tower_num][var_name][stream][section] = array[stream-1, section-1]
            
            # 1D配列データ
            variables_tower[tower_num]['temp_wall'] = {}
            for section in range(1, self.num_sections + 1):
                variables_tower[tower_num]['temp_wall'][section] = tower.temp_wall[section-1]
            
            # 蓋の温度
            variables_tower[tower_num]['temp_lid'] = {
                'up': tower.temp_lid_up,
                'down': tower.temp_lid_down
            }
            
            # スカラー値
            variables_tower[tower_num]['total_press'] = tower.total_press
            variables_tower[tower_num]['vacuum_amt_co2'] = tower.vacuum_amt_co2
            variables_tower[tower_num]['vacuum_amt_n2'] = tower.vacuum_amt_n2
        
        return variables_tower
    
    def from_legacy_format(self, variables_tower: Dict):
        """レガシー形式の辞書から状態変数を更新"""
        for tower_num in range(1, self.num_towers + 1):
            if tower_num not in variables_tower:
                continue
                
            tower = self.towers[tower_num]
            legacy = variables_tower[tower_num]
            
            # 2D配列データの更新
            for var_name in ['temp', 'temp_thermo', 'adsorp_amt', 'mf_co2', 'mf_n2', 
                           'heat_t_coef', 'heat_t_coef_wall', 'outflow_pco2']:
                if var_name in legacy:
                    array = getattr(tower, var_name)
                    for stream in range(1, self.num_streams + 1):
                        if stream in legacy[var_name]:
                            for section in range(1, self.num_sections + 1):
                                if section in legacy[var_name][stream]:
                                    array[stream-1, section-1] = legacy[var_name][stream][section]
            
            # 1D配列データの更新
            if 'temp_wall' in legacy:
                for section in range(1, self.num_sections + 1):
                    if section in legacy['temp_wall']:
                        tower.temp_wall[section-1] = legacy['temp_wall'][section]
            
            # 蓋の温度の更新
            if 'temp_lid' in legacy:
                if 'up' in legacy['temp_lid']:
                    tower.temp_lid_up = legacy['temp_lid']['up']
                if 'down' in legacy['temp_lid']:
                    tower.temp_lid_down = legacy['temp_lid']['down']
            
            # スカラー値の更新
            if 'total_press' in legacy:
                tower.total_press = legacy['total_press']
            if 'vacuum_amt_co2' in legacy:
                tower.vacuum_amt_co2 = legacy['vacuum_amt_co2']
            if 'vacuum_amt_n2' in legacy:
                tower.vacuum_amt_n2 = legacy['vacuum_amt_n2']
    
    def update_from_calc_output(self, tower_num: int, mode: str, calc_output: Dict):
        """計算結果から状態変数を効率的に更新"""
        tower = self.towers[tower_num]
        
        # 温度データの一括更新
        for stream in range(1, self.num_streams + 1):
            for section in range(1, self.num_sections + 1):
                tower.temp[stream-1, section-1] = calc_output["heat"][stream][section]["temp_reached"]
                tower.temp_thermo[stream-1, section-1] = calc_output["heat"][stream][section]["temp_thermocouple_reached"]
                tower.adsorp_amt[stream-1, section-1] = calc_output["material"][stream][section]["accum_adsorp_amt"]
                tower.heat_t_coef[stream-1, section-1] = calc_output["heat"][stream][section]["hw1"]
                tower.heat_t_coef_wall[stream-1, section-1] = calc_output["heat"][stream][section]["u1"]
                tower.outflow_pco2[stream-1, section-1] = calc_output["material"][stream][section]["outflow_pco2"]
        
        # 壁面温度の更新
        for section in range(1, self.num_sections + 1):
            tower.temp_wall[section-1] = calc_output["heat_wall"][section]["temp_reached"]
        
        # 蓋の温度の更新
        tower.temp_lid_up = calc_output["heat_lid"]["up"]["temp_reached"]
        tower.temp_lid_down = calc_output["heat_lid"]["down"]["temp_reached"]
        
        # モル分率の更新
        if mode in ["初回ガス導入", "流通吸着_単独/上流", "バッチ吸着_上流", "均圧_加圧", 
                   "均圧_減圧", "バッチ吸着_下流", "流通吸着_下流"]:
            for stream in range(1, self.num_streams + 1):
                for section in range(1, self.num_sections + 1):
                    tower.mf_co2[stream-1, section-1] = calc_output["material"][stream][section]["outflow_mf_co2"]
                    tower.mf_n2[stream-1, section-1] = calc_output["material"][stream][section]["outflow_mf_n2"]
        elif mode == "真空脱着":
            for stream in range(1, self.num_streams + 1):
                for section in range(1, self.num_sections + 1):
                    tower.mf_co2[stream-1, section-1] = calc_output["mol_fraction"][stream][section]["mf_co2_after_vacuum"]
                    tower.mf_n2[stream-1, section-1] = calc_output["mol_fraction"][stream][section]["mf_n2_after_vacuum"]
        
        # 全圧の更新
        if mode in ["初回ガス導入", "バッチ吸着_上流", "均圧_加圧", "バッチ吸着_下流"]:
            tower.total_press = calc_output["pressure_after_batch_adsorption"]
        elif mode in ["均圧_減圧", "流通吸着_単独/上流", "流通吸着_下流"]:
            tower.total_press = calc_output["total_press"]
        elif mode == "真空脱着":
            tower.total_press = calc_output["pressure_after_desorption"]
        
        # 回収量の更新
        if mode == "真空脱着":
            tower.vacuum_amt_co2 = calc_output["accum_vacuum_amt"]["accum_vacuum_amt_co2"]
            tower.vacuum_amt_n2 = calc_output["accum_vacuum_amt"]["accum_vacuum_amt_n2"]
        else:
            tower.vacuum_amt_co2 = 0.0
            tower.vacuum_amt_n2 = 0.0
    
    def get_mean_temp(self, tower_num: int) -> float:
        """塔の平均温度を効率的に計算"""
        return np.mean(self.towers[tower_num].temp)
    
    def get_mean_mf_co2(self, tower_num: int) -> float:
        """塔の平均CO2モル分率を効率的に計算"""
        return np.mean(self.towers[tower_num].mf_co2)
    
    def get_mean_mf_n2(self, tower_num: int) -> float:
        """塔の平均N2モル分率を効率的に計算"""
        return np.mean(self.towers[tower_num].mf_n2)