from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict
from . import adsorption_base_models
from config.sim_conditions import TowerConditions
from ..state import StateVariables, MaterialBalanceResult, GasFlow


@dataclass
class MassBalanceCalculationResult:
    """マスバランス計算の統一的な結果"""

    material_balance: MaterialBalanceResult
    mole_fraction_data: Optional[Dict[str, float]] = None

    def has_mole_fraction_data(self) -> bool:
        return self.mole_fraction_data is not None


class MassBalanceStrategy(ABC):
    @abstractmethod
    def calculate(
        self, stream: int, section: int, previous_result: Optional[MaterialBalanceResult] = None
    ) -> MassBalanceCalculationResult:
        """統一された戻り値型を返す"""
        pass

    @abstractmethod
    def supports_mole_fraction(self) -> bool:
        """この戦略がモル分率計算をサポートするかどうか"""
        pass


class AdsorptionStrategy(MassBalanceStrategy):
    def __init__(
        self,
        tower_conds: TowerConditions,
        state_manager: StateVariables,
        tower_num: int,
        external_inflow_gas=None,
        equalization_flow_rate=None,
        residual_gas_composition=None,
    ):
        self.tower_conds = tower_conds
        self.state_manager = state_manager
        self.tower_num = tower_num
        self.external_inflow_gas = external_inflow_gas
        self.equalization_flow_rate = equalization_flow_rate
        self.residual_gas_composition = residual_gas_composition

    def calculate(
        self, stream: int, section: int, previous_result: Optional[MaterialBalanceResult] = None
    ) -> MassBalanceCalculationResult:
        kwargs = {
            "tower_conds": self.tower_conds,
            "stream": stream,
            "section": section,
            "state_manager": self.state_manager,
            "tower_num": self.tower_num,
        }
        if section == 1:
            if self.external_inflow_gas is not None:
                # 辞書からGasFlowオブジェクトに変換
                inflow_dict = self.external_inflow_gas[stream]
                # 辞書がGasFlowオブジェクトかチェック
                if isinstance(inflow_dict, GasFlow):
                    kwargs["inflow_gas"] = inflow_dict
                else:
                    # 辞書の場合は変換
                    kwargs["inflow_gas"] = GasFlow(
                        co2_volume=inflow_dict.get("outlet_co2_volume", inflow_dict.get("co2_volume", 0)),
                        n2_volume=inflow_dict.get("outlet_n2_volume", inflow_dict.get("n2_volume", 0)),
                        co2_mole_fraction=inflow_dict.get(
                            "outlet_co2_mole_fraction", inflow_dict.get("co2_mole_fraction", 0)
                        ),
                        n2_mole_fraction=inflow_dict.get(
                            "outlet_n2_mole_fraction", inflow_dict.get("n2_mole_fraction", 0)
                        ),
                    )
            if self.equalization_flow_rate is not None:
                kwargs["equalization_flow_rate"] = self.equalization_flow_rate
            if self.residual_gas_composition is not None:
                kwargs["residual_gas_composition"] = self.residual_gas_composition
        else:
            # previous_resultからGasFlowオブジェクトを作成
            kwargs["inflow_gas"] = GasFlow(
                co2_volume=previous_result.outlet_gas.co2_volume,
                n2_volume=previous_result.outlet_gas.n2_volume,
                co2_mole_fraction=previous_result.outlet_gas.co2_mole_fraction,
                n2_mole_fraction=previous_result.outlet_gas.n2_mole_fraction,
            )
        result = adsorption_base_models.calculate_mass_balance_for_adsorption(**kwargs)
        # Convert dict result to MaterialBalanceResult object
        return MassBalanceCalculationResult(material_balance=result)

    def supports_mole_fraction(self) -> bool:
        return False


class DesorptionStrategy(MassBalanceStrategy):
    def __init__(self, tower_conds, state_manager, tower_num, vacuum_pumping_results):
        self.tower_conds = tower_conds
        self.state_manager = state_manager
        self.tower_num = tower_num
        self.vacuum_pumping_results = vacuum_pumping_results

    def calculate(
        self, stream: int, section: int, previous_result: Optional[MaterialBalanceResult] = None
    ) -> MassBalanceCalculationResult:
        material_balance_result, mole_fraction_result = adsorption_base_models.calculate_mass_balance_for_desorption(
            tower_conds=self.tower_conds,
            stream=stream,
            section=section,
            state_manager=self.state_manager,
            tower_num=self.tower_num,
            vacuum_pumping_results=self.vacuum_pumping_results,
        )
        return MassBalanceCalculationResult(
            material_balance=material_balance_result, mole_fraction_data=mole_fraction_result
        )

    def supports_mole_fraction(self) -> bool:
        return True


class ValveClosedStrategy(MassBalanceStrategy):
    def __init__(self, state_manager, tower_num):
        self.state_manager = state_manager
        self.tower_num = tower_num

    def calculate(
        self, stream: int, section: int, previous_result: Optional[MaterialBalanceResult] = None
    ) -> MassBalanceCalculationResult:
        result = adsorption_base_models.calculate_mass_balance_for_valve_closed(
            stream=stream,
            section=section,
            state_manager=self.state_manager,
            tower_num=self.tower_num,
        )
        return MassBalanceCalculationResult(material_balance=result)

    def supports_mole_fraction(self) -> bool:
        return False
