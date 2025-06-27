from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Type
import adsorption_base_models
from sim_conditions import TowerConditions
from state_variables import StateVariables


@dataclass
class UnifiedMassBalanceResult:
    base_result: Dict[str, float]
    mole_fraction_result: Optional[Dict[str, float]] = None


class MassBalanceStrategy(ABC):
    @abstractmethod
    def calculate(self, stream: int, section: int, previous_result: Optional[Dict] = None) -> UnifiedMassBalanceResult:
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

    def calculate(self, stream: int, section: int, previous_result: Optional[Dict] = None) -> UnifiedMassBalanceResult:
        kwargs = {
            "tower_conds": self.tower_conds,
            "stream": stream,
            "section": section,
            "state_manager": self.state_manager,
            "tower_num": self.tower_num,
        }
        if section == 1:
            if self.external_inflow_gas is not None:
                kwargs["inflow_gas"] = self.external_inflow_gas[stream]
            if self.equalization_flow_rate is not None:
                kwargs["flow_amt_depress"] = self.equalization_flow_rate
            if self.residual_gas_composition is not None:
                kwargs["residual_gas_composition"] = self.residual_gas_composition
        else:
            kwargs["inflow_gas"] = previous_result
        result = adsorption_base_models.calculate_mass_balance_for_adsorption(**kwargs)
        return UnifiedMassBalanceResult(base_result=result)


class DesorptionStrategy(MassBalanceStrategy):
    def __init__(self, tower_conds, state_manager, tower_num, vacuum_pumping_results):
        self.tower_conds = tower_conds
        self.state_manager = state_manager
        self.tower_num = tower_num
        self.vacuum_pumping_results = vacuum_pumping_results

    def calculate(self, stream: int, section: int, previous_result: Optional[Dict] = None) -> UnifiedMassBalanceResult:
        base_result, mole_fraction_result = adsorption_base_models.calculate_mass_balance_for_desorption(
            tower_conds=self.tower_conds,
            stream=stream,
            section=section,
            state_manager=self.state_manager,
            tower_num=self.tower_num,
            vacuum_pumping_results=self.vacuum_pumping_results,
        )
        return UnifiedMassBalanceResult(base_result=base_result, mole_fraction_result=mole_fraction_result)


class ValveClosedStrategy(MassBalanceStrategy):
    def __init__(self, state_manager, tower_num):
        self.state_manager = state_manager
        self.tower_num = tower_num

    def calculate(self, stream: int, section: int, previous_result: Optional[Dict] = None) -> UnifiedMassBalanceResult:
        result = adsorption_base_models.calculate_mass_balance_for_valve_closed(
            stream=stream,
            section=section,
            state_manager=self.state_manager,
            tower_num=self.tower_num,
        )
        return UnifiedMassBalanceResult(base_result=result)
