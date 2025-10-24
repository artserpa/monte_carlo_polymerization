import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class kinetic_constants:

    k_p: float

    k_i: float
    k_d: float

    f: float

    k_tc: float
    k_td: float

    def __post_init__(self):
        if self.f < 0 or self.f > 1:
            raise ValueError("Initiator's efficiency must be between 0 and 1")
        if any(k < 0 for k in [self.k_p, self.k_i, self.k_d, self.k_tc, self.k_td]):
            raise ValueError("Kinetic constants must be positive")
        
@dataclass
class initial_conditions:

    monomer_concentration: float
    initiator_concentration: float
    solvent_concentration: float = 0.0

    n_monomer: Optional[int] = None
    n_initiator: Optional[int] = None
    n_solvent: Optional[int] = None

    def __post_init__(self):
        if self.monomer_concentration <= 0:
            raise ValueError("Monomer concentration must be positive")
        if self.initiator_concentration <= 0:
            raise ValueError("Initiator concentration must be positive")

@dataclass
class reactor_conditions:

    temperature: float
    volume: float
    pressure: float = 1.0

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.volume <= 0:
            raise ValueError("Control volume must be positive")
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
    
@dataclass
class simulation_parameters:

    max_time: float
    time_step: float = None
    max_iterations: int = 1e6
    random_seed: Optional[int] = None

    conversion_threshold: float = 0.8

    def __post_init__(self):
        if self.max_time <= 0:
            raise ValueError("Max time must be positive")
        if self.conversion_threshold < 0 and self.conversion_threshold > 1:
            raise ValueError("Conversion must be between 0 and 1")

class polymerization_parameters:

    def __init__(
            self,
            kinetic_constants: kinetic_constants,
            initial_conditions: initial_conditions,
            reactor_conditions: reactor_conditions,
            simulation_parameters: simulation_parameters
    ):
        
        self.kinetic = kinetic_constants
        self.initial = initial_conditions
        self.reactor = reactor_conditions
        self.simulation = simulation_parameters

        self._calculate_molecule_numbers()

    def _calculate_molecule_numbers(self):

        NA = 6.022e23
        V = self.reactor.volume

        if self.initial.n_monomer is None:
            self.initial.n_monomer = int(
                self.initial.monomer_concentration * V * NA
            )
        
        if self.initial.n_initiator is None:
            self.initial.n_initiator = int(
                self.initial.initiator_concentration * V * NA
            )

        if self.initial.n_solvent is None and self.initial.solvent_concentration > 0:
            self.initial.n_solvent = int(
                self.initial.solvent_concentration * V * NA
            )

    def get_total_rate_constant(self) -> float:

        return (
            self.kinetic.k_p + self.kinetic.k_i + self.kinetic.k_d + 
            self.kinetic.k_tc + self.kinetic.k_td
        )
    
    def to_dict(self) -> Dict:

        return {
            'kinetic_constants': self.kinetic.__dict__,
            'initial_conditions': self.initial.__dict__,
            'reactor_conditions': self.reactor.__dict__,
            'simulation_parameters': self.simulation.__dict__
        }
    
    def __repr__(self) -> str:

        return(
            f"polymerization_parameters(\n"
            f"  T={self.reactor.temperature} K, "
            f"  V={self.reactor.volume} L,\n"
            f"  [M]={self.initial.monomer_concentration} mol/L, "
            f"  [I]={self.initial.initiator_concentration} mol/L\n"
            f")"
        )
        
def get_default_parameters() -> polymerization_parameters:

    kinetic = kinetic_constants(
        k_p=1000.0,
        k_i=0.01,
        k_d=1.e-5,
        f=0.7,
        k_tc=1e7,
        k_td=1e6
    )

    initial = initial_conditions(
        monomer_concentration=8.0,
        initiator_concentration=0.05
    )

    reactor = reactor_conditions(
        temperature=333.15,
        volume=1e-18
    )

    simulation = simulation_parameters(
        max_time=40000,
        max_iterations=1e8,
        random_seed=42,
        conversion_threshold=0.8
    )

    return polymerization_parameters(
        kinetic, initial, reactor, simulation
    )