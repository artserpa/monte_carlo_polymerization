import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from numba import jit

# Physical constants
NA = 6.022e23   # Avogrado's Number (1/mol)
R_GAS = 1.987   # Gas constant (cal/mol/K)

@dataclass
class kinetic_constants:
    """Kinetic constants of the polymerization reaction (deterministic)"""

    # Basic propagation constant
    k_p: float          # Propagation (L/mol/min)

    # Initiation constants
    k_i: float          # Initiation (L/mol/min)
    k_d: float          # Initiator's decomposition (1/min)
    f: float            # Initiator's efficiency

    # Termination constants
    k_tc: float         # Termination by combination (L/mol/min)
    k_td: float = 0.0   # Termination by desproportion (L/mol/min)

    # Constants of chain transfer
    k_tr_m: float = 0.0 # Transfer to monomer (L/mol/min)
    k_tr_s: float = 0.0 # Transfer to solvent (L/mol/min)
    k_tr_p: float = 0.0 # Transfer to polymer (L/mol/min)

    def __post_init__(self):
        """Parameter validation"""
        if self.f < 0 or self.f > 1:
            raise ValueError("Initiator's efficiency must be between 0 and 1")
        if any(k < 0 for k in [self.k_p, self.k_i, self.k_d, self.k_tc]):
            raise ValueError("Kinetic constants must be positive")
        


@dataclass
class initial_conditions:
    """Initial conditions of the reaction"""

    # Initial concentrations (mol/L)
    monomer_concentration: float        # [M]
    initiator_concentration: float      # [I]
    solvent_concentration: float = 0.0  # [S]

    # Initial number of molecules
    n_monomer: Optional[int] = None
    n_initiator: Optional[int] = None
    n_solvent: Optional[int] = None

    def __post_init__(self):
        """Initial conditions validation"""
        if self.monomer_concentration <= 0:
            raise ValueError("Monomer concentration must be positive")
        if self.initiator_concentration <= 0:
            raise ValueError("Initiator concentration must be positive")

@dataclass
class reactor_conditions:
    """Reactor conditions"""
    temperature: float      # Temperature (K)
    volume: float           # Control volume (L)
    pressure: float = 1.0   # Pressure (atm)

    def __post_init__(self):
        """Reactor conditions validation"""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if self.volume <= 0:
            raise ValueError("Control volume must be positive")
        if self.pressure <= 0:
            raise ValueError("Pressure must be positive")
    
@dataclass
class simulation_parameters:
    """Monte Carlo simulation parameters"""

    max_time: float                     # Max simulation time (min)
    max_iterations: int = int(1e8)      # Max number of iterations
    random_seed: Optional[int] = None   # Seed for reproducibility
    conversion_threshold: float = 0.8   # Max conversion

    # Output parameters
    save_interval: int = 10000  # Save state at N iterations

    def __post_init__(self):
        """Simulation parameters validation"""
        if self.max_time <= 0:
            raise ValueError("Max time must be positive")
        if self.conversion_threshold < 0 and self.conversion_threshold > 1:
            raise ValueError("Conversion must be between 0 and 1")

class polymerization_parameters:
    """Main class to handle all parameters"""

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

        # Calculates the number of molecules
        self._calculate_molecule_numbers()

    def _calculate_molecule_numbers(self):
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
    
    def to_dict(self) -> Dict:
        """Convert parameters to a dictionary"""
        return {
            'kinetic_constants': self.kinetic.__dict__,
            'stochastic_constants': self.stochastic.__dict__,
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
            f"  N_monomer={self.initial.n_monomer:.2e}, "
            f"  N_initiator={self.initial.n_initiator:.2e}\n"
            f")"
        )