import numpy as np
from dataclasses import dataclass
from typing import Tuple
from numba import jit

from .parameters import(
    kinetic_constants, initial_conditions, reactor_conditions,
    simulation_parameters, polymerization_parameters, NA, R_GAS
)

@dataclass
class styrene_kinetic_constants(kinetic_constants):
    """Specific kinetic constants for styrene with bifunctional initiator"""

    # Additional Arrhenius parameters
    k_th: float = 0.0       # Thermal initiation (L/mol.min)
    k_d2: float = 0.0       # Second decomposition (1/min)

    # Gel effect parameters
    gel_A1: float = 0.0
    gel_A2: float = 0.0
    gel_A3: float = 0.0

    @classmethod
    def from_temperature(cls, temperature: float, f: float = 0.7):
        """
        Calculates the kinetic constants using the temperature (Arrhenius)

        Args
        ----
            temperature : float
                Temperature in Kelvin
            f : float
                Initiator's efficiency

        Returns
        -------
            styrene_kinetic_constants with the calculated values
        """

        # Thermal initiation
        k_th = 1.314e7 * np.exp(-27440.5 / (R_GAS * temperature))

        # Propagation
        k_p = 6.128e8 * np.exp(-7067.8 / (R_GAS * temperature))

        # Initiation
        k_i = k_p

        # Transfer to monomer
        k_fm = 2.319e8 * np.exp(-12000 / (R_GAS * temperature))

        # Termination by combination
        k_tc0 = 7.53e10 * np.exp(-1680 / (R_GAS * temperature))

        # Bifunctional decomposition
        k_d1 = 1.269e18 * np.exp(-35662.08 / (R_GAS * temperature))
        k_d2 = 1.09e21 * np.exp(-42445 / (R_GAS * temperature))

        gel_A1 = 2.57 - 5.05e-3 * temperature
        gel_A2 = 9.56 - 1.76e-2 * temperature
        gel_A3 = -3.03 + 7.85e-3 * temperature

        return cls(
            k_p = k_p,
            k_i = k_i,
            k_d = k_d1,
            f = f,
            k_tc = k_tc0,
            k_td = 0.0,
            k_tr_m = k_fm,
            k_tr_s = 0.0,
            k_tr_p = 0.0,
            k_th = k_th,
            k_d2 = k_d2,
            gel_A1 = gel_A1,
            gel_A2 = gel_A2,
            gel_A3 = gel_A3 
        )
    
    def to_stochastic_array(self, volume: float) -> np.ndarray:
        """
        Converts constants to NumPy array (Numba)

        Args
        ----
            volume : float
                Control volume (L)
        
        Returns
        -------
            Array : np.ndarray
                Array with constants converted to stochastic values
        """

        factor = NA * volume

        # Stochastic values
        k_th_MC = 6 * self.k_th / (factor ** 2)
        k_p_MC = self.k_p / factor
        k_i1_MC = self.k_i / factor
        k_i2_MC = self.k_i / factor
        k_fm_MC = self.k_tr_m / factor
        k_tc0_MC = 2 * self.k_tc / factor
        k_d1_MC = self.k_d
        k_d2_MC = self.k_d2

        return np.ndarray([
            k_th_MC, k_p_MC, k_i1_MC, k_i2_MC, k_fm_MC,
            k_tc0_MC, k_d1_MC, k_d2_MC, self.f,
            self.gel_A1, self.gel_A2, self.gel_A3
        ], dtype=np.float64)
    
def create_styrene_parameters(
        temperature_C: float = 100.0,
        volume: float = 2.e-16,
        monomer_conc: float = 8.0785,
        initiator_conc: float = 0.01,
        f: float = 0.7,
        max_time: float = 10000.0,
        conversion_target: float = 0.9,
        random_seed: int = 42
) -> Tuple[polymerization_parameters, np.ndarray, np.ndarray]:
    """
    Create complete array of parameters for styrene polymerization
    
    Args:
        temperature_C: Temperature in Celsius
        volume: Control volume (L)
        monomer_conc: Initial monomer concentration (mol/L)
        initiator_conc: Initial initiator concentration (mol/L)
        f: Initiator's efficiency
        max_time: Max simulation time (min)
        conversion_target: Max conversion threshold (0-1)
        random_seed: Seed for reproducibility
    
    Returns:
        Tuple (params, X_initial, k_constants)
        - params: polymerization_parameters
        - X_initial: Initial state [IB, R1, R2, M, P1, P2, D1, D2, D3]
        - k_constants: Stochastic constants array
    """

    temperature_K = temperature_C + 273.15

    # Create the kinetic constants
    kinetic = styrene_kinetic_constants.from_temperature(temperature_K, f)

    # Initial conditions
    initial = initial_conditions(
        monomer_concentration=monomer_conc,
        initiator_concentration=initiator_conc
    )

    # Reactor conditions
    reactor = reactor_conditions(
        temperature=temperature_K,
        volume=volume,
    )

    # Simulation parameters
    simulation = simulation_parameters(
        max_time=max_time,
        conversion_threshold=conversion_target,
        random_seed=random_seed,
        save_interval=100000
    )

    # Create object for the parameters
    params = polymerization_parameters(kinetic, initial, reactor, simulation)

    # Initial state: [IB, R1, R2, M, P1_count, P2_count, D1, D2, D3]
    X_initial = np.ndarray([
        params.initial.n_initiator, # Bifunctional initiator
        0,                          # Primary radical
        0,                          # Primary radical with peroxide
        params.initial.n_monomer,   # Monomer
        0,                          # Counter for type 1 radicals
        0,                          # Counter for type 2 radicals
        0,                          # Dead polymer w/o peroxide
        0,                          # Dead polymer w/ 1 peroxide
        0                           # Dead polymer w/ 2 peroxide
    ], dtype=np.int64)

    # Stochsatic constants
    k_constants = kinetic.to_stochastic_array(reactor.volume)

    return params, X_initial, k_constants

@jit(nopython=True)
def calculate_gel_effect(conversion: float, A1: float, A2: float, A3: float) -> float:
    """
    Calculates the gel effect factor based on the conversion

    Args
    ----
        conversion : float
            current coversion (0-1)
        A1, A2, A3 : float
            Gel effect model parameters

    Returns
    -------
        Gel effect factor : float
    """
    x = conversion
    return np.exp(2.0 * (A1 * x + A2 * x ** 2 + A3 * x ** 3))

@jit(nopython=True)
def calculate_reaction_rates_styrene(
    X: np.ndarray,
    k_constants: np.ndarray,
    M0: int
) -> np.ndarray:
    """
    Calculates rates for the 13 reactions for bifunctional styrene

    Args
        X : np.ndarray
            Current state of the reaction
        k_constants : np.ndarray
            Array with kinetic constants
        M0 : int
            Initial number of monomer molecules
    
    Returns
    -------
        rates : np.ndarray
            Array with reaction rates for all the reactions
    """

    # Unpack state
    IB = X[0]   # Bifunctional initiator
    R1 = X[1]   # Type 1 primary radical
    R2 = X[2]   # Type 2 primary radical
    M = X[3]    # Monomer
    P1 = X[4]   # Type 1 polymer radical
    P2 = X[5]   # Type 2 polymer radical
    D1 = X[6]   # Dead polymer (0 peroxide)
    D2 = X[7]   # Dead polymer (1 peroxide)
    D3 = X[8]   # Dead polymer (2 peroxide)

    # Unpack constants
    k_th = k_constants[0]
    k_p = k_constants[1]
    k_i1 = k_constants[2]
    k_i2 = k_constants[3]
    k_fm = k_constants[4]
    k_tc0 = k_constants[5]
    k_d1 = k_constants[6]
    k_d2 = k_constants[7]
    f = k_constants[8]
    A1 = k_constants[9]
    A2 = k_constants[10]
    A3 = k_constants[11]

    if M0 > 0:
        conversion = 1.0 - float(M) / float(M0)
    else:
        conversion = 0.0
    
    g = calculate_gel_effect(conversion, A1, A2, A3)
    k_tc = k_tc0 * g

    # Initialize the rates' array
    rates = np.zeros(13, dtype=np.float64)

    # Reaction 0: Propagation P1 + M -> P1'
    rates[0] = k_p * M * P1

    # Reaction 1: Propagation P2 + M -> P2'
    rates[1] = k_p * M * P2

    # Reaction 2: Initiator decompositon: IB -> R1 + R2
    rates[2] = 2.0 * k_d1 * IB

    # Reaction 3: Chemical initiation: R1 + M -> P1
    rates[3] = k_i1 * R1 * M

    # Reaction 4: Chemical initiation: R2 + M -> P2
    rates[4] = k_i2 * R2 * M

    # Reaction 5: Termination by combination: P1 + P2 -> D1
    rates[5] = 0.5 * k_tc * P1 * P2

    # Reaction 6: Transfer to monomer P1 + M -> D1 + P1
    rates[6] = k_fm * M * P1

    # Reaction 7: Termination by combination: P1 + P1 -> D1
    if P1 > 1:
        rates[7] = 0.25 * k_tc * P1 * (P1 - 1)

    # Reaction 8: Transfer to monomer P2 + M -> D2 + P1
    rates[8] = k_fm * M * P2

    # Reaction 9: Second decomposition D2 -> D1 + R1
    rates[9] = k_d2 * D2

    # Reaction 10: Termination by combination P2 + P2 -> D3
    if P2 > 1:
        rates[10] = 0.25 * k_tc * P2 * (P2 - 1)

    # Reaction 11: Thermal initiation 3M -> 2P1
    if M > 2:
        rates[11] = k_th * M * (M - 1) * (M - 2) / 6.0

    # Reaction 12: Second decomposition D3 -> D2 + R1
    rates[12] = 2.0 * k_d2 * D3
