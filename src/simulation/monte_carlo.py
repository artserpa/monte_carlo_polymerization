import numpy as np
from numba import jit
from numba.typed import List
from typing import Tuple

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

    return rates

@jit(nopython=True)
def run_monte_carlo_styrene(
    X_initial: np.ndarray,
    k_constants: np.ndarray,
    conversion_target: float,
    max_iterations: int,
    save_interval: int,
    random_seed: int
) -> Tuple:
    """
    Executes the Monte Carlo Simulation for styrene polymerization with bifunctional initiator
    Implements the Gillespie algorithm

    Args
    ----
        X_initial : np.ndarray
            Initial state
        k_constants : np.ndarray
            Array of stochastic constants
        conversion_target : float
            Target conversion (0-1)
        max_iterations : int
            Max number of iterations
        save_interval : int
            Interval for snapshot
        random_seed : int
            Seed for reproducibility
    Returns
    -------
        Tuple
            (L1, L2, L3, time_hist, conv_hist, M_hist, IB_hist)
    """
    # Sets up seed
    np.random.seed(random_seed)

    # Initial state
    X = X_initial.copy()
    M0 = X[3]
    t = 0.0
    conversion = 0.0

    f = k_constants[8]

    P1 = List()
    P2 = List()

    L1 = List()
    L2 = List()
    L3 = List()

    time_hist = List()
    conv_hist = List()
    M_hist = List()
    IB_hist = List()

    iteration = 0
    while conversion < conversion and iteration < max_iterations:
        rates = calculate_reaction_rates_styrene(X, k_constants, M0)
        total_rate = np.sum(rates)

        if total_rate <= 0:
            break

        r1 = np.random.rand()
        dt = -np.log(r1) / total_rate
        t += dt

        r2 = np.random.rand()
        threshold = r2 * total_rate
        cumulative = 0.0
        reaction = -1

        for i in range(len(rates)):
            cumulative += rates[i]
            if cumulative >= threshold:
                reaction = i
                break

        if reaction == 0:
            if X[4] > 0:
                m = int(np.random.rand() * X[4])
                X[3] -= 1
                P1[m] += 1

        elif reaction == 1:
            if X[5] > 0:
                m = int(np.random.rand() * X[5])
                X[3] -= 1
                P2[m] += 1

        elif reaction == 2:
            X[0] -= 1
            r3 = np.random.rand()
            if r3 <= f:
                X[1] += 1
                X[2] += 1

        elif reaction == 3:
            X[1] -= 1
            X[3] -= 1
            X[4] += 1
            P1.append(np.int64(1))
        
        elif reaction == 4:
            X[1] -= 1
            X[3] -= 1
            X[5] += 1
            P2.append(np.int64(1))
            
        elif reaction == 5:
            if X[4] > 0 and X[5] > 0:
                m = int(np.random.rand() * X[4])
                n = int(np.random.rand() * X[5])
                X[4] -= 1
                X[5] -= 1
                X[7] += 1
                L2.append(P1[m] + P2[n])
                P1.pop(m)
                P2.pop(n)
        
        elif reaction == 6:
            if X[4] > 0:
                m = int(np.random.rand() * X[4])
                X[3] -= 1
                X[6] += 1
                L1.append(P1[m])
                P1[m] = np.int64(1)

        elif reaction == 7:
            if X[4] > 1:
                m = int(np.random.rand() * X[4])
                n = int(np.random.rand() * X[4])
            while m == n:
                if m == 0:
                    n += 1
                else:
                    n -= 1
            X[4] -= 2
            X[6] += 1
            L1.append(P1[m] + P1[n])
            if m > n:
                P1.pop(m)
                P1.pop(n)
            else:
                P1.pop(n)
                P1.pop(m)
        
        elif reaction == 8:
            if X[5] > 0:
                m = int(np.random.rand() * X[5])
                X[3] -= 1
                X[7] += 1
                L2.append(P2[m])
                X[5] -= 1
                X[4] += 1
                P1.append(np.int64(1))
                P2.pop(m)

        elif reaction == 9:
            if X[7] > 0 and len(L2) > 0:
                m = int(np.random.rand() * len(L2))
                r4 = np.random.rand()
                X[7] -= 1
                X[1] += 1
                X[4] += 1
                P1.append(L2[m])
                L2.pop(m)

                if r4 > f:
                    L1.append(P1[X[4] - 1])
                    P1.pop(X[4] - 1)
                    X[1] -= 1
                    X[4] -= 1
                    X[6] += 1
        
        elif reaction == 10:
            if X[5] > 1:
                m = int(np.random.rand() * X[5])
                n = int(np.random.rand() * X[5])
            while m == n:
                if m == 0:
                    n += 1
                else:
                    n -= 1
            X[5] -= 2
            X[8] += 1
            L3.append(P2[m] + P2[n])
            if m > n:
                P2.pop(m)
                P2.pop(n)
            else:
                P2.pop(n)
                P2.pop(m)

        elif reaction == 11:
            X[3] -= 3
            X[4] += 2
            P1.append(np.int64(1))
            P1.append(np.int64(1))
        
        elif reaction == 12:
            if X[8] > 0 and len(L3) > 0:
                m = int(np.random.rand() * len(L3))
                r5 = np.random.rand()
                X[8] -= 1
                X[1] += 1
                X[5] += 1
                P2.append(L3[m])
                L3.pop(m)

                if r5 > f:
                    X[7] += 1
                    L2.append(P2[X[5] - 1])
                    P2.pop(X[5] - 1)
                    X[1] -= 1
                    X[5] -= 1

        if M0 > 0:
            conversion = 1.0 - float(X[3]) / float(M0)

        if iteration % save_interval == 0:
            time_hist.append(t)
            conv_hist.append(conversion)
            M_hist.append(int(X[3]))
            IB_hist.append(int(X[0]))

        iteration += 1

    return (
        L1,
        L2,
        L3,
        time_hist,
        conv_hist,
        M_hist,
        IB_hist
    )