import numpy as np
import sys
import os

from src.simulation.polymerization import (
    styrene_kinetic_constants,
    create_styrene_parameters
)

def test_styrene_constants_from_temperature():
    """Test: Calculate constants from temperature"""
    k = styrene_kinetic_constants.from_temperature(373.15, f=0.7)

    assert k.k_p > 0
    assert k.k_tc > 0 
    assert k.k_d > 0
    assert k.k_th > 0 
    assert k.f == 0.7

    print("Arrhenius constant calculated correctly")

def test_to_stochastic_array():
    """Test: Convert to stochastic array"""
    k = styrene_kinetic_constants.from_temperature(373.15)
    k_array = k.to_stochastic_array(volume=2.e-16)

    assert len(k_array) == 12
    assert isinstance(k_array, np.ndarray)
    assert np.all(k_array[0:8] >= 0),   "Kinetic constants must be positive"
    assert 0 <= k_array[8] <= 1,        "Efficiency must be between 0 and 1"

    print("Conversion to stochastic array working")

def test_create_styrene_parameters():
    """Test: Create complete parameters for styrene"""
    params, X_initial, k_constants = create_styrene_parameters(
        temperature_C=100.0,
        volume=2.e-16,
        monomer_conc=8.0,
        initiator_conc=0.01
    )

    assert params is not None
    assert len(X_initial) == 9
    assert len(k_constants) == 12

    assert X_initial[0] > 0
    assert X_initial[3] > 0

    print("create_styrene_parameters is working")
    print(f"  - Initial monomer: {X_initial[3]}")
    print(f"  - Initial initiator: {X_initial[0]}")

def test_arrhenius_temperature_dependence():
    """Test: Constant values increase with temperature"""
    k1 = styrene_kinetic_constants.from_temperature(323.15)
    k2 = styrene_kinetic_constants.from_temperature(373.15)

    assert k2.k_p > k1.k_p

    print("Correct behavior for the Arrhenius equation")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING: polymerization.py")
    print("="*60 + "\n")
    
    test_styrene_constants_from_temperature()
    test_to_stochastic_array()
    test_create_styrene_parameters()
    test_arrhenius_temperature_dependence()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60 + "\n")