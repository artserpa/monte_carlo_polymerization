import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.simulation.parameters import (
    kinetic_constants,
    initial_conditions,
    reactor_conditions,
    simulation_parameters,
    polymerization_parameters
)

def test_kinetic_constants_creation():
    """Test: create basic kinetic constants"""
    k = kinetic_constants(
        k_p = 1000.0,
        k_i = 500.0,
        k_d = 0.01,
        f = 0.7,
        k_tc = 1e7
    )
    assert k.k_p == 1000.0

def test_initial_conditions_creation():
    """Test: Create initial conditions"""
    ic = initial_conditions(
        monomer_concentration=8.0,
        initiator_concentration=0.01
    )
    assert ic.monomer_concentration == 8.0

def test_reactor_conditions_creation():
    """Test: Create reactor conditions"""
    rc = reactor_conditions(
        temperature=373.15,
        volume=2.e-16
    )
    assert rc.temperature == 373.15

def test_simulation_parameters_creation():
    """Test: Create simulation parameters"""
    sp = simulation_parameters(max_time=1000.0)
    assert sp.max_time == 1000.0

def test_polymerization_parameters_full():
    """Test: Create complete object and calculate molecules"""
    kinetic = kinetic_constants(k_p=1000.0, k_i=500, k_d=0.01, f=0.7, k_tc=1e7)
    initial = initial_conditions(monomer_concentration=8.0, initiator_concentration=0.01)
    reactor = reactor_conditions(temperature=373.15, volume=2.e-16)
    simulation = simulation_parameters(max_time=1000.0)

    params = polymerization_parameters(kinetic, initial, reactor, simulation)

    assert params.initial.n_monomer is not None
    assert params.initial.n_monomer > 0

def test_to_dict():
    """Test: Convert to dictionary"""
    kinetic = kinetic_constants(k_p=1000.0, k_i=500, k_d=0.01, f=0.7, k_tc=1e7)
    initial = initial_conditions(monomer_concentration=8.0, initiator_concentration=0.01)
    reactor = reactor_conditions(temperature=373.15, volume=2.e-16)
    simulation = simulation_parameters(max_time=1000.0)

    params = polymerization_parameters(kinetic, initial, reactor, simulation)
    params_dict = params.to_dict()

    assert 'kinetic_constants' in params_dict