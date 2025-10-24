from src.simulation.parameters import get_default_parameters

if __name__ == "__main__":
    params = get_default_parameters()
    print(params)
    print(f"\nNumber of monomer molecules: {params.initial.n_monomer:.2e}")
    print(f"\nSum of kinetic constants: {params.get_total_rate_constant():.2e}")