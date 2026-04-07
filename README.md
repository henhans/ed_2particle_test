# ed_2particle_test

Simple utility to generate a fermionic Fock basis for `N` orbitals.

It also includes:

- fermionic creation/annihilation operators on bit-encoded states
- Anderson impurity Hamiltonian builder
- real-symmetric diagonalization and eigenbasis operator transforms

## Usage

Generate all sectors:

```bash
python fock_basis.py 3
```

Generate fixed-particle sector:

```bash
python fock_basis.py 4 --num-particles 2
```

## Plotting scripts

Install plotting dependency:

```bash
pip install matplotlib
```

Generate a single-particle imaginary-time Green's function plot:

```bash
python scripts/plot_single_particle_green.py --beta 10 --U 4 --mu 2 --output g_tau.png
```

Generate the six time-order contribution plot for a two-particle Green's function point:

```bash
python scripts/plot_two_particle_contributions.py --beta 10 --U 4 --mu 2 --tau1 8 --tau2 5 --tau3 2 --output g2_contributions.png
```

Generate a single-frequency spin susceptibility comparison plot between
the finite double-Matsubara sum and the single-frequency Lehmann formula:

```bash
python scripts/plot_single_frequency_spin_susceptibility.py --beta 10 --U 4 --mu 2 --omega-max 4 --nu-cutoff 4 --nup-cutoff 4 --output chi_spin_single_frequency.png
```
