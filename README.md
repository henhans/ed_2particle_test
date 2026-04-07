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
