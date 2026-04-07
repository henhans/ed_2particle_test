# ed_2particle_test

Simple utility to generate a fermionic Fock basis for `N` orbitals.

## Usage

Generate all sectors:

```bash
python fock_basis.py 3
```

Generate fixed-particle sector:

```bash
python fock_basis.py 4 --num-particles 2
```
