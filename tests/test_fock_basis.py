from fock_basis import generate_fock_basis


def test_full_basis_size():
    basis = generate_fock_basis(4)
    assert len(basis) == 16
    assert basis[0] == (0, 0, 0, 0)
    assert basis[-1] == (1, 1, 1, 1)


def test_fixed_particle_sector():
    basis = generate_fock_basis(4, num_particles=2)
    assert len(basis) == 6
    assert (1, 1, 0, 0) in basis
    assert (0, 1, 0, 1) in basis


def test_invalid_inputs():
    for kwargs in (
        {"num_orbitals": -1},
        {"num_orbitals": 3, "num_particles": -1},
        {"num_orbitals": 3, "num_particles": 4},
    ):
        try:
            generate_fock_basis(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
