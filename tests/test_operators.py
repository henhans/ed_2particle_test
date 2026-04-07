import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from operators import (
    annihilate,
    annihilation_operator_matrix,
    anderson_impurity_hamiltonian,
    creation_operator_matrix,
    create,
    diagonalize_symmetric,
    lehmann_green_iwn,
    lehmann_green_tau,
    two_particle_green_iwn_inu_inup,
    two_particle_green_tau,
    two_particle_green_tau_contributions,
    transform_to_eigenbasis,
)


def test_create_empty_orbital_even_parity():
    # state 0b0100, create at orb=1; left side has 0 occupied orbitals
    new_state, sign = create(0b0100, 1)
    assert new_state == 0b0110
    assert sign == 1


def test_create_empty_orbital_odd_parity():
    # state 0b0101, create at orb=1; left side has one occupied orbital (orb=0)
    new_state, sign = create(0b0101, 1)
    assert new_state == 0b0111
    assert sign == -1


def test_create_on_occupied_orbital_is_zero():
    new_state, sign = create(0b0101, 0)
    assert new_state is None
    assert sign == 0


def test_annihilate_occupied_orbital_even_parity():
    # state 0b1110, annihilate at orb=1; left side has 0 occupied orbitals
    new_state, sign = annihilate(0b1110, 1)
    assert new_state == 0b1100
    assert sign == 1


def test_annihilate_occupied_orbital_odd_parity():
    # state 0b0111, annihilate at orb=2; left side has two occupied orbitals -> +1
    new_state, sign = annihilate(0b0111, 2)
    assert new_state == 0b0011
    assert sign == 1


def test_annihilate_occupied_orbital_odd_single_left():
    # state 0b0011, annihilate at orb=1; left side has one occupied orbital -> -1
    new_state, sign = annihilate(0b0011, 1)
    assert new_state == 0b0001
    assert sign == -1


def test_annihilate_empty_orbital_is_zero():
    new_state, sign = annihilate(0b0100, 0)
    assert new_state is None
    assert sign == 0


@pytest.mark.parametrize("func", [create, annihilate])
def test_negative_orbital_raises(func):
    with pytest.raises(ValueError):
        func(0b1, -1)


def test_anderson_impurity_hamiltonian_default_basis_diagonal():
    ham = anderson_impurity_hamiltonian(U=4.0, mu=1.5)
    assert ham == [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -1.5, 0.0, 0.0],
        [0.0, 0.0, -1.5, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_anderson_impurity_hamiltonian_custom_basis_ordering():
    basis = [0b11, 0b00, 0b10, 0b01]
    ham = anderson_impurity_hamiltonian(U=3.0, mu=0.5, basis=basis)
    assert ham == [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, 0.0],
        [0.0, 0.0, 0.0, -0.5],
    ]


def test_diagonalize_symmetric_2x2():
    matrix = [
        [2.0, 1.0],
        [1.0, 2.0],
    ]
    eigvals, eigvecs = diagonalize_symmetric(matrix)
    assert eigvals == pytest.approx([1.0, 3.0], rel=1e-9, abs=1e-9)

    transformed = transform_to_eigenbasis(matrix, eigvecs)
    assert transformed[0][1] == pytest.approx(0.0, abs=1e-9)
    assert transformed[1][0] == pytest.approx(0.0, abs=1e-9)
    assert transformed[0][0] == pytest.approx(1.0, rel=1e-9, abs=1e-9)
    assert transformed[1][1] == pytest.approx(3.0, rel=1e-9, abs=1e-9)


def test_diagonalize_symmetric_anderson_hamiltonian():
    ham = anderson_impurity_hamiltonian(U=4.0, mu=1.5)
    eigvals, eigvecs = diagonalize_symmetric(ham)
    assert eigvals == pytest.approx([-1.5, -1.5, 0.0, 1.0], rel=1e-9, abs=1e-9)

    transformed = transform_to_eigenbasis(ham, eigvecs)
    for i in range(len(transformed)):
        for j in range(len(transformed)):
            if i == j:
                assert transformed[i][j] == pytest.approx(eigvals[i], abs=1e-8)
            else:
                assert transformed[i][j] == pytest.approx(0.0, abs=1e-8)


def test_diagonalize_requires_square_and_symmetric():
    with pytest.raises(ValueError):
        diagonalize_symmetric([[1.0, 2.0]])
    with pytest.raises(ValueError):
        diagonalize_symmetric([[1.0, 2.0], [3.0, 4.0]])


def test_transform_to_eigenbasis_dimension_checks():
    with pytest.raises(ValueError):
        transform_to_eigenbasis([[1.0, 0.0], [0.0, 1.0]], [[1.0]])


def test_creation_annihilation_operator_matrices_one_site():
    basis = [0b00, 0b01, 0b10, 0b11]
    c_up = annihilation_operator_matrix(basis, orb=0)
    cdg_up = creation_operator_matrix(basis, orb=0)

    assert c_up == [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    for i in range(4):
        for j in range(4):
            assert cdg_up[i][j] == c_up[j][i]


def test_lehmann_green_iwn_matches_anderson_atom_formula():
    import math

    U = 3.0
    mu = 1.1
    beta = 5.0
    basis = [0b00, 0b01, 0b10, 0b11]

    ham = anderson_impurity_hamiltonian(U=U, mu=mu, basis=basis)
    energies, eigvecs = diagonalize_symmetric(ham)
    c_up_occ = annihilation_operator_matrix(basis, orb=0)
    c_up_eig = transform_to_eigenbasis(c_up_occ, eigvecs)

    z = 1.0 + 2.0 * math.exp(beta * mu) + math.exp(beta * (2.0 * mu - U))
    n_dn = (math.exp(beta * mu) + math.exp(beta * (2.0 * mu - U))) / z

    for n in (-2, -1, 0, 1, 2):
        giwn = lehmann_green_iwn(energies, c_up_eig, beta=beta, matsubara_index=n)
        omega_n = (2 * n + 1) * math.pi / beta
        expected = (1.0 - n_dn) / complex(mu, omega_n) + n_dn / complex(mu - U, omega_n)
        assert giwn == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_lehmann_green_tau_matches_anderson_atom_formula():
    import math

    U = 4.0
    mu = 1.7
    beta = 6.0
    tau = 1.25
    basis = [0b00, 0b01, 0b10, 0b11]

    ham = anderson_impurity_hamiltonian(U=U, mu=mu, basis=basis)
    energies, eigvecs = diagonalize_symmetric(ham)
    c_up_occ = annihilation_operator_matrix(basis, orb=0)
    c_up_eig = transform_to_eigenbasis(c_up_occ, eigvecs)

    g_tau = lehmann_green_tau(energies, c_up_eig, beta=beta, tau=tau)

    z = 1.0 + 2.0 * math.exp(beta * mu) + math.exp(beta * (2.0 * mu - U))
    expected = -(math.exp(tau * mu) + math.exp(beta * mu - tau * (U - mu))) / z
    assert g_tau == pytest.approx(expected, rel=1e-11, abs=1e-11)


def test_two_particle_tau_contributions_select_single_time_sector():
    energies = [0.0]
    op = [[1.0]]
    contrib = two_particle_green_tau_contributions(
        energies, op, op, op, op, beta=3.0, tau1=2.0, tau2=1.0, tau3=0.5
    )
    assert contrib["tau1>tau2>tau3"] == pytest.approx(1.0)
    assert sum(abs(v) for k, v in contrib.items() if k != "tau1>tau2>tau3") == pytest.approx(0.0)


def test_two_particle_frequency_matches_numerical_tau_integral_with_regularization():
    import cmath
    import math

    energies = [0.0]
    op = [[1.0]]
    beta = 2.0
    n = 0
    m = 0
    o = 0

    g_iw = two_particle_green_iwn_inu_inup(energies, op, op, op, op, beta, n, m, o)

    omega = (2 * n + 1) * math.pi / beta
    nu = 2 * m * math.pi / beta
    nu_p = 2 * o * math.pi / beta

    grid = 32
    dt = beta / grid
    numeric = 0.0 + 0.0j
    for i in range(grid):
        tau1 = (i + 0.5) * dt
        for j in range(grid):
            tau2 = (j + 0.5) * dt
            for k in range(grid):
                tau3 = (k + 0.5) * dt
                g_tau = two_particle_green_tau(energies, op, op, op, op, beta, tau1, tau2, tau3)
                kernel = cmath.exp(1j * omega * tau1 + 1j * nu * tau2 - 1j * nu_p * tau3)
                numeric += kernel * g_tau

    numeric *= (dt**3) / (beta * beta)
    assert g_iw == pytest.approx(numeric, rel=2e-2, abs=2e-2)


def _one_particle_time_ordered_green(
    energies: list[float],
    c_eig: list[list[float]],
    cd_eig: list[list[float]],
    beta: float,
    tau_a: float,
    tau_b: float,
) -> float:
    import math

    z = sum(math.exp(-beta * e) for e in energies)
    dt = tau_a - tau_b

    def spectral_sum(delta_tau: float) -> float:
        total = 0.0
        for l in range(len(energies)):
            w_l = math.exp(-beta * energies[l])
            for m in range(len(energies)):
                total += (
                    w_l
                    * math.exp(delta_tau * (energies[l] - energies[m]))
                    * c_eig[l][m]
                    * cd_eig[m][l]
                )
        return total / z

    if dt >= 0.0:
        return -spectral_sum(dt)
    return spectral_sum(beta + dt)


def test_two_particle_green_reduces_to_wick_in_non_interacting_limit():
    U = 0.0
    mu = 0.37
    beta = 5.0
    tau1, tau2, tau3 = 3.4, 1.1, 2.2
    basis = [0b00, 0b01, 0b10, 0b11]

    ham = anderson_impurity_hamiltonian(U=U, mu=mu, basis=basis)
    energies, eigvecs = diagonalize_symmetric(ham)

    c_up_occ = annihilation_operator_matrix(basis, orb=0)
    cd_up_occ = creation_operator_matrix(basis, orb=0)
    c_up_eig = transform_to_eigenbasis(c_up_occ, eigvecs)
    cd_up_eig = transform_to_eigenbasis(cd_up_occ, eigvecs)

    g2 = two_particle_green_tau(
        energies, c_up_eig, cd_up_eig, c_up_eig, cd_up_eig, beta, tau1, tau2, tau3
    )

    g12 = _one_particle_time_ordered_green(energies, c_up_eig, cd_up_eig, beta, tau1, tau2)
    g30 = _one_particle_time_ordered_green(energies, c_up_eig, cd_up_eig, beta, tau3, 0.0)
    g10 = _one_particle_time_ordered_green(energies, c_up_eig, cd_up_eig, beta, tau1, 0.0)
    g32 = _one_particle_time_ordered_green(energies, c_up_eig, cd_up_eig, beta, tau3, tau2)

    wick = g12 * g30 - g10 * g32
    assert g2 == pytest.approx(wick, rel=1e-11, abs=1e-11)


def test_two_particle_static_sum_rule_equal_time_anticommutator():
    U = 2.3
    mu = 0.9
    beta = 4.0
    t = 1.7
    eps = 1e-6
    basis = [0b00, 0b01, 0b10, 0b11]

    ham = anderson_impurity_hamiltonian(U=U, mu=mu, basis=basis)
    energies, eigvecs = diagonalize_symmetric(ham)
    c_up = transform_to_eigenbasis(annihilation_operator_matrix(basis, orb=0), eigvecs)
    cd_up = transform_to_eigenbasis(creation_operator_matrix(basis, orb=0), eigvecs)
    identity = [[1.0 if i == j else 0.0 for j in range(len(energies))] for i in range(len(energies))]

    term_1 = two_particle_green_tau(energies, c_up, cd_up, identity, identity, beta, t + eps, t, 0.0)
    term_2 = two_particle_green_tau(energies, cd_up, c_up, identity, identity, beta, t + eps, t, 0.0)

    assert (term_1 + term_2) == pytest.approx(1.0, rel=2e-6, abs=2e-6)
