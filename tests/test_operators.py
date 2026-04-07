import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from operators import (
    annihilate,
    anderson_impurity_hamiltonian,
    create,
    diagonalize_symmetric,
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
