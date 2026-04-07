import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from operators import annihilate, create


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
