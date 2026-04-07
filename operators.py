"""Fermionic creation and annihilation operators on a bit-encoded basis.

Each many-body basis state is represented as an integer where bit ``i`` stores the
occupation of orbital ``i``:

- bit ``i`` == 1: occupied
- bit ``i`` == 0: empty

The fermionic phase for acting on orbital ``orb`` is ``(-1)**N_left`` where
``N_left`` is the number of occupied orbitals with index smaller than ``orb``.
"""

from __future__ import annotations


def _parity_left(state: int, orb: int) -> int:
    """Return the fermionic sign from occupied orbitals left of ``orb``."""
    mask = (1 << orb) - 1
    n_left = (state & mask).bit_count()
    return -1 if (n_left % 2) else 1


def create(state: int, orb: int) -> tuple[int | None, int]:
    """Apply a fermionic creation operator at orbital ``orb``.

    Returns ``(new_state, sign)``. If ``orb`` is already occupied, the action is
    zero and ``(None, 0)`` is returned.
    """
    if orb < 0:
        raise ValueError("orb must be non-negative")

    if (state >> orb) & 1:
        return None, 0

    sign = _parity_left(state, orb)
    new_state = state | (1 << orb)
    return new_state, sign


def annihilate(state: int, orb: int) -> tuple[int | None, int]:
    """Apply a fermionic annihilation operator at orbital ``orb``.

    Returns ``(new_state, sign)``. If ``orb`` is empty, the action is zero and
    ``(None, 0)`` is returned.
    """
    if orb < 0:
        raise ValueError("orb must be non-negative")

    if not ((state >> orb) & 1):
        return None, 0

    sign = _parity_left(state, orb)
    new_state = state & ~(1 << orb)
    return new_state, sign


def anderson_impurity_hamiltonian(
    U: float,
    mu: float,
    basis: list[int] | None = None,
    up_orb: int = 0,
    dn_orb: int = 1,
) -> list[list[float]]:
    """Build the Anderson impurity Hamiltonian matrix.

    The model is

    ``H = U n_up n_dn - mu (n_up + n_dn)``.

    in a bit-encoded many-body basis. This Hamiltonian is diagonal in the
    occupation basis.

    Args:
        U: On-site interaction strength.
        mu: Chemical potential.
        basis: Ordered list of bit-encoded basis states. If ``None``, the
            canonical one-site basis ``[0b00, 0b01, 0b10, 0b11]`` is used.
        up_orb: Orbital index used for the spin-up occupation bit.
        dn_orb: Orbital index used for the spin-down occupation bit.

    Returns:
        Square Hamiltonian matrix as a nested list.
    """
    if basis is None:
        basis = [0b00, 0b01, 0b10, 0b11]

    size = len(basis)
    ham = [[0.0 for _ in range(size)] for _ in range(size)]

    for i, state in enumerate(basis):
        n_up = (state >> up_orb) & 1
        n_dn = (state >> dn_orb) & 1
        ham[i][i] = U * n_up * n_dn - mu * (n_up + n_dn)

    return ham
