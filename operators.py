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
