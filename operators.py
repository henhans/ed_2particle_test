"""Fermionic creation and annihilation operators on a bit-encoded basis.

Each many-body basis state is represented as an integer where bit ``i`` stores the
occupation of orbital ``i``:

- bit ``i`` == 1: occupied
- bit ``i`` == 0: empty

The fermionic phase for acting on orbital ``orb`` is ``(-1)**N_left`` where
``N_left`` is the number of occupied orbitals with index smaller than ``orb``.
"""

from __future__ import annotations

import math


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


def diagonalize_symmetric(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Diagonalize a real symmetric matrix with a Jacobi rotation method.

    Args:
        matrix: Square, real symmetric matrix.

    Returns:
        ``(eigenvalues, eigenvectors)`` where eigenvectors are returned as columns
        of the nested list matrix.
    """
    n = len(matrix)
    if n == 0:
        return [], []
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")

    a = [[float(v) for v in row] for row in matrix]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > 1e-12:
                raise ValueError("matrix must be symmetric")

    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    def max_offdiag() -> tuple[float, int, int]:
        value = 0.0
        p = q = 0
        for i in range(n):
            for j in range(i + 1, n):
                candidate = abs(a[i][j])
                if candidate > value:
                    value = candidate
                    p, q = i, j
        return value, p, q

    max_sweeps = 50 * n * n
    for _ in range(max_sweeps):
        offdiag, p, q = max_offdiag()
        if offdiag < 1e-12:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        tau = (aqq - app) / (2.0 * apq)
        t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        a[p][p] = app - t * apq
        a[q][q] = aqq + t * apq
        a[p][q] = a[q][p] = 0.0

        for k in range(n):
            if k not in (p, q):
                akp = a[k][p]
                akq = a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq

        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    else:
        raise RuntimeError("Jacobi diagonalization did not converge")

    eigvals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigvals[i])
    eigvals_sorted = [eigvals[i] for i in order]
    eigvecs_sorted = [[v[row][i] for i in order] for row in range(n)]
    return eigvals_sorted, eigvecs_sorted


def transform_to_eigenbasis(matrix: list[list[float]], eigenvectors: list[list[float]]) -> list[list[float]]:
    """Transform an operator into the eigenbasis given by column eigenvectors.

    Computes ``V^T A V`` where ``A`` is ``matrix`` and ``V`` is ``eigenvectors``.
    """
    n = len(matrix)
    if n == 0:
        return []
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    if len(eigenvectors) != n or any(len(row) != n for row in eigenvectors):
        raise ValueError("eigenvectors must be a square matrix matching matrix size")

    tmp = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            tmp[i][j] = sum(matrix[i][k] * eigenvectors[k][j] for k in range(n))

    transformed = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            transformed[i][j] = sum(eigenvectors[k][i] * tmp[k][j] for k in range(n))

    return transformed
