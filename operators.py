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
import cmath
from itertools import permutations


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


def annihilation_operator_matrix(basis: list[int], orb: int) -> list[list[float]]:
    """Build the annihilation operator matrix for a chosen orbital.

    The returned matrix ``c`` is in the occupation basis provided by ``basis``
    with elements ``c[m][n] = <m|c_orb|n>``.
    """
    if orb < 0:
        raise ValueError("orb must be non-negative")

    size = len(basis)
    state_to_index = {state: i for i, state in enumerate(basis)}
    operator = [[0.0 for _ in range(size)] for _ in range(size)]

    for n, state in enumerate(basis):
        new_state, sign = annihilate(state, orb)
        if new_state is None:
            continue
        m = state_to_index.get(new_state)
        if m is None:
            raise ValueError("basis must be closed under annihilation on the selected orbital")
        operator[m][n] = float(sign)

    return operator


def creation_operator_matrix(basis: list[int], orb: int) -> list[list[float]]:
    """Build the creation operator matrix for a chosen orbital."""
    if orb < 0:
        raise ValueError("orb must be non-negative")

    size = len(basis)
    state_to_index = {state: i for i, state in enumerate(basis)}
    operator = [[0.0 for _ in range(size)] for _ in range(size)]

    for n, state in enumerate(basis):
        new_state, sign = create(state, orb)
        if new_state is None:
            continue
        m = state_to_index.get(new_state)
        if m is None:
            raise ValueError("basis must be closed under creation on the selected orbital")
        operator[m][n] = float(sign)

    return operator


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


def lehmann_green_tau(
    energies: list[float],
    c_eigenbasis: list[list[float]],
    beta: float,
    tau: float,
) -> float:
    """Evaluate ``G(tau)`` from a Lehmann representation.

    Uses the fermionic convention for ``0 <= tau <= beta``:

    ``G(tau) = -(1/Z) * sum_{m,n} exp(-beta E_m) exp(tau(E_m-E_n)) |<m|c|n>|^2``.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if tau < 0.0 or tau > beta:
        raise ValueError("tau must satisfy 0 <= tau <= beta")

    n = len(energies)
    if len(c_eigenbasis) != n or any(len(row) != n for row in c_eigenbasis):
        raise ValueError("c_eigenbasis must match energies dimensions")

    boltz = [math.exp(-beta * e) for e in energies]
    partition = sum(boltz)
    if partition == 0.0:
        raise ValueError("partition function underflowed to zero")

    total = 0.0
    for m in range(n):
        for nn in range(n):
            amp = c_eigenbasis[m][nn]
            total += boltz[m] * math.exp(tau * (energies[m] - energies[nn])) * (amp * amp)

    return -total / partition


def lehmann_green_iwn(
    energies: list[float],
    c_eigenbasis: list[list[float]],
    beta: float,
    matsubara_index: int,
) -> complex:
    """Evaluate ``G(iω_n)`` from the fermionic Lehmann representation."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    n = len(energies)
    if len(c_eigenbasis) != n or any(len(row) != n for row in c_eigenbasis):
        raise ValueError("c_eigenbasis must match energies dimensions")

    omega_n = (2 * matsubara_index + 1) * math.pi / beta
    iwn = complex(0.0, omega_n)

    boltz = [math.exp(-beta * e) for e in energies]
    partition = sum(boltz)
    if partition == 0.0:
        raise ValueError("partition function underflowed to zero")

    total = 0.0 + 0.0j
    for m in range(n):
        for nn in range(n):
            amp = c_eigenbasis[m][nn]
            weight = amp * amp
            total += ((boltz[m] + boltz[nn]) * weight) / (iwn + energies[m] - energies[nn])

    return total / partition


def _perm_sign(order: tuple[int, int, int]) -> int:
    inversions = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                inversions += 1
    return -1 if (inversions % 2) else 1


def _i1_exp(beta: float, a: complex) -> complex:
    if abs(a) < 1e-14:
        return complex(beta, 0.0)
    return cmath.exp(a * beta) / a - 1.0 / a


def _d_i1_exp(beta: float, a: complex) -> complex:
    if abs(a) < 1e-10:
        return complex(0.5 * beta * beta, 0.0)
    exp_term = cmath.exp(a * beta)
    return (beta * a * exp_term - exp_term + 1.0) / (a * a)


def _i2_ordered(beta: float, a: complex, b: complex) -> complex:
    if abs(b) < 1e-12:
        return _d_i1_exp(beta, a)
    return (_i1_exp(beta, a + b) - _i1_exp(beta, a)) / b


def _d_i2_db(beta: float, a: complex, b: complex) -> complex:
    if abs(b) < 1e-10:
        eps = 1e-7
        return (_d_i1_exp(beta, a + eps) - _d_i1_exp(beta, a - eps)) / (2.0 * eps)
    i1_ab = _i1_exp(beta, a + b)
    i1_a = _i1_exp(beta, a)
    return (b * _d_i1_exp(beta, a + b) - (i1_ab - i1_a)) / (b * b)


def _i3_ordered(beta: float, a: complex, b: complex, c: complex) -> complex:
    """Evaluate ∫_0^β dt1 e^{a t1} ∫_0^{t1} dt2 e^{b t2} ∫_0^{t2} dt3 e^{c t3}."""
    if abs(c) < 1e-12:
        return _d_i2_db(beta, a, b)
    return (_i2_ordered(beta, a, b + c) - _i2_ordered(beta, a, b)) / c


def _validate_square_matrix(matrix: list[list[float]], name: str, n: int) -> None:
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError(f"{name} must match energies dimensions")


def two_particle_green_tau_contributions(
    energies: list[float],
    op1: list[list[float]],
    op2: list[list[float]],
    op3: list[list[float]],
    op4: list[list[float]],
    beta: float,
    tau1: float,
    tau2: float,
    tau3: float,
) -> dict[str, float]:
    """Lehmann 2PGF ``G^(2)(τ1,τ2,τ3)=<T O1(τ1)O2(τ2)O3(τ3)O4(0)>`` split by time-order sectors."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    for tau in (tau1, tau2, tau3):
        if tau < 0.0 or tau > beta:
            raise ValueError("all taus must satisfy 0 <= tau <= beta")

    n = len(energies)
    _validate_square_matrix(op1, "op1", n)
    _validate_square_matrix(op2, "op2", n)
    _validate_square_matrix(op3, "op3", n)
    _validate_square_matrix(op4, "op4", n)

    boltz = [math.exp(-beta * e) for e in energies]
    partition = sum(boltz)
    if partition == 0.0:
        raise ValueError("partition function underflowed to zero")

    tau_values = [tau1, tau2, tau3]
    operators = [op1, op2, op3]
    contributions: dict[str, float] = {}

    for order in permutations((0, 1, 2)):
        if not (tau_values[order[0]] >= tau_values[order[1]] >= tau_values[order[2]]):
            contributions[f"tau{order[0] + 1}>tau{order[1] + 1}>tau{order[2] + 1}"] = 0.0
            continue

        sign = _perm_sign(order)
        op_a, op_b, op_c = (operators[idx] for idx in order)
        t_a, t_b, t_c = (tau_values[idx] for idx in order)
        total = 0.0
        for l in range(n):
            w_l = boltz[l]
            for m in range(n):
                phase_a = math.exp((energies[l] - energies[m]) * t_a)
                elm = op_a[l][m]
                if elm == 0.0:
                    continue
                for nn in range(n):
                    phase_b = math.exp((energies[m] - energies[nn]) * t_b)
                    emn = op_b[m][nn]
                    if emn == 0.0:
                        continue
                    for k in range(n):
                        enk = op_c[nn][k]
                        ekl = op4[k][l]
                        if enk == 0.0 or ekl == 0.0:
                            continue
                        phase_c = math.exp((energies[nn] - energies[k]) * t_c)
                        total += w_l * phase_a * phase_b * phase_c * elm * emn * enk * ekl
        contributions[f"tau{order[0] + 1}>tau{order[1] + 1}>tau{order[2] + 1}"] = sign * total / partition

    return contributions


def two_particle_green_tau(
    energies: list[float],
    op1: list[list[float]],
    op2: list[list[float]],
    op3: list[list[float]],
    op4: list[list[float]],
    beta: float,
    tau1: float,
    tau2: float,
    tau3: float,
) -> float:
    """Total ``G^(2)(τ1,τ2,τ3)`` obtained as the sum of time-order contributions."""
    return sum(
        two_particle_green_tau_contributions(energies, op1, op2, op3, op4, beta, tau1, tau2, tau3).values()
    )


def two_particle_green_iwn_inu_inup_contributions(
    energies: list[float],
    op1: list[list[float]],
    op2: list[list[float]],
    op3: list[list[float]],
    op4: list[list[float]],
    beta: float,
    fermionic_index_n: int,
    bosonic_index_m: int,
    bosonic_index_o: int,
) -> dict[str, complex]:
    """Lehmann 2PGF ``G^(2)(iω_n,iν_m,iν'_o)`` split by the six time-order sectors.

    Fourier convention:
    ``exp(+iω_n τ1) exp(+iν_m τ2) exp(-iν'_o τ3)``, with
    ``ω_n=(2n+1)π/β`` and ``ν_m=2mπ/β``.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    n_states = len(energies)
    _validate_square_matrix(op1, "op1", n_states)
    _validate_square_matrix(op2, "op2", n_states)
    _validate_square_matrix(op3, "op3", n_states)
    _validate_square_matrix(op4, "op4", n_states)

    omega_n = (2 * fermionic_index_n + 1) * math.pi / beta
    nu_m = 2 * bosonic_index_m * math.pi / beta
    nu_o = 2 * bosonic_index_o * math.pi / beta
    fourier_coeff = [complex(0.0, omega_n), complex(0.0, nu_m), complex(0.0, -nu_o)]

    boltz = [math.exp(-beta * e) for e in energies]
    partition = sum(boltz)
    if partition == 0.0:
        raise ValueError("partition function underflowed to zero")

    operators = [op1, op2, op3]
    contributions: dict[str, complex] = {}

    for order in permutations((0, 1, 2)):
        sign = _perm_sign(order)
        op_a, op_b, op_c = (operators[idx] for idx in order)
        fc_a, fc_b, fc_c = (fourier_coeff[idx] for idx in order)
        total = 0.0 + 0.0j
        for l in range(n_states):
            w_l = boltz[l]
            for m in range(n_states):
                elm = op_a[l][m]
                if elm == 0.0:
                    continue
                a = (energies[l] - energies[m]) + fc_a
                for nn in range(n_states):
                    emn = op_b[m][nn]
                    if emn == 0.0:
                        continue
                    b = (energies[m] - energies[nn]) + fc_b
                    for k in range(n_states):
                        enk = op_c[nn][k]
                        ekl = op4[k][l]
                        if enk == 0.0 or ekl == 0.0:
                            continue
                        c = (energies[nn] - energies[k]) + fc_c
                        total += w_l * elm * emn * enk * ekl * _i3_ordered(beta, a, b, c)

        key = f"tau{order[0] + 1}>tau{order[1] + 1}>tau{order[2] + 1}"
        contributions[key] = sign * total / (partition * beta * beta)

    return contributions


def two_particle_green_iwn_inu_inup(
    energies: list[float],
    op1: list[list[float]],
    op2: list[list[float]],
    op3: list[list[float]],
    op4: list[list[float]],
    beta: float,
    fermionic_index_n: int,
    bosonic_index_m: int,
    bosonic_index_o: int,
) -> complex:
    """Total ``G^(2)(iω_n,iν_m,iν'_o)`` as a sum over time-order sectors."""
    return sum(
        two_particle_green_iwn_inu_inup_contributions(
            energies,
            op1,
            op2,
            op3,
            op4,
            beta,
            fermionic_index_n,
            bosonic_index_m,
            bosonic_index_o,
        ).values()
    )


def one_particle_time_ordered_green_tau(
    energies: list[float],
    c_eigenbasis: list[list[float]],
    cd_eigenbasis: list[list[float]],
    beta: float,
    tau_a: float,
    tau_b: float,
) -> float:
    """Time-ordered Green's function ``G(τ_a,τ_b)=-<T c(τ_a)c†(τ_b)>``."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not (0.0 <= tau_a <= beta and 0.0 <= tau_b <= beta):
        raise ValueError("tau_a and tau_b must satisfy 0 <= tau <= beta")

    n = len(energies)
    _validate_square_matrix(c_eigenbasis, "c_eigenbasis", n)
    _validate_square_matrix(cd_eigenbasis, "cd_eigenbasis", n)

    boltz = [math.exp(-beta * e) for e in energies]
    partition = sum(boltz)
    if partition == 0.0:
        raise ValueError("partition function underflowed to zero")

    delta_tau = tau_a - tau_b

    def _spectral_sum(dt: float) -> float:
        total = 0.0
        for l in range(n):
            w_l = boltz[l]
            for m in range(n):
                total += (
                    w_l
                    * math.exp(dt * (energies[l] - energies[m]))
                    * c_eigenbasis[l][m]
                    * cd_eigenbasis[m][l]
                )
        return total / partition

    if delta_tau >= 0.0:
        return -_spectral_sum(delta_tau)
    return _spectral_sum(beta + delta_tau)


def disconnected_two_particle_green_iwn_inu_inup(
    energies: list[float],
    c1_eigenbasis: list[list[float]],
    cd2_eigenbasis: list[list[float]],
    c3_eigenbasis: list[list[float]],
    cd4_eigenbasis: list[list[float]],
    beta: float,
    fermionic_index_n: int,
    bosonic_index_m: int,
    bosonic_index_o: int,
    *,
    include_exchange: bool,
    tau_grid: int = 80,
) -> complex:
    """Disconnected contribution in the three-frequency convention used for ``G^(2)``.

    The disconnected term is computed from products of one-particle propagators:
    ``G12*G30`` and (optionally) ``-G10*G32`` for equal-spin exchange.
    """
    if tau_grid <= 1:
        raise ValueError("tau_grid must be > 1")

    omega_n = (2 * fermionic_index_n + 1) * math.pi / beta
    nu_m = 2 * bosonic_index_m * math.pi / beta
    nu_o = 2 * bosonic_index_o * math.pi / beta

    dt = beta / tau_grid
    total = 0.0 + 0.0j
    for i in range(tau_grid):
        tau1 = (i + 0.5) * dt
        for j in range(tau_grid):
            tau2 = (j + 0.5) * dt
            g12 = one_particle_time_ordered_green_tau(
                energies, c1_eigenbasis, cd2_eigenbasis, beta, tau1, tau2
            )
            for k in range(tau_grid):
                tau3 = (k + 0.5) * dt
                g30 = one_particle_time_ordered_green_tau(
                    energies, c3_eigenbasis, cd4_eigenbasis, beta, tau3, 0.0
                )
                disconnected = g12 * g30
                if include_exchange:
                    g10 = one_particle_time_ordered_green_tau(
                        energies, c1_eigenbasis, cd4_eigenbasis, beta, tau1, 0.0
                    )
                    g32 = one_particle_time_ordered_green_tau(
                        energies, c3_eigenbasis, cd2_eigenbasis, beta, tau3, tau2
                    )
                    disconnected -= g10 * g32

                kernel = cmath.exp(1j * omega_n * tau1 + 1j * nu_m * tau2 - 1j * nu_o * tau3)
                total += kernel * disconnected

    return total * (dt**3) / (beta * beta)


def three_frequency_susceptibilities(
    energies: list[float],
    c_up: list[list[float]],
    cd_up: list[list[float]],
    c_dn: list[list[float]],
    cd_dn: list[list[float]],
    beta: float,
    fermionic_index_n: int,
    bosonic_index_m: int,
    bosonic_index_o: int,
    *,
    tau_grid: int = 80,
) -> dict[str, complex]:
    """Compute ``χ↑↑`` and ``χ↑↓`` and spin/charge combinations.

    The connected susceptibilities are defined as
    ``χ = G^(2) - G^(2)_disc`` in the same three-frequency convention as
    :func:`two_particle_green_iwn_inu_inup`.
    """
    g2_upup = two_particle_green_iwn_inu_inup(
        energies, c_up, cd_up, c_up, cd_up, beta, fermionic_index_n, bosonic_index_m, bosonic_index_o
    )
    disc_upup = disconnected_two_particle_green_iwn_inu_inup(
        energies,
        c_up,
        cd_up,
        c_up,
        cd_up,
        beta,
        fermionic_index_n,
        bosonic_index_m,
        bosonic_index_o,
        include_exchange=True,
        tau_grid=tau_grid,
    )
    chi_upup = g2_upup - disc_upup

    g2_updn = two_particle_green_iwn_inu_inup(
        energies, c_up, cd_up, c_dn, cd_dn, beta, fermionic_index_n, bosonic_index_m, bosonic_index_o
    )
    disc_updn = disconnected_two_particle_green_iwn_inu_inup(
        energies,
        c_up,
        cd_up,
        c_dn,
        cd_dn,
        beta,
        fermionic_index_n,
        bosonic_index_m,
        bosonic_index_o,
        include_exchange=False,
        tau_grid=tau_grid,
    )
    chi_updn = g2_updn - disc_updn

    return {
        "chi_upup": chi_upup,
        "chi_updn": chi_updn,
        "chi_charge": chi_upup + chi_updn,
        "chi_spin": chi_upup - chi_updn,
    }


def single_frequency_bosonic_susceptibility(
    energies: list[float],
    op_a_eigenbasis: list[list[float]],
    op_b_eigenbasis: list[list[float]],
    beta: float,
    bosonic_index_m: int,
) -> complex:
    """Connected single-frequency Lehmann susceptibility ``χ_AB(iν_m)``."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    n = len(energies)
    _validate_square_matrix(op_a_eigenbasis, "op_a_eigenbasis", n)
    _validate_square_matrix(op_b_eigenbasis, "op_b_eigenbasis", n)

    nu_m = 2 * bosonic_index_m * math.pi / beta
    iv = complex(0.0, nu_m)
    boltz = [math.exp(-beta * e) for e in energies]
    partition = sum(boltz)
    if partition == 0.0:
        raise ValueError("partition function underflowed to zero")

    total = 0.0 + 0.0j
    for l in range(n):
        for m in range(n):
            num = boltz[l] - boltz[m]
            amp = op_a_eigenbasis[l][m] * op_b_eigenbasis[m][l]
            denom = iv + energies[l] - energies[m]
            if abs(denom) < 1e-14:
                continue
            total += num * amp / denom

    return total / partition
