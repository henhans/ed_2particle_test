"""Plot χ_sp(iω) from a double Matsubara sum and from a Lehmann single-frequency formula."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from operators import (
    anderson_impurity_hamiltonian,
    annihilation_operator_matrix,
    creation_operator_matrix,
    diagonalize_symmetric,
    single_frequency_bosonic_susceptibility,
    three_frequency_susceptibilities,
    transform_to_eigenbasis,
)


def _number_operator(basis: list[int], orbital: int) -> list[list[float]]:
    size = len(basis)
    op = [[0.0 for _ in range(size)] for _ in range(size)]
    for i, state in enumerate(basis):
        op[i][i] = float((state >> orbital) & 1)
    return op


def _double_sum_spin_susceptibility(
    energies: list[float],
    c_up: list[list[float]],
    cd_up: list[list[float]],
    c_dn: list[list[float]],
    cd_dn: list[list[float]],
    beta: float,
    bosonic_index_m: int,
    fermionic_cutoff: int,
    transfer_cutoff: int,
    tau_grid: int,
) -> complex:
    total = 0.0 + 0.0j
    for n in range(-fermionic_cutoff, fermionic_cutoff + 1):
        for o in range(-transfer_cutoff, transfer_cutoff + 1):
            chi = three_frequency_susceptibilities(
                energies,
                c_up,
                cd_up,
                c_dn,
                cd_dn,
                beta,
                fermionic_index_n=n,
                bosonic_index_m=bosonic_index_m,
                bosonic_index_o=o,
                tau_grid=tau_grid,
            )["chi_spin"]
            total += chi
    return (1.0 / (beta * beta)) * total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare χ_sp(iω) from a finite double Matsubara sum over χ_sp(iω,·,·) "
            "with the direct single-frequency Lehmann result."
        )
    )
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="On-site repulsion")
    parser.add_argument("--mu", type=float, default=2.0, help="Chemical potential")
    parser.add_argument("--omega-max", type=int, default=4, help="Max |m| for bosonic iω_m")
    parser.add_argument("--nu-cutoff", type=int, default=4, help="Max |n| in first Matsubara sum")
    parser.add_argument("--nup-cutoff", type=int, default=4, help="Max |o| in second Matsubara sum")
    parser.add_argument(
        "--tau-grid",
        type=int,
        default=48,
        help="Imaginary-time grid used in disconnected three-frequency contribution",
    )
    parser.add_argument(
        "--output",
        default="chi_spin_single_frequency.png",
        help="Output image path",
    )
    args = parser.parse_args()

    basis = [0b00, 0b01, 0b10, 0b11]
    ham = anderson_impurity_hamiltonian(U=args.U, mu=args.mu, basis=basis)
    energies, eigvecs = diagonalize_symmetric(ham)

    c_up = transform_to_eigenbasis(annihilation_operator_matrix(basis, orb=0), eigvecs)
    cd_up = transform_to_eigenbasis(creation_operator_matrix(basis, orb=0), eigvecs)
    c_dn = transform_to_eigenbasis(annihilation_operator_matrix(basis, orb=1), eigvecs)
    cd_dn = transform_to_eigenbasis(creation_operator_matrix(basis, orb=1), eigvecs)

    n_up = transform_to_eigenbasis(_number_operator(basis, orbital=0), eigvecs)
    n_dn = transform_to_eigenbasis(_number_operator(basis, orbital=1), eigvecs)
    s_z = [[0.5 * (n_up[i][j] - n_dn[i][j]) for j in range(len(basis))] for i in range(len(basis))]

    bosonic_indices = list(range(-args.omega_max, args.omega_max + 1))
    omega_values = [2.0 * m * 3.141592653589793 / args.beta for m in bosonic_indices]
    from_double_sum = [
        _double_sum_spin_susceptibility(
            energies,
            c_up,
            cd_up,
            c_dn,
            cd_dn,
            args.beta,
            bosonic_index_m=m,
            fermionic_cutoff=args.nu_cutoff,
            transfer_cutoff=args.nup_cutoff,
            tau_grid=args.tau_grid,
        )
        for m in bosonic_indices
    ]
    from_lehmann = [
        single_frequency_bosonic_susceptibility(energies, s_z, s_z, args.beta, bosonic_index_m=m)
        for m in bosonic_indices
    ]

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(
        omega_values,
        [val.real for val in from_double_sum],
        "o-",
        label=r"Re $T^2\sum_{\nu,\nu'}\chi_{sp}(i\omega,i\nu,i\nu')$",
    )
    plt.plot(
        omega_values,
        [val.real for val in from_lehmann],
        "s--",
        label=r"Re Lehmann $\chi_{sp}(i\omega)$",
    )
    plt.xlabel(r"$\omega_m$")
    plt.ylabel(r"$\chi_{sp}(i\omega_m)$")
    plt.title(
        "Single-frequency spin susceptibility: "
        f"beta={args.beta}, U={args.U}, mu={args.mu}, "
        f"cutoffs=({args.nu_cutoff}, {args.nup_cutoff})"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
