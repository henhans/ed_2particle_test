"""Plot the single-particle imaginary-time Green's function for the Anderson atom."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from operators import (
    anderson_impurity_hamiltonian,
    annihilation_operator_matrix,
    diagonalize_symmetric,
    lehmann_green_tau,
    transform_to_eigenbasis,
)


def build_g_tau(beta: float, u: float, mu: float, num_points: int) -> tuple[list[float], list[float]]:
    basis = [0b00, 0b01, 0b10, 0b11]
    ham = anderson_impurity_hamiltonian(U=u, mu=mu, basis=basis)
    energies, eigvecs = diagonalize_symmetric(ham)

    c_up = annihilation_operator_matrix(basis, orb=0)
    c_up_eig = transform_to_eigenbasis(c_up, eigvecs)

    taus = [beta * i / (num_points - 1) for i in range(num_points)]
    values = [lehmann_green_tau(energies, c_up_eig, beta=beta, tau=tau) for tau in taus]
    return taus, values


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot G(tau) for the atomic Anderson impurity model.")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="On-site repulsion")
    parser.add_argument("--mu", type=float, default=2.0, help="Chemical potential")
    parser.add_argument("--points", type=int, default=200, help="Number of tau grid points")
    parser.add_argument("--output", default="g_tau.png", help="Output image path")
    args = parser.parse_args()

    taus, values = build_g_tau(beta=args.beta, u=args.U, mu=args.mu, num_points=args.points)

    plt.figure(figsize=(6, 4))
    plt.plot(taus, values, linewidth=2)
    plt.xlabel(r"$\\tau$")
    plt.ylabel(r"$G(\\tau)$")
    plt.title(f"Atomic Anderson model: beta={args.beta}, U={args.U}, mu={args.mu}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
