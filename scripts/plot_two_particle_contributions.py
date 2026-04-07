"""Visualize two-particle imaginary-time Green's-function time-order contributions."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from operators import (
    anderson_impurity_hamiltonian,
    annihilation_operator_matrix,
    creation_operator_matrix,
    diagonalize_symmetric,
    transform_to_eigenbasis,
    two_particle_green_tau_contributions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2PGF time-order sector contributions.")
    parser.add_argument("--beta", type=float, default=10.0, help="Inverse temperature")
    parser.add_argument("--U", type=float, default=4.0, help="On-site repulsion")
    parser.add_argument("--mu", type=float, default=2.0, help="Chemical potential")
    parser.add_argument("--tau1", type=float, default=8.0, help="Tau1 in [0, beta]")
    parser.add_argument("--tau2", type=float, default=5.0, help="Tau2 in [0, beta]")
    parser.add_argument("--tau3", type=float, default=2.0, help="Tau3 in [0, beta]")
    parser.add_argument("--output", default="g2_contributions.png", help="Output image path")
    args = parser.parse_args()

    basis = [0b00, 0b01, 0b10, 0b11]
    ham = anderson_impurity_hamiltonian(U=args.U, mu=args.mu, basis=basis)
    energies, eigvecs = diagonalize_symmetric(ham)

    c_up = annihilation_operator_matrix(basis, orb=0)
    c_up_dag = creation_operator_matrix(basis, orb=0)

    c_up_eig = transform_to_eigenbasis(c_up, eigvecs)
    c_up_dag_eig = transform_to_eigenbasis(c_up_dag, eigvecs)

    contributions = two_particle_green_tau_contributions(
        energies,
        op1=c_up_dag_eig,
        op2=c_up_eig,
        op3=c_up_dag_eig,
        op4=c_up_eig,
        beta=args.beta,
        tau1=args.tau1,
        tau2=args.tau2,
        tau3=args.tau3,
    )

    labels = list(contributions.keys())
    values = [contributions[label] for label in labels]

    plt.figure(figsize=(8, 4))
    colors = ["#4C72B0" if value >= 0.0 else "#DD8452" for value in values]
    plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Contribution")
    plt.title(
        f"2PGF sector contributions at (tau1, tau2, tau3)=({args.tau1}, {args.tau2}, {args.tau3})"
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
