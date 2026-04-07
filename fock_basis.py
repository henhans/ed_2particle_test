"""Utilities for generating Fock bases.

A basis state is represented as a tuple of occupation numbers (0 or 1)
for each orbital in a spinless fermionic system.
"""

from __future__ import annotations

from itertools import combinations, product


OccupationState = tuple[int, ...]


def generate_fock_basis(num_orbitals: int, num_particles: int | None = None) -> list[OccupationState]:
    """Generate a Fock basis for ``num_orbitals`` spinless orbitals.

    Args:
        num_orbitals: Number of available orbitals.
        num_particles: Optional fixed particle-number sector. If ``None``,
            all sectors are returned.

    Returns:
        A list of occupation-number tuples.

    Raises:
        ValueError: If inputs are invalid.
    """

    if num_orbitals < 0:
        raise ValueError("num_orbitals must be non-negative")

    if num_particles is None:
        return [tuple(state) for state in product((0, 1), repeat=num_orbitals)]

    if num_particles < 0:
        raise ValueError("num_particles must be non-negative")
    if num_particles > num_orbitals:
        raise ValueError("num_particles cannot exceed num_orbitals")

    basis: list[OccupationState] = []
    for occupied in combinations(range(num_orbitals), num_particles):
        state = [0] * num_orbitals
        for idx in occupied:
            state[idx] = 1
        basis.append(tuple(state))

    return basis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Fock basis states.")
    parser.add_argument("num_orbitals", type=int, help="Number of orbitals")
    parser.add_argument(
        "-p",
        "--num-particles",
        type=int,
        default=None,
        help="Fix the particle number (optional)",
    )
    args = parser.parse_args()

    for state in generate_fock_basis(args.num_orbitals, args.num_particles):
        print(state)
