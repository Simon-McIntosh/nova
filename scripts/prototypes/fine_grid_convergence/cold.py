"""Run the certificate solver with geometric edge-passage strength."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from scripts.prototypes.fine_grid_convergence.path_weight import install


def main():
    from scripts.prototypes.fast_map_production.solver import main as solve

    install()
    solve()


if __name__ == "__main__":
    main()
