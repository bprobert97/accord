# Contributing to ACCORD

We welcome contributions to the ACCORD framework. Whether you are optimising the PoISE consensus logic, adding new orbital perturbation models, or improving the visualisations, your input is appreciated.

## Getting Started
1. Fork the repository and create your feature branch from `main`.
2. Ensure you have the required dependencies installed via `requirements.txt` or `requirements_windows.txt`.
3. Review the `FilterConfig` structure in `src/filter.py` before running local tests.

## Code Style & Best Practices
* **Performance:** We process data for large constellations. Prefer vectorised NumPy and Pandas array operations over standard Python loops.
* **Configuration:** Avoid hardcoding physical constraints (like ISL distances or sensor noise) deep in the modules. Always expose these via `FilterConfig`.

## Pull Request Process
1. Ensure any new consensus validations are fully covered by pytest.
2. If you alter the simulation output data structures, update the corresponding plotting functions in `src/plotting.py`.
3. Submit the PR and request a review from the maintainers.
