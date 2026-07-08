# Contributing to ACCORD

Thank you for your interest in contributing to the Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework!

This project aims for rigorous open-source software standards, specifically aligning with the Journal of Open Source Software (JOSS) review criteria. To maintain this standard, all contributions must pass comprehensive automated testing, static type checking, and documentation reviews.

## 1. Development Workflow

External contributors do not have direct push access to this repository. All development must be done via a **Fork and Pull Request** model.

1. **Fork the Repository:** Click the "Fork" button at the top right of the repository page.
2. **Clone your Fork:** git clone https://github.com/YOUR_USERNAME/accord.git
3. **Create a Feature Branch:** git checkout -b feature/your-feature-name (or bugfix/issue-description).
4. **Commit your Changes:** Write clear, concise commit messages.
5. **Push to your Fork:** git push origin feature/your-feature-name
6. **Submit a Pull Request (PR):** Open a PR against the `main` branch of the original ACCORD repository. Describe your changes in detail and link to any relevant open issues.

*Note on Git LFS:* The Monte Carlo simulations generate large `.npz` data files. You must either use Git LFS to track these files or explicitly add `*.npz` to your `.gitignore` to prevent bloating the repository history.

## 2. Coding Standards & Tools

All code must be formatted, typed, and spell-checked before a PR can be merged. We use several tools to enforce this. You should run these locally before submitting a PR.

* **Static Type Checking:** We use `mypy` to enforce static typing across the Python codebase. All new functions and methods must include appropriate type hints.
  `mypy .`

* **Linting & Formatting:** We use `pylint` to ensure clean code.
  `pylint src/ tests/ accord_demo.py mc_demo.py`

* **Spell Checking:** We use `codespell` to catch typos in code, comments, and documentation.
  `codespell`

## 3. Testing

We rely on `pytest` for unit testing to ensure the stability of the orbital filters, DAG structure, and consensus mechanisms.

* **Running Tests:** Before opening a PR, ensure all tests pass:
  pytest tests/

* **Adding Tests:** If you are adding a new feature, you must include corresponding unit tests in the `tests/` directory. Submissions without appropriate test coverage will not be merged.

## 4. Repository Layout

For context when navigating the codebase, here is the structure of the ACCORD repository:

<pre>
📁 accord/
│
├── 📁 .github/workflows/       # GitHub Workflow files
│   └── main.yml             # CI configuration for github: Pylint, Mypy and Pytest
│
├── 📁 design/                  # Design documents, Jupyter notebooks and PlantUML diagrams
│
├── 📁 images/                  # Image assets
│
├── 📁 src/                    # Main source code
│   └── 📁 filters/            # Implementation files for different Kalman Filters
│       └── ekf.py             # Implementation of an Extended Kalman Filter
│       └── filter_interface.py # Reusable code for implementing different Kalman Filters and interfacing with PoISE
│       └── ukf.py             # Implementation of an Unscented Kalman Filter
│   └── 📁 legacy/             # Legacy code maintained for easy plot generation
│   └── __init__.py            # Empty file, for module creation
│   └── consensus_mech.py      # Code for the PoISE consensus mechanism
│   └── dag.py                 # Code for the Directed Acyclic Graph ledger structure
│   └── logger.py              # Code for the app logger
│   └── mc_comparison.py       # Code for generating comparison plots for different Monte Carlo data sets
│   └── mc_plotting.py         # Code for plotting aggregated Monte Carlo simulation results
|   └── plotting.py            # Code for plotting individual simulation results
│   └── reputation.py          # Code for the satellite reputation manager
│   └── satellite_node.py      # Code representing a satellite in the network
│   └── simulation.py          # Helper functions for generating and converting satellite orbital elements
│   └── transaction.py         # Code representing a transaction submitted by a satellite
│   └── visualise_orbits.py    # Code for generating a gif of truth orbits
│
├── 📁 tests/                  # Unit tests, written with pytest
|
├── .codespellrc             # Codespell configuration file
├── .coveragerc              # Pytest coverage configuration file
├── .gitignore               # Files/folders to ignore in Git
├── .pylintrc                # Pylint configuration file
├── accord_demo.py           # Demonstration of ACCORD
├── changelog.md             # Release change log
├── CONTRIBUTING.md          # Guidance on contributing to the project
├── LICENSE.MD               # License file
├── mc_demo.py               # Monte Carlo Simulation of ACCORD
├── mypy.ini                 # Mypy configuration
├── README.md                # Project overview
├── requirements.txt         # List of python package dependencies for Linux and CI
└── requirements_windows.txt   # List of python package dependencies for Windows
</pre>
