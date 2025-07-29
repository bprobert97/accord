# Introduction to ACCORD

This research introduces the Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework, a decentralised approach to satellite operations in Low Earth Orbit (LEO). To overcome the limitations of centralised coordination, including vulnerability to single points of failure and trust barriers, ACCORD employs a Directed Acyclic Graph (DAG)-based Distributed Ledger Technology (DLT). Satellites collaboratively determine each other’s orbits through mutual observation using on-board sensors, and submit this data to the ledger. 

Proof of Inter Satellite Evaluation (PoISE) is a reputation-based, Byzantine Fault Tolerant consensus mechanism, that validates these observations without relying on financial incentives or intermediaries. It assesses witnessed orbit determination data submitted by satellites in a network, adding valid data to the DAG and reaching consensus on which satellites and which observations can be trusted.

By incorporating Smart Contracts, the system enables autonomous decision-making, allowing satellites to self-govern, coordinate tasks, and reroute services in real time. This decentralised model not only enhances data integrity and trust but also improves scalability and performance as more satellites join the network. ACCORD thus fosters multi-system collaboration and resilience across heterogeneous constellations, making secure, autonomous satellite operations practical in zero-trust environments.

This code is licensed under a GNU General Public License v3.0.

# Current Status

The project is currently at TRL 0. The consensus mechanism is in the early phases of development by Beth Probert, a PhD student at the University of Strathclyde's Applied Space Technology Laboratory.

# Repository Layout

<pre>
📁 accord/
│
├── 📁 .github/workflows/              # GitHub Workflow files
│   └── main.yml                       # CI configuration for github: Pylint and demo notebook execution
│
├── 📁 design/                      # Design documents
│   └── consensus_design.ipynb     # Initial consensys mechanism design
│   └── dlt_design_mpl.ipynb       # Initial DAG design using matplotlib
│   └── dlt_design_plotly.ipynb    # Initial DAG design using plotly
│
├── 📁 images/                         # Image assets
│   └── consensus_flowchart.png        # Flowchart of consensus mechanism
│
├── 📁 references/                    # References
│   └── references.ipynb              # List of project references
│
├── 📁 src/                        # Main source code
│   └── __init__.py                # Empty file, for module creation
│   └── consensus_mech.py          # Code for the PoISE consensus mechanism
│   └── dag.py                     # Code for the Directed Acyclic Graph ledger structure
│   └── satellite_node.py          # Code representing a satellite in the network
│   └── transaction.py             # Code representing a transaction submitted by a satellite
│   └── utils.py                   # Utility functions and global variables
│
├── .gitignore               # Files/folders to ignore in Git
├── LICENSE.MD               # License file
├── README.md                # Project overview
├── od_data.json            # Example orbit determination data for use in consensus
├── requirements.txt        # List of python package dependencies
└── accord_demo.ipynb       # Jupyter notebook demonstration of ACCORD
</pre>

# Installation

Make sure Python 3.13 is installed on your system before proceeding with the installation.
Follow these steps to set up the project in a Python virtual environment:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/bprobert97/accord.git
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   On Windows:

   ```bash
   venv\Scripts\activate
   ```
   On macOS/Linux:

   ```bash
   source venv/bin/activate
   ```
4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
