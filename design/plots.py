import re
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
filename = "app.log"  # path to your file
threshold = 0.6                # consensus threshold line
cmap = "viridis"               # color map for correctness

# === Step 1: Parse the file ===
pattern = re.compile(
    r"correctness: ([0-9.]+), reputation: ([0-9.]+), dof_norm: ([0-9.]+), consensus score: ([0-9.]+)"
)

data = []
with open(filename, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data.append(tuple(map(float, match.groups())))

# Convert to DataFrame
df = pd.DataFrame(data, columns=["correctness", "reputation", "dof_norm", "consensus_score"])

# === Step 2: Plot for each dof_norm ===
for dof in sorted(df["dof_norm"].unique()):
    subset = df[df["dof_norm"] == dof]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        subset["reputation"],
        subset["consensus_score"],
        c=subset["correctness"],
        cmap=cmap,
        s=80,
    )

    # Add threshold line and colorbar
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")
    plt.colorbar(scatter, label="Correctness")

    # Labels and title
    plt.title(f"Consensus vs Reputation (dof_norm = {dof})")
    plt.xlabel("Reputation")
    plt.ylabel("Consensus Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
