import re
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
filename = "app.log"  # your log file path
threshold = 0.6                # consensus threshold
cmap = "viridis"               # color map for correctness

# === Step 1: Parse the log file ===
pattern = re.compile(
    r"correctness: ([0-9.]+), reputation: ([0-9.]+), dof_norm: ([0-9.]+),\s*consensus score: ([0-9.]+)"
)

data = []
with open(filename, "r") as f:
    content = f.read()
    for match in pattern.finditer(content):
        data.append(tuple(map(float, match.groups())))

# Convert to DataFrame
df = pd.DataFrame(data, columns=["correctness", "reputation", "dof_norm", "consensus_score"])

# === Step 2: Plot all data in one chart ===
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    df["reputation"],
    df["consensus_score"],
    c=df["correctness"],
    cmap=cmap,
    s=80,
    alpha=0.8,
)

# Add threshold line
plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")

# Add colorbar and labels
plt.colorbar(scatter, label="Correctness")
plt.title("Consensus vs Reputation (all dof_norm values combined)")
plt.xlabel("Reputation")
plt.ylabel("Consensus Score")
plt.legend()
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.show()