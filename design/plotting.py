import re
import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
filename = "app.log"  # your log file path
threshold = 0.6                # consensus threshold
cmap = "viridis"               # color map for correctness

def plot_consensus_vs_reputation(df):
    """Plots consensus score vs reputation."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["reputation"],
        df["consensus_score"],
        c=df["correctness"],
        cmap=cmap,
        s=80,
        alpha=0.8,
    )
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")
    plt.colorbar(scatter, label="Correctness")
    plt.title("Consensus vs Reputation")
    plt.xlabel("Reputation")
    plt.ylabel("Consensus Score")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_nis_vs_correctness(df):
    """Plots NIS vs correctness."""
    plt.figure(figsize=(10, 7))
    plt.scatter(
        df["nis"],
        df["correctness"],
        cmap=cmap,
        s=80,
        alpha=0.8,
    )
    plt.title("NIS vs Correctness")
    plt.xlabel("NIS")
    plt.ylabel("Correctness")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_nis_vs_consensus(df):
    """Plots NIS vs consensus score."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df["nis"],
        df["consensus_score"],
        c=df["correctness"],
        cmap=cmap,
        s=80,
        alpha=0.8,
    )
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold}")
    plt.colorbar(scatter, label="Correctness")
    plt.title("NIS vs Consensus Score")
    plt.xlabel("NIS")
    plt.ylabel("Consensus Score")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def main():
    """Main function to parse log and generate plots."""
    # === Step 1: Parse the log file ===
    pattern = re.compile(
        r"NIS=([0-9.]+), DOF=([0-9]+), correctness=([0-9.]+), consensus_score=([0-9.]+),\s*reputation=([0-9.]+)"
    )

    data = []
    try:
        with open(filename, "r") as f:
            content = f.read()
            for match in pattern.finditer(content):
                data.append(tuple(map(float, match.groups())))
    except FileNotFoundError:
        print(f"Error: Log file not found at '{filename}'. Make sure the path is correct.")
        return

    if not data:
        print("No data found in log file matching the pattern.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["nis", "dof", "correctness", "consensus_score", "reputation"])

    # === Step 2: Generate plots ===
    plot_consensus_vs_reputation(df)
    plot_nis_vs_correctness(df)
    plot_nis_vs_consensus(df)

if __name__ == "__main__":
    main()
