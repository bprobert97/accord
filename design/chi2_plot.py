import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from curved_text import curved_text

def plot_chi_squared_cdf(xmax: float = 10.0) -> None:
    """
    Plot chi-squared cumulative distribution function 1, 2 and 3 DOF.
    These are the possible DOF from satellite sensor measurements.

    Args:
    - xmax (float): Maximum x value to plot.

    Returns:
    None. Plots a graph using matplotlib.
    """
    # Fixed: Reduced font size from 20 to 12 for standard sizing
    plt.rcParams.update({"font.size": 12})

    x = np.linspace(0, xmax, 500)
    cdf1 = chi2.cdf(x, df=1)
    cdf2 = chi2.cdf(x, df=2)
    cdf3 = chi2.cdf(x, df=3)

    _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, cdf1, label="χ²(1 DOF)", linewidth=2)
    curved_text(ax, x, cdf1, "χ²(1 DOF)",
                pos=0.5, anchor="center", offset=10.0, color="C0")

    ax.plot(x, cdf2, label="χ²(2 DOF)", linewidth=2)
    curved_text(ax, x, cdf2, "χ²(2 DOF)",
                pos=0.5, anchor="center", offset=10.0, color="C1")

    ax.plot(x, cdf3, label="χ²(3 DOF)", linewidth=2)
    curved_text(ax, x, cdf3, "χ²(3 DOF)",
                pos=0.5, anchor="center", offset=10.0, color="C2")

    ax.set_xlabel("Normalised Innovation Squared value")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True)
    plt.show()

def plot_chi_squared_pdf(xmax: float = 10.0) -> None:
    """
    Plot chi-squared probability density function 1, 2 and 3 DOF.
    These are the possible DOF from satellite sensor measurements.

    Args:
    - xmax (float): Maximum x value to plot.

    Returns:
    None. Plots a graph using matplotlib.
    """
    # Fixed: Start x at 0.01 instead of 0 to avoid Infinity at df=1
    x = np.linspace(0.01, xmax, 500)
    pdf1 = chi2.pdf(x, df=1)
    pdf2 = chi2.pdf(x, df=2)
    pdf3 = chi2.pdf(x, df=3)

    _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, pdf1, label="χ²(1 DOF)", linewidth=2)
    curved_text(ax, x, pdf1, "χ²(1 DOF)",
                pos=0.2, anchor="center", offset=10.0, color="C0")

    ax.plot(x, pdf2, label="χ²(2 DOF)", linewidth=2)
    curved_text(ax, x, pdf2, "χ²(2 DOF)",
                pos=0.15, anchor="center", offset=10.0, color="C1")

    ax.plot(x, pdf3, label="χ²(3 DOF)", linewidth=2)
    curved_text(ax, x, pdf3, "χ²(3 DOF)",
                pos=0.4, anchor="center", offset=10.0, color="C2")

    ax.set_xlabel("Normalised Innovation Squared value")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    plt.show()

plot_chi_squared_cdf()
plot_chi_squared_pdf()