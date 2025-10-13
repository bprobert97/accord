import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def plot_chi_squared(xmax: float = 10.0) -> None:
    """
    Plot chi-squared probability density function 1, 2 and 3 DOF.
    These are the possible DOF from satellite sensor measurements.

    Args:
    - xmax (float): Maximum x value to plot.

    Returns:
    None. Plots a graph using matplotlib.
    """
    x = np.linspace(0, xmax, 500)
    pdf1 = chi2.pdf(x, df=1)
    pdf2 = chi2.pdf(x, df=2)
    pdf3 = chi2.pdf(x, df=3)

    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf1, label=f"χ²(1 DOF)", linewidth=2)
    plt.plot(x, pdf2, label=f"χ²(2 DOF)", linewidth=2)
    plt.plot(x, pdf3, label=f"χ²(3 DOF)", linewidth=2)
    plt.title(f"Chi-squared Distribution")
    plt.xlabel("NIS value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_chi_squared_cdf(xmax: float = 10.0) -> None:
    """
    Plot chi-squared cumulative distribution function 1, 2 and 3 DOF.
    These are the possible DOF from satellite sensor measurements.

    Args:
    - xmax (float): Maximum x value to plot.

    Returns:
    None. Plots a graph using matplotlib.
    """
    x = np.linspace(0, xmax, 500)
    cdf1 = chi2.cdf(x, df=1)
    cdf2 = chi2.cdf(x, df=2)
    cdf3 = chi2.cdf(x, df=3)

    plt.figure(figsize=(8, 5))
    plt.plot(x, cdf1, label=f"χ²(1 DOF)", linewidth=2)
    plt.plot(x, cdf2, label=f"χ²(2 DOF)", linewidth=2)
    plt.plot(x, cdf3, label=f"χ²(3 DOF)", linewidth=2)
    plt.title(f"Chi-squared CDF")
    plt.xlabel("NIS value")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)
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
    x = np.linspace(0, xmax, 500)
    pdf1 = chi2.pdf(x, df=1)
    pdf2 = chi2.pdf(x, df=2)
    pdf3 = chi2.pdf(x, df=3)

    plt.figure(figsize=(8, 5))
    plt.plot(x, pdf1, label=f"χ²(1 DOF)", linewidth=2)
    plt.plot(x, pdf2, label=f"χ²(2 DOF)", linewidth=2)
    plt.plot(x, pdf3, label=f"χ²(3 DOF)", linewidth=2)
    plt.title(f"Chi-squared PDF")
    plt.xlabel("NIS value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_chi_squared()
plot_chi_squared_cdf()
plot_chi_squared_pdf()