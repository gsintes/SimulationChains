"""Analyse the smoculhowski equation."""

import numpy as np
import matplotlib.pyplot as plt

def concentration(time: float, n: int) -> float:
    """Calculate the concentration at time t."""
    return 4 * (time / (time + 2)) ** (n - 1) / (time + 2) ** 2

if __name__=="__main__":
    time = np.linspace(0, 2, 100)
    plt.figure()
    for n in range(1, 9):
        plt.plot(time, concentration(time, n), label=f"n={n}")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.show()