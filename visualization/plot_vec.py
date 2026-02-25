#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread

def plot_vector(filename, save=True):
    v = mmread(filename)

    v = np.array(v).flatten()

    n = len(v)
    x = np.arange(n)

    plt.figure(figsize=(6,4))
    plt.plot(x, v, marker='o')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Vector plot: {filename}")
    plt.grid(True)

    if save:
        outname = filename.replace(".mtx", ".png")
        plt.savefig(outname, dpi=150)
        print(f"Saved figure to {outname}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_vec.py file.mtx")
        sys.exit(1)

    plot_vector(sys.argv[1])