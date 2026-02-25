import numpy as np
import matplotlib.pyplot as plt

rows = []
cols = []

with open("A.mtx", "r") as f:
    for line in f:
        if line.startswith("%"):
            continue
        parts = line.strip().split()
        # first non-comment line is size: nrows ncols nnz
        nrows, ncols, nnz = map(int, parts)
        break

    for line in f:
        i, j, v = line.split()
        rows.append(int(i) - 1)  # 0-based
        cols.append(int(j) - 1)

plt.figure()
plt.scatter(cols, rows, s=4)
plt.gca().invert_yaxis()
plt.xlabel("column j")
plt.ylabel("row i")
plt.title("Sparsity pattern (nonzeros)")
plt.show()