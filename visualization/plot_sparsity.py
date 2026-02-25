import sys
import numpy as np
import matplotlib.pyplot as plt


def read_mtx(path):
    rows = []
    cols = []

    with open(path, "r") as f:
        # skip header/comments
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                nrows, ncols, nnz = map(int, parts)
                break

        # read triplets
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            i, j, v = parts
            rows.append(int(i) - 1)
            cols.append(int(j) - 1)

    return rows, cols


if __name__ == "__main__":
    # ----------------------------
    # Handle command line args
    # ----------------------------
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    else:
        input_path = "A.mtx"

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = "sparsity.png"

    rows, cols = read_mtx(input_path)

    plt.figure()
    plt.scatter(cols, rows, s=4)
    plt.gca().invert_yaxis()
    plt.xlabel("column j")
    plt.ylabel("row i")
    plt.title("Sparsity pattern (nonzeros)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Saved figure to {output_path}")
