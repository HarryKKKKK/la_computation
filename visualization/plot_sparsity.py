import sys
import numpy as np
import matplotlib.pyplot as plt


def read_mtx(path):
    rows = []
    cols = []
    vals = []

    with open(path, "r") as f:
        # skip header/comments
        for line in f:
            if line.startswith("%"):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                nrows, ncols, nnz = map(int, parts)
                break

        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            i, j, v = parts
            rows.append(int(i) - 1)
            cols.append(int(j) - 1)
            vals.append(float(v))

    return nrows, ncols, np.array(rows), np.array(cols), np.array(vals)


def plot_matrix_heatmap(nrows, ncols, rows, cols, vals, output_path):
    # ---- build dense matrix
    A = np.zeros((nrows, ncols))
    A[rows, cols] = vals

    # ---- create mask: True where value is zero
    mask = (A == 0)

    # ---- create colormap
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')   # masked values appear white

    # ---- masked array (zeros will be white)
    A_masked = np.ma.masked_where(mask, A)

    # ---- find min/max of nonzero entries
    nonzero_vals = vals
    vmin = np.min(nonzero_vals)
    vmax = np.max(nonzero_vals)

    # ---- plot
    plt.figure(figsize=(6,6))
    im = plt.imshow(A_masked,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin='upper',
                    interpolation='nearest')

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("column j")
    plt.ylabel("row i")
    plt.title("Matrix heatmap (zeros = white)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":

    input_path = sys.argv[1] if len(sys.argv) >= 2 else "A.mtx"
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "heatmap.png"

    nrows, ncols, rows, cols, vals = read_mtx(input_path)

    plot_matrix_heatmap(nrows, ncols, rows, cols, vals, output_path)

    print(f"Saved figure to {output_path}")