import sys
import numpy as np
import matplotlib.pyplot as plt


def read_mtx_array(path):
    with open(path, "r") as f:
        header = f.readline().strip()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError("Not a MatrixMarket file")

        # example: %%MatrixMarket matrix array real general
        parts = header.split()
        if len(parts) < 5:
            raise ValueError("Bad MatrixMarket header")
        fmt = parts[2].lower()
        if fmt != "array":
            raise ValueError(f"Expected array format, got {fmt}")

        # skip comments
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()

        nrows, ncols = map(int, line.split())

        data = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            data.append(float(s))

    A = np.array(data, dtype=float)
    if A.size != nrows * ncols:
        raise ValueError("Data size mismatch in array MTX")

    A = A.reshape((nrows, ncols), order="F")  # column-major (MatrixMarket array)
    return A


# def plot_vector(v, out_path="vector_plot.png", title="Vector plot"):
#     v = v.reshape(-1)
#     idx = np.arange(v.size)

#     plt.figure()
#     plt.plot(idx, v, marker="o")
#     plt.xlabel("Index")
#     plt.ylabel("Value")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()

def plot_vector(v, out_path="vector_heatmap.png", title="Vector heatmap"):
    v = v.reshape(-1, 1)
    cmap = plt.cm.viridis.copy()
    v_masked = np.ma.masked_where(v == 0, v)
    cmap.set_bad("white")

    plt.figure(figsize=(2.5, 6))
    im = plt.imshow(v_masked, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im)
    plt.xticks([0], ["v"])
    plt.yticks(np.arange(v.shape[0]))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "x_true.mtx"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "vector_plot.png"

    A = read_mtx_array(in_path)

    # treat n×1 or 1×n as vector
    if A.shape[1] == 1:
        v = A[:, 0]
    elif A.shape[0] == 1:
        v = A[0, :]
    else:
        raise ValueError(f"Not a vector: shape={A.shape}")

    plot_vector(v, out_path, title=f"Vector plot: {in_path}")
    print(f"Saved figure to {out_path}")