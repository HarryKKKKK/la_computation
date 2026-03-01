import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "bench_random.csv"
OUT_DIR = "plots"
MAKE_PDF = False
LOGLOG = True

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)
required = {"op", "storage", "N", "time_per_call"}
missing = required - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns in CSV: {missing}")

df = df.sort_values(["op", "storage", "N"])

def savefig(path):
    plt.tight_layout()
    if MAKE_PDF:
        plt.savefig(path)
    else:
        plt.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close()

def plot_one_op(op_name):
    sub = df[df["op"] == op_name]
    if sub.empty:
        return

    plt.figure()
    for storage in sorted(sub["storage"].unique()):
        s = sub[sub["storage"] == storage]
        plt.plot(s["N"], s["time_per_call"], marker="o", label=storage)

    plt.xlabel("N")
    plt.ylabel("Time per call (seconds)")
    plt.title(op_name)

    if LOGLOG:
        plt.xscale("log")
        plt.yscale("log")

    plt.legend()
    savefig(os.path.join(OUT_DIR, f"{op_name}.pdf"))

def plot_summary(op_name, storages=("dense", "sparse")):
    sub = df[(df["op"] == op_name) & (df["storage"].isin(storages))]
    if sub.empty:
        return

    plt.figure()
    for storage in storages:
        s = sub[sub["storage"] == storage]
        if not s.empty:
            plt.plot(s["N"], s["time_per_call"], marker="o", label=storage)

    plt.xlabel("N")
    plt.ylabel("Time per call (seconds)")
    plt.title(f"{op_name}: dense vs sparse")

    if LOGLOG:
        plt.xscale("log")
        plt.yscale("log")

    plt.legend()
    savefig(os.path.join(OUT_DIR, f"summary_{op_name}.pdf"))

def main():
    ops = sorted(df["op"].unique())

    # 1) per-op plots
    for op in ops:
        plot_one_op(op)

    # 2) key summaries
    plot_summary("matvec")
    plot_summary("residual")

    print(f"Done. Plots saved in: {OUT_DIR}/")
    print("Key files:")
    print("  summary_matvec.pdf")
    print("  summary_residual.pdf")
    print("  speedup_matvec.pdf")
    print("  speedup_residual.pdf")

if __name__ == "__main__":
    main()