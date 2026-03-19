import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bench_performance_0.csv')

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 6))
vec_data = df[df['operation'] == 'vec_add']
data = vec_data[vec_data['storage'] == 'sparse']
plt.loglog(data['n'], data['execution_time_sec'], marker='o', label=f'Vector Add')
plt.title('Vector Addition Performance: $O(n)$ Complexity')
plt.xlabel('Number of Unknowns ($n$)')
plt.ylabel('Time (seconds)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('vector_addition.png')

plt.figure(figsize=(10, 6))
mat_add_data = df[df['operation'] == 'mat_add']
for storage in ['dense', 'sparse']:
    data = mat_add_data[mat_add_data['storage'] == storage]
    if not data.empty:
        plt.loglog(data['n'], data['execution_time_sec'], marker='s', label=f'Matrix Add ({storage})')
plt.title('Matrix Addition: Dense $O(n^2)$ vs Sparse $O(n)$')
plt.xlabel('Number of Unknowns ($n$)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('matrix_addition.png')

plt.figure(figsize=(10, 6))
matvec_data = df[df['operation'] == 'matvec']
for storage in ['dense', 'sparse']:
    data = matvec_data[matvec_data['storage'] == storage]
    if not data.empty:
        plt.loglog(data['n'], data['execution_time_sec'], marker='^', label=f'Mat-Vec Mult ({storage})')
plt.title('Matrix-Vector Multiplication Performance')
plt.xlabel('Number of Unknowns ($n$)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('matvec_comparison.png')

plt.figure(figsize=(10, 6))
res_data = df[df['operation'] == 'res_nominal']
for storage in ['dense', 'sparse']:
    data = res_data[res_data['storage'] == storage]
    if not data.empty:
        plt.loglog(data['n'], data['execution_time_sec'], marker='D', label=f'Residual ({storage.capitalize()})')

plt.title('Residual Evaluation Performance: Dense vs Sparse Storage')
plt.xlabel('Number of Unknowns ($n$)')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('residual_storage_comparison.png')

plt.show()