import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('bench_dense_final.csv')
plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Execution Time Scaling
plt.figure(figsize=(10, 6))
nom_data = df[df['operation'] == 'nominal']
plt.loglog(nom_data['n'], nom_data['execution_time_sec'], 'r-o', linewidth=2, label='Nominal (Serial)')

for p in [1, 2, 4, 8]:
    p_data = df[(df['operation'] == 'optimised') & (df['threads'] == p)]
    plt.loglog(p_data['n'], p_data['execution_time_sec'], '--', label=f'Optimised (p={p})')

plt.title('Dense Matrix Residual: Execution Time Scaling', fontsize=14)
plt.xlabel('Number of Unknowns (n)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('dense_time_scaling.png')
plt.show()

# Figure 2: Speedup Analysis
plt.figure(figsize=(10, 6))
opt_p1 = df[(df['operation'] == 'optimised') & (df['threads'] == 1)].set_index('n')['execution_time_sec']

for p in [2, 4, 8]:
    opt_p = df[(df['operation'] == 'optimised') & (df['threads'] == p)].set_index('n')['execution_time_sec']
    speedup = opt_p1 / opt_p
    plt.plot(opt_p.index, speedup, '-s', label=f'Speedup (p={p})')

plt.axhline(y=1, color='black', linestyle='-', alpha=0.3)
plt.title('Parallel Speedup Analysis: Optimised Implementation', fontsize=14)
plt.xlabel('Number of Unknowns (n)', fontsize=12)
plt.ylabel('Speedup Factor ($S_p$)', fontsize=12)
plt.setp(plt.gca(), xscale='log')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig('dense_speedup_analysis.png')
plt.show()