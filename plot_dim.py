import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df = pd.read_csv('bench_sparsity_study.csv')
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 6))

d_data = df[df['storage'] == 'dense']
s_data = df[df['storage'] == 'sparse']

plt.plot(d_data['fill_ratio'] * 100, d_data['time_sec'], 'r-o', label='Dense Matrix')
plt.plot(s_data['fill_ratio'] * 100, s_data['time_sec'], 'b-s', label='Sparse Matrix (CRS)')

ax = plt.gca()

plt.xscale('linear')
plt.yscale('linear') 

ax.set_xticks(range(0, 101, 10))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

plt.title('Performance Threshold: Dense vs Sparse Storage', fontsize=14)
plt.xlabel('Fill Ratio (Sparsity Percentage)', fontsize=12)
plt.ylabel('Execution Time per Ax (seconds)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.savefig('sparsity_threshold_analysis.png')
plt.show()