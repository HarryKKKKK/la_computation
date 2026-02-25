import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("x.csv", delimiter=",")
i = data[:,0]
x = data[:,1]

plt.figure()
plt.plot(i, x)
plt.xlabel("index i")
plt.ylabel("x[i]")
plt.title("Vector x")
plt.grid(True)
plt.show()