import numpy as np
from ouxinfo import shannon_entropy
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 20


N = 10000
S = np.linspace(1.e0, 5.e0, 100)

Ht = np.zeros_like(S)
Hn = np.zeros_like(S)

for i, s in enumerate(S):
  x = np.random.normal(0.e0, s, N)
  Ht[i] = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))
  Hn[i] = shannon_entropy(x.reshape(-1,1), k=5)

plt.figure(figsize=(7,7))
plt.plot(S, Ht, color='black', linestyle='solid')
plt.scatter(S, Hn, color='blue')
plt.xlabel(r'$\sigma$', fontsize=20, style='italic')
plt.ylabel("H(X)", fontsize=20)
plt.ylim([1.4, 3.2])
plt.show()
plt.close()

K  = np.linspace(3, 50, 48)
Ht = np.zeros_like(K)
Hn = np.zeros_like(K)
x  = np.random.normal(0.e0, 1.e0, N)
for i, k in enumerate(K):
  Hn[i] = shannon_entropy(x.reshape(-1,1), k=int(k))
Ht[:] = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * 1.e0**2))


plt.figure(figsize=(7,7))
plt.plot(K, Ht, color='black', linestyle='solid')
plt.scatter(K, Hn, color='blue')
plt.xlabel(r'$k$', fontsize=20, style='italic')
plt.ylabel("H(X)", fontsize=20)
plt.ylim([1.35, 1.45])
plt.show()
plt.close()

