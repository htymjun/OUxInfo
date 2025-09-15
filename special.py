import numpy as np
import os
import time
from scipy.special import psi
import matplotlib.pyplot as plt

X = np.linspace(0.1e0, 10.e0, 100)
Psi = np.zeros_like(X)

t1 = time.time()
for itr, x in enumerate(X):
  Psi[itr] = psi(x)
t2 = time.time()
print("Elapsed time:", t2-t1)

save_path = os.path.join("psi_scipy.d")
with open(save_path, "w", encoding="UTF-8") as f:
  print("# x      psi", file=f)
  for itr, x in enumerate(X):
    print(f'{x:.3e}', f'{Psi[itr]:.3e}', file=f)

