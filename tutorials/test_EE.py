import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
from shannon import embedding_entropy


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 20


@njit(cache=True, nogil=True)
def coupling_system(nt, x0, y0, bxy, byx, gx=3.7e0, gy=3.72e0):
  x = np.zeros(nt+1)
  y = np.zeros(nt+1)
  x[0] = x0
  y[0] = y0
  for i in range(nt):
    x[i+1] = x[i] * (gx - (gx - byx) * x[i] - byx * y[i]) + np.random.normal(0.e0, 0.01e0)
    y[i+1] = y[i] * (gy - (gy - bxy) * y[i] - bxy * x[i]) + np.random.normal(0.e0, 0.01e0)
  return x, y


def test_coupling_system(nt, n, byx, x0, y0, trial):
  bxy = np.linspace(0.e0, 0.3e0, n)
  EExy = np.zeros(n)
  EEyx = np.zeros(n)
  for j in tqdm(range(n)):
    for i in range(trial):
      x, y = coupling_system(nt, x0, y0, bxy[j], byx)
      EEyx[j] += embedding_entropy(x, y, p=1, tau=1, k=5, Thei=10)
      EExy[j] += embedding_entropy(y, x, p=1, tau=1, k=5, Thei=10)
    EExy[j] /= trial
    EEyx[j] /= trial
  plt.figure(figsize=(6,6))
  plt.plot(bxy, EExy, color='blue', linestyle='solid')
  plt.plot(bxy, EEyx, color='red',  linestyle='solid')
  plt.xlabel(r'$\beta_{xy}$', fontsize=20, style='italic')
  plt.ylabel("EE", fontsize=20)
  plt.ylim([-0.05, 0.9])
  plt.show()


test_coupling_system(nt=500, n=31, byx=0.1e0, x0=0.4e0, y0=0.6e0, trial=1)

