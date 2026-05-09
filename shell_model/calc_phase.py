import numpy as np
import os
import time
import matplotlib.pyplot as plt
from ouxinfo import myParams


myParams(fontsize=16)


model = "sabra"


def post(model, init=0.01):
  result_dir = "./output"
  data = np.load(os.path.join(result_dir, model+".npz")) 
  u      = data["u"]
  k      = data["k"]
  t      = data["t"]
  params = data["params"]
  Nt = u.shape[0]
  u  = u[int(init * Nt)::2]
  t  = t[int(init * Nt)::2]
  dt = -t[0] + t[1]
  Nt, N = u.shape
  # Time scale
  te = np.zeros(N)
  for i in range(N):
    te[i] = np.real(1.e0 / (k[i] * np.mean(np.abs(u[:,i]))))
  # shell phase
  phase = np.zeros((N, Nt, 2))
  for i in range(N):
    theta = np.angle(u[:,i])
    phase[i,:,0] = np.cos(theta)
    phase[i,:,1] = np.sin(theta)
  plt.figure(figsize=(7, 4))
  plt.plot(t / te[0], phase[5,:,0],  color='black', linestyle='solid')
  plt.plot(t / te[0], phase[5,:,1],  color='black', linestyle='dashed')
  plt.xlim(100,105)
  plt.ylim(-1.1, 1.1)
  plt.xlabel(r'$t/T_\epsilon$')
  plt.ylabel(r'$sin(\theta_i), cos(\theta_i)$')
  plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1), frameon=False)
  plt.tight_layout()
  plt.show()
  plt.close()
  file_path = os.path.join(result_dir, model+"_phase.npz")
  np.savez(file_path, t=t, te=te, k=k, phase=phase)


post(model)

