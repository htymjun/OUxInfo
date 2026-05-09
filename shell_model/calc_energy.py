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
  # shell energy
  uconj = np.zeros((N, Nt))
  uconj[0,:] = np.real(u[:,0] * np.conj(u[:,0]))
  for i in range(1,N):
    uconj[i,:] = np.mean(np.real(u[:,:i] * np.conj(u[:,:i])), axis=1)
    max_val = np.max(uconj[i,:])
    uconj[i,:] /= max_val
  plt.figure(figsize=(7, 4))
  label = r'$\Sigma_{%d}$' % 5
  plt.plot(t / te[0], uconj[5,:],  label=label, color='black')
  label = r'$\Sigma_{%d}$' % 8
  plt.plot(t / te[0], uconj[8,:],  label=label, color='blue')
  label = r'$\Sigma_{%d}$' % 11
  plt.plot(t / te[0], uconj[11,:], label=label, color='red')
  plt.xlim(100,105)
  plt.ylim(0, 0.7)
  plt.xlabel(r'$t/T_\epsilon$')
  plt.ylabel(r'$\Sigma_i / \mathrm{max}(\Sigma_i)$')
  plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1), frameon=False)
  plt.tight_layout()
  plt.show()
  plt.close()
  file_path = os.path.join(result_dir, model+"_energy.npz")
  np.savez(file_path, t=t, te=te, k=k, uconj=uconj)


post(model)

