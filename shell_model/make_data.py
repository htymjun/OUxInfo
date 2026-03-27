import numpy as np
import os
import time
import matplotlib.pyplot as plt
from shell_model import ShellModel


# model params
model = "sabra"
N     = 24
nu    = 1e-8
k0    = 2.e0**(-4)
lam   = 2.e0
eps   = 0.5e0
f0    = (1.e0 + 1.j)*5e-3
# Time evelopment
dt    = 1.e-4
endT  = 2.e4
Np    = 100000


def run(N, nu, k0, lam, eps, f0, dt, endT, Np, model):
  k     = np.array([k0 * lam**n for n in range(N)])
  kmax  = k[-1]
  Nt    = int(endT / dt)
  Model = ShellModel(N=N, nu=nu, k0=k0, lam=lam, eps=eps, f0=f0, model=model)
  t1 = time.time()
  u  = Model.RK44(Nt, Np, dt)
  t2 = time.time()
  print("Elapsed time: ", t2-t1, " [s]")
  t  = np.linspace(0, endT, Np + 1)
  params = (N, nu, k0, lam, eps, f0)
  result_dir = "./output"
  os.makedirs(result_dir, exist_ok=True)
  np.savez(os.path.join(result_dir, model+".npz"), u=u, k=k, t=t, params=params)


run(N, nu, k0, lam, eps, f0, dt, endT, Np, model)

