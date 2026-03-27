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
  u  = u[int(init * Nt):]
  t  = t[int(init * Nt):]
  dt = -t[0] + t[1]
  _, nu, k0, lam, eps, f0 = params
  Nt, N = u.shape
  # Kolmogorov scale
  eps = np.mean(nu * np.sum(k**2 * abs(u)**2, axis=1))
  ue  = (nu * eps) ** 0.25
  te  = np.real(1.e0 / (k[0] * np.mean(np.abs(u[:,0]))))
  kd  = np.real((eps / nu**3) ** 0.25)
  # shell velocity
  plt.figure(figsize=(7, 4))
  colors = ['black', 'blue', 'darkgreen', 'red', 'orange']
  for i in range(7, 12):
    label = r'$u_{%d}$' % i
    plt.plot(t / te, np.real(u[:, i]), label=label, color=colors[i-7])
  plt.xlabel(r'$t / T_\epsilon$')
  plt.ylabel(r'$\mathrm{Real}(u_n)$')
  plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
  plt.tight_layout()
  plt.show()
  plt.close()
  # total energy
  Et = np.zeros(Nt)
  plt.figure(figsize=(6, 4))
  for i in range(N):
    Et += abs(u[:,i] / ue)**2
  plt.plot(t / te, Et)
  plt.xlabel("$t / T_\\epsilon$")
  plt.ylabel(f"$E_t$")
  plt.tight_layout()
  plt.show()
  plt.close()
  # energy spectra
  E = np.zeros(N)
  for n in range(N):
    E[n] = np.mean(np.abs(u[:,n] / ue)**2)
  pow_law = 10000 * k ** (-2.e0 / 3.e0)
  plt.figure(figsize=(6, 6))
  plt.loglog(k/kd, E,       color='blue',  label='Energy spectra', marker='o')
  plt.loglog(k/kd, pow_law, color='black', label='-5/3 slope')
  plt.xlabel(f"$k / k_d$")
  plt.ylabel(f"$E$")
  plt.xlim(1e-7, 1e1)
  plt.xticks(10.e0**np.arange(-7,2,1))
  plt.legend(frameon=False)
  plt.show()
  plt.close()


post(model)

