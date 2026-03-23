import numpy as np
import os
from ouxinfo import TEIFL, plot_TEIFL, myParams


myParams()

# load data
model      = "sabra" 
result_dir = "./output"
data       = np.load(os.path.join(result_dir, model+".npz")) 
u          = data["u"] 
k          = data["k"]
t          = data["t"]
params     = data["params"]
N, nu, k0, lam, eps, f0 = params
Nt = u.shape[0]
u  = u[int(0.5e0 * Nt):-1:2]
t  = t[int(0.5e0 * Nt):-1:2]
Nt = u.shape[0]
t -= t[0]
dt = -t[0] + t[1]

offset = 7
u = np.transpose(u[:,offset:-offset])
k = k[offset:-offset]
N = u.shape[0]

uconj = np.zeros((N,Nt), dtype=np.float64)
label = np.empty((N),    dtype=object)
for i in range(N):
  label[i] = r'$k_{%d}$' % (i+offset)
  for itr in range(Nt):
    uconj[i,itr] = np.real(u[i,itr] * np.conj(u[i,itr]))

# time delay
print("Nt=", Nt, " Time delay")
for i in range(N):
  te = 1.e0 / (k[i] * np.mean(np.abs(u[:,i])))
  print("k:", i, ", Time scale:", te / dt)

for tau in range(1, 6):
  result_dir = f"teifl_tau_{tau}"
  file_path  = os.path.join(result_dir, "teifl.npz")
  TEIFL(uconj, tau=tau, dt=dt, lag=1, max_m=10, tol=0.1, k=5, n_threads=4, result_dir=result_dir)
  plot_TEIFL(file_path, label)
 
