import numpy as np
import matplotlib.pyplot as plt
from ouxinfo import information_flow_1d


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 20


# --- generate 1D driven AR chain -------------------------------------------
# cell i drives cell i+1 unidirectionally (left → right)
nx    = 6
nt    = 3000
alpha = 0.5e0
beta  = 0.5e0
sigma = 1.0e0

rng  = np.random.default_rng(42)
data = np.zeros((nx, nt + 1))
data[:, 0] = rng.normal(0.e0, sigma, nx)
for t in range(nt):
  eps = rng.normal(0.e0, sigma, nx)
  data[0, t + 1] = alpha * data[0, t] + eps[0]
  for i in range(1, nx):
    data[i, t + 1] = alpha * data[i, t] + beta * data[i - 1, t] + eps[i]
data = data[:, 1:]  # shape (nx, nt)

# --- compute 1D information flux -------------------------------------------
result = information_flow_1d(data, dt=1.e0, tau=1, k=5, n_jobs=4)

# interface positions i + 1/2
x_iface = np.arange(nx - 1) + 0.5e0

# --- plot 1: spatiotemporal map --------------------------------------------
plt.figure(figsize=(8, 4))
plt.imshow(data[:, :200], aspect='auto', origin='lower', cmap='RdBu_r',
           extent=[0, 200, -0.5e0, nx - 0.5e0])
plt.xlabel(r'$t$', fontsize=20, style='italic')
plt.ylabel(r'$i$', fontsize=20, style='italic')
plt.colorbar(label=r'$q_i(t)$')
plt.show()

# --- plot 2: directional fluxes --------------------------------------------
plt.figure(figsize=(7, 7))
plt.plot(x_iface, result['J_fwd'], color='blue',  linestyle='solid',  label=r'$J_{i \to i+1}$')
plt.plot(x_iface, result['J_bwd'], color='red',   linestyle='solid',  label=r'$J_{i+1 \to i}$')
plt.plot(x_iface, result['J_net'], color='black', linestyle='dashed', label=r'$J^{\rm net}$')
plt.axhline(0.e0, color='gray', linestyle='dotted')
plt.xlabel(r'$i + \frac{1}{2}$', fontsize=20)
plt.ylabel(r'information flux', fontsize=20)
plt.legend(frameon=False)
plt.show()

# --- plot 3: coupling strength vs leak -------------------------------------
plt.figure(figsize=(7, 7))
plt.plot(x_iface, result['J_sym'], color='blue', linestyle='solid',  label=r'$J^{\rm sym}$')
plt.plot(x_iface, result['Leak'],  color='red',  linestyle='dashed', label=r'Leak')
plt.axhline(0.e0, color='gray', linestyle='dotted')
plt.xlabel(r'$i + \frac{1}{2}$', fontsize=20)
plt.ylabel(r'information flux', fontsize=20)
plt.legend(frameon=False)
plt.show()
