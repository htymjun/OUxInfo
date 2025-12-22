import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from ouxinfo import transfer_entropy, information_flux


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 20
cm   = plt.cm.bone_r.copy()
cm.set_bad(color='#F4B9B9') # pink


# Load data
path = './data/energy_cascade_signals.mat'
loaddata = loadmat(path)
X = loaddata['X']

Nv, Nt = X.shape

tau = int(0.046e0 / 165.e0 * Nt) # tau = 0.046Te, Lt = 165Te
TE  = np.zeros((Nv,Nv))
IF  = np.zeros((Nv,Nv))
for j in tqdm(range(Nv)):
  for i in range(Nv):
    if i == j:
      TE[j,i] = np.nan
      IF[j,i] = np.nan
    else:
      # causality from i -> j
      TE[j,i] = transfer_entropy(X[i,:].reshape(-1,1), \
                                 X[j,:].reshape(-1,1), k=5, tau=tau, trial=1)
      IF[j,i] = information_flux(X[i,:].reshape(-1,1), \
                                 X[j,:].reshape(-1,1), k=5, tau=tau)

plt.imshow(TE, cmap=cm, extent=None, origin='lower')
plt.xticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.yticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.colorbar(label=r'TE')
plt.show()

plt.imshow(IF, cmap=cm, extent=None, origin='lower')
plt.xticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.yticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.colorbar(label=r'dI/dt')
plt.show()

