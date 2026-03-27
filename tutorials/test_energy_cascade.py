import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from ouxinfo import transfer_entropy_causal_map
from ouxinfo import information_flow_causal_map
from shannon import embedding_entropy_surrogate as embedding_entropy


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
X = X[:,:Nt//10]

tau = int(0.046e0 / 165.e0 * Nt) # tau = 0.046Te, Lt = 165Te
taus = np.zeros(Nv, dtype=np.int32)
taus[:] = tau

TE = transfer_entropy_causal_map(X, taus, m=3, lag=1, k=5)

plt.imshow(TE, cmap=cm, extent=None, origin='lower')
plt.xticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.yticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.colorbar(label=r'$TE$')
plt.show()

IF, _, _ = information_flow_causal_map(X, taus, k=5)

plt.imshow(IF, cmap=cm, vmin=0.e0, extent=None, origin='lower')
plt.xticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.yticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.colorbar(label=r'$dI/dt$')
plt.show()

'''
EE = np.zeros((Nv,Nv))
for j in tqdm(range(Nv)):
  for i in range(Nv):
    if i == j:
      EE[j,i] = np.nan
    else:
      EE[j,i] = embedding_entropy(X[j,:], X[i,:], p=8, tau=tau, k=5, trial=10)

plt.imshow(EE, cmap=cm, vmin=0.e0, extent=None, origin='lower')
plt.xticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.yticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.colorbar(label=r'$EE$')
plt.show()

dEE = np.zeros_like(EE)
for j in range(Nv):
  for i in range(Nv):
    if i == j:
      dEE[j,i] = np.nan
    else:
      dEE[j,i] = EE[j,i] - EE[i,j]

plt.imshow(dEE, cmap=cm, vmin=0.e0, vmax=0.6, extent=None, origin='lower')
plt.xticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.yticks(ticks=[0,1,2,3], labels=[r'$\Pi_{1}$', r'$\Pi_{2}$', r'$\Pi_{3}$', r'$\Pi_{4}$'])
plt.colorbar(label=r'$\Delta EE$')
plt.show()
'''
