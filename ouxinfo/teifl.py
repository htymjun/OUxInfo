import numpy as np
import os
import time
import matplotlib.pyplot as plt
from ._core import transfer_entropy, transfer_entropy_causal_map
from ._core import information_flow_causal_map
from .backwardTE import backward_transfer_entropy


def TEIFL(X, tau=1, dt=1.e0, lag=1, max_m=10, tol=0.01e0, k=5, n_threads=1, result_dir='.'):
  '''
  Parameters
  ----------
  X          : ndarray (N, dim)
  tau        : int, optional
               Length of time delay
  dt         : double, optional
               Physical time
  lag        : int, optional
               Time lag for embedding
  max_m      : int, optional
               Maximum embedding dimension
  tol        : int, optional
               Tolerance for mTE calculation
  k          : int, optional
               Number of nearest neighbors.
  n_threads  : int, optional
               The number of trials for surrogate analysis.
  result_dir : str, optional
               The path of directory to save result.
  Returns
  -------
  numpy binary
    Saves results to a compressed file in result_dir.
  '''
  # Transfer Entropy, Information Flow, Leak
  # sTE > mTE > IF
  # error check
  if max_m < 2:
    raise InvalidInputError("m must be greater than 2")
  N = X.shape[0]
  # time delay
  taus = np.zeros(N, dtype=np.int32)
  taus[:] = tau
  # single-time step transfer entropy
  print("calc single-time step transfer entropy")
  t_start = time.time()
  sTE = transfer_entropy_causal_map(X, taus, dt=dt, k=k, n_threads=n_threads)
  t_end = time.time()
  print("Elapsed time:", t_end - t_start, " [s]")
  # multi-time step transfer entropy
  print("calc multi-time step transfer entropy & backward transfer entropy")
  t_start = time.time()
  mTE  = np.zeros_like(sTE)
  bTE  = np.zeros_like(sTE)
  mmap = np.zeros_like(sTE, dtype=np.int32)
  for j in range(N):
    for i in range(N):
      if i == j:
        mTE[j,i] = np.nan
        bTE[j,i] = np.nan
      else:
        xj_ = X[j].reshape(-1,1) if X[j].ndim == 1 else X[j]
        xi_ = X[i].reshape(-1,1) if X[i].ndim == 1 else X[i]
        xj_ = np.ascontiguousarray(xj_)
        xi_ = np.ascontiguousarray(xi_)
        # forward multi-time step transfer entropy
        prev_tmp = sTE[j,i]
        for m in range(2, max_m+1):
          tmp = transfer_entropy(xi_, xj_, tau=tau, m=m, lag=lag, dt=dt, k=k)
          mTE[j,i] = tmp
          if prev_tmp > 1e-10:
            decrease_rate = (prev_tmp - tmp) / prev_tmp
            if decrease_rate <= tol:
              break
          prev_tmp  = tmp
          mmap[j,i] = m
        # backward multi-time step transfer entropy
        bTE[j,i] = backward_transfer_entropy(xi_, xj_, tau=tau, m=m, lag=lag, dt=dt, k=k)
  t_end = time.time()
  print("Elapsed time:", t_end - t_start, " [s]")
  # information flow
  print("calc information flow")
  t_start = time.time()
  IF, Leak, dI = information_flow_causal_map(X, taus, dt=dt, k=k, n_threads=n_threads)
  t_end = time.time()
  print("Elapsed time:", t_end - t_start, " [s]")
  os.makedirs(result_dir, exist_ok=True)
  file_path = os.path.join(result_dir, 'teifl.npz')
  np.savez_compressed(file_path, sTE=sTE, mTE=mTE, bTE=bTE, mmap=mmap, IF=IF, Leak=Leak, dI=dI)


def plot_TEIFL(file_path, labels):
  '''
  Parameters
  ----------
  file_path : str
              The path of directory to load result.
  labels    : str
              The labels for plot causal maps.
  Returns
  -------
  numpy binary
    Saves results in result_dir.
  '''
  result_dir = os.path.dirname(file_path)
  teifl = np.load(file_path)
  sTE  = teifl['sTE']
  mTE  = teifl['mTE']
  bTE  = teifl['bTE']
  mmap = teifl['mmap']
  IF   = teifl['IF']
  Leak = teifl['Leak']
  dI   = teifl['dI']
  N = sTE.shape[0]
  ticks=np.linspace(0, N-1, N)
  # colormap
  bone = plt.cm.bone_r.copy()
  bone.set_bad(color='#F4B9B9') # pink
  RdBu = plt.cm.RdBu_r.copy()
  RdBu.set_bad(color='black')
  # single-time step transfer entropy
  plt.imshow(sTE, cmap=bone, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$sTE$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "sTE.png"))
  plt.close()
  # dimention of multi-time step transfer entropy
  plt.imshow(mmap, cmap=bone, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$m$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "m.png"))
  plt.close()
  # multi-time step transfer entropy
  plt.imshow(mTE, cmap=bone, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$mTE$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "mTE.png"))
  plt.close()
  # transfer entropy - backward transfer entropy
  val = np.nanmax(np.abs(mTE - bTE))
  plt.imshow(mTE - bTE, vmin=-val, vmax=val, cmap=RdBu, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label='$\\Delta fTE - \\Delta bTE$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "fTE_bTE.png"))
  plt.close()
  # information flow
  val = np.nanmax(np.abs(IF))
  plt.imshow(IF, vmin=-val, vmax=val, cmap=RdBu, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$IF$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "IF.png"))
  plt.close()
  # leak
  val = np.nanmax(np.abs(Leak))
  plt.imshow(Leak, vmin=-val, vmax=val, cmap=RdBu, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$Leak$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "Leak.png"))
  plt.close()
  # dI
  val = np.nanmax(np.abs(dI))
  plt.imshow(dI, vmin=-val, vmax=val, cmap=RdBu, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$dI$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "dI.png"))
  plt.close()
  # sensor capacity
  sSC = np.divide(IF, sTE, where=~np.isnan(sTE))
  sSC[np.isnan(sTE)] = np.nan
  plt.imshow(sSC, vmin=-1.2, vmax=1.2, cmap=RdBu, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$sSC$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "sSC.png"))
  plt.close()
  # sensor capacity
  mSC = np.divide(IF, mTE, where=~np.isnan(mTE))
  mSC[np.isnan(mTE)] = np.nan
  plt.imshow(mSC, vmin=-1.2, vmax=1.2, cmap=RdBu, extent=None, origin='lower')
  plt.xticks(ticks=ticks, labels=labels)
  plt.yticks(ticks=ticks, labels=labels)
  plt.colorbar(label=r'$mSC$')
  plt.tight_layout()
  plt.savefig(os.path.join(result_dir, "mSC.png"))
  plt.close()

