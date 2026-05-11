import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from ._core import information_flow_causal_map


def process_2d(i, data, taus, dt, k_nn):
    pair = np.ascontiguousarray(data[i:i+2, :])
    IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k_nn)
    return i, IF[1, 0], IF[0, 1], Leak[1, 0]


def process_3d(k_idx, data, n_ifaces, taus, dt, k_nn):
    j_fwd_k = np.zeros(n_ifaces)
    j_bwd_k = np.zeros(n_ifaces)
    leak_k  = np.zeros(n_ifaces)
    for i in range(n_ifaces):
      pair = np.ascontiguousarray(data[k_idx, i:i+2, :])
      IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k_nn)
      j_fwd_k[i] = IF[1, 0]
      j_bwd_k[i] = IF[0, 1]
      leak_k[i]  = Leak[1, 0] # symmetric Leak[1,0] = Leak[0,1]
    return j_fwd_k, j_bwd_k, leak_k



def information_flux_1d(data, dt=1.0, tau=1, k=5, n_jobs=1):
    """
    Compute directed information flux across interfaces of a 1D spatial grid.

    For each neighboring cell interface i+½, evaluates both directional
    information flows and derives interface flux quantities.

    Parameters
    ----------
    data : ndarray, shape (nx, nt) or (nz, nx, nt)
        Spatiotemporal data. data[i, :] or data[:, i, :] is the time series at spatial cell i.
        Must be float64 (or convertible).
    dt : float, optional
        Physical time step between samples. Default 1.0.
    tau : int, optional
        Time delay. Default 1.
    k : int, optional
        Number of nearest neighbours for the KSG estimator. Default 5.
    n_threads : int, optional
        OpenMP thread count forwarded to information_flow_causal_map. Default 1.

    Returns
    -------
    dict
        All values are ndarray of shape (nx-1,), indexed by interface i:

        'J_fwd' : J_{i → i+1}
        'J_bwd' : J_{i+1 → i}
        'J_net' : J_fwd - J_bwd  (positive = left-to-right dominant)
        'J_sym' : 0.5*(J_fwd + J_bwd)  (bidirectional coupling strength)
        'Leak'  : leak associated with the forward direction
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
      nx, nt = data.shape
    elif data.ndim == 3:
      nz, nx, nt = data.shape
    else:
      raise ValueError(f"data must be 2D (nx, nt) or 3D (nz, nx, nt), got shape {data.shape}")
    if nx < 2:
      raise ValueError("data must have at least 2 spatial cells (nx >= 2)")

    n_ifaces = nx - 1
    taus = np.array([tau, tau], dtype=np.int32)

    J_fwd = np.zeros(n_ifaces)
    J_bwd = np.zeros(n_ifaces)
    Leak  = np.zeros(n_ifaces)

    if data.ndim == 2:
      results = Parallel(n_jobs=n_jobs)(
        delayed(process_2d)(i, data, taus, dt, k) for i in tqdm(range(n_ifaces))
      )
      for i, jf, jb, l in tqdm(results, total=n_ifaces, desc="Processing 2D"):
        J_fwd[i] = jf # cell i   → cell i+1
        J_bwd[i] = jb # cell i+1 → cell i
        Leak[i]  = l
    else:
      results = Parallel(n_jobs=n_jobs)(
        delayed(process_3d)(k_idx, data, n_ifaces, taus, dt, k) for k_idx in tqdm(range(nz))
      )
      for jf, jb, l in tqdm(results, total=nz, desc="Processing 3D"):
        J_fwd += jf
        J_bwd += jb
        Leak  += l
      J_fwd /= nz
      J_bwd /= nz
      Leak  /= nz
    return {
      'J_fwd': J_fwd,
      'J_bwd': J_bwd,
      'J_net': J_fwd - J_bwd,
      'J_sym': 0.5 * (J_fwd + J_bwd),
      'Leak' : Leak,
    }
