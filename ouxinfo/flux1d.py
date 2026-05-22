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


def process_x_3d(j, data, n_ifaces, taus, dt, k_nn):
    j_fwd = np.zeros(n_ifaces)
    j_bwd = np.zeros(n_ifaces)
    leak  = np.zeros(n_ifaces)
    for i in range(n_ifaces):
      pair = np.ascontiguousarray(data[j, i:i+2, :])
      IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k_nn)
      j_fwd[i] = IF[1, 0]
      j_bwd[i] = IF[0, 1]
      leak[i]  = Leak[1, 0]
    return j, j_fwd, j_bwd, leak


def process_y_3d(i, data, n_jfaces, taus, dt, k_nn):
    j_fwd = np.zeros(n_jfaces)
    j_bwd = np.zeros(n_jfaces)
    leak  = np.zeros(n_jfaces)
    for j in range(n_jfaces):
      pair = np.ascontiguousarray(data[j:j+2, i, :])
      IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k_nn)
      j_fwd[j] = IF[1, 0]
      j_bwd[j] = IF[0, 1]
      leak[j]  = Leak[1, 0]
    return i, j_fwd, j_bwd, leak


def process_x_4d(k_idx, data, ny, n_ifaces, taus, dt, k_nn):
    j_fwd = np.zeros((ny, n_ifaces))
    j_bwd = np.zeros((ny, n_ifaces))
    leak  = np.zeros((ny, n_ifaces))
    for j in range(ny):
      for i in range(n_ifaces):
        pair = np.ascontiguousarray(data[k_idx, j, i:i+2, :])
        IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k_nn)
        j_fwd[j, i] = IF[1, 0]
        j_bwd[j, i] = IF[0, 1]
        leak[j, i]  = Leak[1, 0]
    return j_fwd, j_bwd, leak


def process_y_4d(k_idx, data, n_jfaces, nx, taus, dt, k_nn):
    j_fwd = np.zeros((n_jfaces, nx))
    j_bwd = np.zeros((n_jfaces, nx))
    leak  = np.zeros((n_jfaces, nx))
    for j in range(n_jfaces):
      for i in range(nx):
        pair = np.ascontiguousarray(data[k_idx, j:j+2, i, :])
        IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k_nn)
        j_fwd[j, i] = IF[1, 0]
        j_bwd[j, i] = IF[0, 1]
        leak[j, i]  = Leak[1, 0]
    return j_fwd, j_bwd, leak


def information_flow_1d(data, dt=1.0, tau=1, k=5, n_jobs=1):
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
    n_jobs : int, optional
        Number of parallel jobs (joblib). Default 1.
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


def information_flow_2d(data, dt=1.0, tau=1, k=5, n_jobs=1, x1=None, x2=None):
    """
    Compute directed information flux across interfaces of a 2D spatial grid.
    For each neighboring cell interface i+½ and j+½, evaluates both directional
    information flows and derives interface flux quantities.
    Parameters
    ----------
    data : ndarray, shape (ny, nx, nt) or (nz, ny, nx, nt)
        Spatiotemporal data. data[j,i,:] or data[:,j,i,:] is the time series at spatial cell (i,j).
        Must be float64 (or convertible).
    dt : float, optional
        Physical time step between samples. Default 1.0.
    tau : int, optional
        Time delay. Default 1.
    k : int, optional
        Number of nearest neighbours for the KSG estimator. Default 5.
    n_jobs : int, optional
        Number of parallel jobs (joblib). Default 1.
    Returns
    -------
    dict
        Values are ndarray of shape (ny,nx-1), indexed by interface i:
        'Jx_fwd' : J_{i → i+1} [ny,nx-1]
        'Jx_bwd' : J_{i+1 → i} [ny,nx-1]
        'Jx_net' : J_fwd - J_bwd  (positive = left-to-right dominant)
        'Jx_sym' : 0.5*(J_fwd + J_bwd)  (bidirectional coupling strength)
        'Leakx'  : leak associated with the forward direction
        Values are ndarray of shape (ny-1,nx), indexed by interface j:
        'Jy_fwd' : J_{j → j+1} [ny-1,nx]
        'Jy_bwd' : J_{j+1 → j} [ny-1,nx]
        'Jy_net' : J_fwd - J_bwd  (positive = down-to-top dominant)
        'Jy_sym' : 0.5*(J_fwd + J_bwd)  (bidirectional coupling strength)
        'Leaky'  : leak associated with the forward direction
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 3:
      ny, nx, nt = data.shape
    elif data.ndim == 4:
      nz, ny, nx, nt = data.shape
    else:
      raise ValueError(f"data must be 3D (ny, nx, nt) or 4D (nz, ny, nx, nt), got shape {data.shape}")
    if nx < 2 or ny < 2:
      raise ValueError("data must have at least 2 spatial cells (nx >= 2 and ny >= 2)")

    n_ifaces = nx - 1
    n_jfaces = ny - 1
    taus = np.array([tau, tau], dtype=np.int32)

    Jx_fwd = np.zeros((ny,n_ifaces))
    Jx_bwd = np.zeros((ny,n_ifaces))
    Leakx  = np.zeros((ny,n_ifaces))
    Jy_fwd = np.zeros((n_jfaces,nx))
    Jy_bwd = np.zeros((n_jfaces,nx))
    Leaky  = np.zeros((n_jfaces,nx))

    if data.ndim == 3:
      # X direction flux (parallel over y-axis)
      results_x = Parallel(n_jobs=n_jobs)(
        delayed(process_x_3d)(j, data, n_ifaces, taus, dt, k) for j in tqdm(range(ny), desc="X-flux 3D")
      )
      for j, jf, jb, lk in results_x:
        Jx_fwd[j, :] = jf
        Jx_bwd[j, :] = jb
        Leakx[j, :]  = lk
      # Y direction flux (parallel over x-axis)
      results_y = Parallel(n_jobs=n_jobs)(
        delayed(process_y_3d)(i, data, n_jfaces, taus, dt, k) for i in tqdm(range(nx), desc="Y-flux 3D")
      )
      for i, jf, jb, lk in results_y:
        Jy_fwd[:, i] = jf
        Jy_bwd[:, i] = jb
        Leaky[:, i]  = lk
    else: # data.ndim == 4
      # X direction flux (parallel over z-axis/ensemble)
      results_x = Parallel(n_jobs=n_jobs)(
        delayed(process_x_4d)(k_idx, data, ny, n_ifaces, taus, dt, k) for k_idx in tqdm(range(nz), desc="X-flux 4D")
      )
      for jf, jb, lk in results_x:
        Jx_fwd += jf
        Jx_bwd += jb
        Leakx  += lk
      Jx_fwd /= nz
      Jx_bwd /= nz
      Leakx  /= nz
      # Y direction flux (parallel over z-axis/ensemble)
      results_y = Parallel(n_jobs=n_jobs)(
        delayed(process_y_4d)(k_idx, data, n_jfaces, nx, taus, dt, k) for k_idx in tqdm(range(nz), desc="Y-flux 4D")
      )
      for jf, jb, lk in results_y:
        Jy_fwd += jf
        Jy_bwd += jb
        Leaky  += lk
      Jy_fwd /= nz
      Jy_bwd /= nz
      Leaky  /= nz
    ################################################################
    if x1 is None or x2 is None:
      return {
        'Jx_fwd': Jx_fwd,
        'Jx_bwd': Jx_bwd,
        'Jx_net': Jx_fwd - Jx_bwd,
        'Jx_sym': 0.5 * (Jx_fwd + Jx_bwd),
        'Leakx' : Leakx,
        'Jy_fwd': Jy_fwd,
        'Jy_bwd': Jy_bwd,
        'Jy_net': Jy_fwd - Jy_bwd,
        'Jy_sym': 0.5 * (Jy_fwd + Jy_bwd),
        'Leaky' : Leaky,
      }
    else:
      return {
        'Jx_fwd': Jx_fwd,
        'Jx_bwd': Jx_bwd,
        'Jx_net': Jx_fwd - Jx_bwd,
        'Jx_sym': 0.5 * (Jx_fwd + Jx_bwd),
        'Leakx' : Leakx,
        'Jy_fwd': Jy_fwd,
        'Jy_bwd': Jy_bwd,
        'Jy_net': Jy_fwd - Jy_bwd,
        'Jy_sym': 0.5 * (Jy_fwd + Jy_bwd),
        'Leaky' : Leaky,
        'x1'    : x1,
        'x2'    : x2,
      }

