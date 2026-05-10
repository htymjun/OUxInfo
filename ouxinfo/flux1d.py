import numpy as np
from tqdm import tqdm
from ._core import information_flow_causal_map


def information_flux_1d(data, dt=1.0, tau=1, k=5, n_threads=1):
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

        'J_fwd'    : J_{i → i+1}
        'J_bwd'    : J_{i+1 → i}
        'J_net'    : J_fwd - J_bwd  (positive = left-to-right dominant)
        'J_sym'    : 0.5*(J_fwd + J_bwd)  (bidirectional coupling strength)
        'Leak_fwd' : leak associated with the forward direction
        'Leak_bwd' : leak associated with the backward direction (== Leak_fwd)
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

    J_fwd    = np.zeros(n_ifaces)
    J_bwd    = np.zeros(n_ifaces)
    Leak_fwd = np.zeros(n_ifaces)
    Leak_bwd = np.zeros(n_ifaces)

    if data.ndim == 2:
      for i in tqdm(range(n_ifaces)):
        # Pair shape (2, nt): variable 0 = cell i, variable 1 = cell i+1
        pair = np.ascontiguousarray(data[i:i+2, :])
        IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k, n_threads=n_threads)
        # IF[j, i] = flow from variable i to variable j
        J_fwd[i]    = IF[1, 0]    # cell i   → cell i+1
        J_bwd[i]    = IF[0, 1]    # cell i+1 → cell i
        Leak_fwd[i] = Leak[1, 0]
        Leak_bwd[i] = Leak[0, 1]  # symmetric: always equal to Leak_fwd[i]
    else:
      for k in tqdm(range(nz)):
        for i in range(n_ifaces):
          # Pair shape (2, nt): variable 0 = cell i, variable 1 = cell i+1
          pair = np.ascontiguousarray(data[k, i:i+2, :])
          IF, Leak, _ = information_flow_causal_map(pair, taus, dt=dt, k=k, n_threads=n_threads)
          # IF[j, i] = flow from variable i to variable j
          J_fwd[i]    += IF[1, 0]    # cell i   → cell i+1
          J_bwd[i]    += IF[0, 1]    # cell i+1 → cell i
          Leak_fwd[i] += Leak[1, 0]
          Leak_bwd[i] += Leak[0, 1]  # symmetric: always equal to Leak_fwd[i]
      J_fwd /= nz
      J_bwd /= nz
      Leak_fwd /= nz
      Leak_bwd /= nz
    return {
      'J_fwd':    J_fwd,
      'J_bwd':    J_bwd,
      'J_net':    J_fwd - J_bwd,
      'J_sym':    0.5 * (J_fwd + J_bwd),
      'Leak_fwd': Leak_fwd,
      'Leak_bwd': Leak_bwd,
    }
