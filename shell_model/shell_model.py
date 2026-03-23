import numpy as np
from numba import njit


class RK:
  @staticmethod
  @njit(cache=True, nogil=True, fastmath=True)
  def _RK44(RHS, N, params, Nt, Np, dt, x0):
    X = np.zeros((Np+1, N), dtype=np.complex128)
    X[0,:] = x0.copy()
    x   = x0.copy()
    k1  = np.zeros(N, dtype=np.complex128)
    k2v = np.zeros(N, dtype=np.complex128)
    k3  = np.zeros(N, dtype=np.complex128)
    k4  = np.zeros(N, dtype=np.complex128)
    step_per_save = Nt // Np
    one_sixth     = 1.e0 / 6.e0
    for j in range(Np):
      for _ in range(step_per_save):
        RHS(params, x, k1)
        RHS(params, x + 0.5e0 * dt * k1, k2v)
        RHS(params, x + 0.5e0 * dt * k2v, k3)
        RHS(params, x + dt * k3, k4)
        x += dt * (k1 + 2.e0 * k2v + 2.e0 * k3 + k4) * one_sixth
      X[j+1,:] = x
    return X


class ShellModel(RK):
  def __init__(self, N=24, nu=1e-8, k0=2.e0**(-4), lam=2.0,
               eps=0.5, f0=(1.e0+1.j)*5e-3, model=None):
    self.N     = N
    self.nu    = nu
    self.k     = np.array([k0 * lam**n for n in range(self.N)])
    self.k2_nu = self.k * self.k * self.nu
    self.eps   = eps
    self.f     = np.zeros(self.N, dtype=np.complex128)
    self.f[3]  = f0
    self.model = model

  def init(self):
    u0    = np.zeros(self.N, dtype=np.complex128)
    theta = 2.e0 * np.pi * np.random.randn(self.N)
    if self.f[0] == 0.e0:
      A = 1e-4
    else:
      A = self.f[0]
    u0 = A * self.k**(-1.e0/3.e0) * np.exp(1.j * theta)
    return u0

  @staticmethod
  @njit(cache=True, nogil=True, fastmath=True)
  def _GOY(params, u, du):
    N, nu, k, k2_nu, eps, f = params
    # boundary n = 0
    du[0] = 1.j * np.conj(
      k[0] * u[1] * u[2]
    ) - k2_nu[0] * u[0] + f[0]
    # boundary n = 1
    du[1] = 1.j * np.conj(
      - eps * k[0] * u[0] * u[2]
            + k[1] * u[2] * u[3]
    ) - k2_nu[1] * u[1] + f[1]
    # boundary n = 2 ~ N-2
    for n in range(2, N-2):
      du[n] = 1.j * np.conj(
       - eps * k[n-2] * u[n-2] * u[n-1]
       - eps * k[n-1] * u[n-1] * u[n+1]
             + k[n]   * u[n+1] * u[n+2]
      ) - k2_nu[n] * u[n] + f[n]
    # boundary n = N-1
    du[N-2] = 1.j * np.conj(
       - eps * k[N-4] * u[N-4] * u[N-3]
       - eps * k[N-3] * u[N-3] * u[N-1]
    ) - k2_nu[N-2] * u[N-2] + f[N-2]
    # boundary n = N
    du[N-1] = 1.j * np.conj(
      - eps * k[N-3] * u[N-3] * u[N-2]
    ) - k2_nu[N-1] * u[N-1] + f[N-1]
    return du

  @staticmethod
  @njit(cache=True, nogil=True, fastmath=True)
  def _Sabra(params, u, du):
    N, nu, k, k2_nu, eps, f = params
    uc = np.conj(u)
    # boundary n = 0
    du[0] = 1.j * (
      k[1] * uc[1] * u[2]
    ) - k2_nu[0] * u[0] + f[0]
    # boundary n = 1
    du[1] = 1.j * (
      -eps * k[1] * uc[0] * u[2]
           + k[2] * uc[2] * u[3]
    ) - k2_nu[1] * u[1] + f[1]
    # boundary n = 2 ~ N-2
    for n in range(2, N-2):
      du[n] = 1.j * (
      (1.e0 - eps) * k[n-1] *  u[n-2] * u[n-1]
            - eps  * k[n]   * uc[n-1] * u[n+1]
                   + k[n+1] * uc[n+1] * u[n+2]
      ) - k2_nu[n] * u[n] + f[n]
    # boundary n = N-1
    du[N-2] = 1j * (
      (1.e0 - eps) * k[N-3] *  u[N-4] * u[N-3]
            - eps  * k[N-2] * uc[N-3] * u[N-1]
    ) - k2_nu[N-2] * u[N-2] + f[N-2]
    # boundary n = N
    du[N-1] = 1j * (
      (1.e0 - eps) * k[N-2] * u[N-3] * u[N-2]
    ) - k2_nu[N-1] * u[N-1] + f[N-1]
    return du

  def RK44(self, Nt, Np, dt):
    params = (self.N, self.nu, self.k, self.k2_nu, self.eps, self.f)
    u0 = self.init()
    if self.model == "sabra":
      u  = self._RK44(self._Sabra, self.N, params, Nt, Np, dt, u0)
    elif self.model == "goy":
      u  = self._RK44(self._GOY, self.N, params, Nt, Np, dt, u0)
    else:
      print("Available models are sabra & goy")
    return u

