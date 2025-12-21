import numpy as np
import cupy as cp
from scipy.special import psi, gamma
from scipy.spatial import KDTree
from cuml.neighbors import NearestNeighbors
import time

N = 1000000
s = 1.e0
x = np.random.normal(0.e0, s, N)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))

def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx


def shannon_entropy(X, k=3, Thei=1, Z=None):
  # Kozachenko Leonenko
  N = X.shape[0]
  X, dx = reshape_matrix(X)
  eps = np.zeros(N)
  tree_X = KDTree(X)
  # volume of unit ball in d*n
  Cdx = np.pi**(0.5e0*dx) / gamma(1.e0 + 0.5e0 * dx) / 2.e0**dx
  for i in range(N):
    distance, _ = tree_X.query(X[i], k=k+1, p=np.inf)
    eps[i] = 2.e0 * distance[-1]
  H = -psi(k) + psi(N) + np.log(Cdx) + dx * np.mean(np.log(eps))
  return H

#意外と遅い．次元上げたら速くなるかも
def shannon_entropy_rp(X, k=3, Thei=1, Z=None):
  # Kozachenko Leonenko
  X = cp.asarray(X, dtype=cp.float32)
  X_gpu = X.reshape(-1, 1)
  N = X.shape[0]
  dx = X_gpu.shape[1]

    # 最近傍探索 (L∞距離)
  nn = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='chebyshev')
  nn.fit(X_gpu)
  dist, _ = nn.kneighbors(X_gpu)

    # k+1番目の距離（自身を除外）
  eps = 2.0 * dist[:, -1]
  eps = cp.where(eps <= 0, 1e-12, eps)
  Cdx = np.pi**(0.5e0*dx) / gamma(1.e0 + 0.5e0 * dx) / 2.e0**dx

  H = -psi(k) + psi(N) + np.log(Cdx) + dx * cp.mean(cp.log(eps)) # Use cp.mean and cp.log for CuPy array
  return cp.asnumpy(H) # Convert back to numpy for return value

#メモリ食いすぎで計算できてない
def shannon_entropy_rp_cupy(X, k=3):
    """
    CuPyを用いたKozachenko–Leonenko法によるシャノンエントロピー推定
    （最近傍探索をscikit-learnではなくcp.cdistで自作）
    """
    # --- GPU配列に変換 ---
    X = cp.asarray(X, dtype=cp.float32)
    N, d = X.shape if X.ndim > 1 else (X.size, 1)
    X = X.reshape(N, d)

    # --- L∞距離（Chebyshev距離）を計算 ---
    # cdist の metric='chebyshev' は CuPy では未サポートなので自前で定義
    # |x_i - x_j| の絶対値の最大値を取る
    diff = cp.abs(X[:, None, :] - X[None, :, :])   # shape (N, N, d)
    dist = cp.max(diff, axis=2)                    # shape (N, N)

    # --- 自分自身との距離を無限大にして除外 ---
    cp.fill_diagonal(dist, cp.inf)

    # --- 各点におけるk番目の最近傍距離を取得 ---
    # cp.partition は部分ソート（全ソートより高速）
    kth_dist = cp.partition(dist, k-1, axis=1)[:, k-1]

    # --- eps: ボール半径（L∞距離なので2倍する） ---
    eps = 2.0 * kth_dist
    eps = cp.where(eps <= 0, 1e-12, eps)

    # --- 定数項（単位球体の体積係数）---
    C_d = (np.pi ** (0.5 * d)) / gamma(1.0 + 0.5 * d) / (2.0 ** d)

    # --- Kozachenko–Leonenko 推定式 ---
    H = -psi(k) + psi(N) + np.log(C_d) + d * cp.mean(cp.log(eps))

    # --- 結果をCPU側へ戻す ---
    return float(cp.asnumpy(H))



#t1 = time.time()
#H1 = shannon_entropy(X.astype(np.float64), k=k)
#t2 = time.time()
#print("C++:",H1)
#print("C-T:",t2-t1)
k=3 # Define k
print("The:",Ht)
t3 = time.time()
H2 = shannon_entropy(x, k=k)
t4 = time.time()
print("py :",H2)
print("pyt:",t4-t3)
t5 = time.time()
H3 = shannon_entropy_rp(x, k=k)
t6 = time.time()
print("pyr:",H3)
print("prt:",t6-t5)
t7 = time.time()
H4 = shannon_entropy_rp_cupy(x, k=k)
t8 = time.time()
print("pyr:",H4)
print("prt:",t8-t7)