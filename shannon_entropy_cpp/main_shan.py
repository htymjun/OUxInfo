import numpy as np
from shannon_entropy_cpp import shannon_entropy
from scipy.special import psi as psi_c, gamma as gamma_c
from scipy.spatial import KDTree
import time
import cupy as cp
from cupyx.scipy.special import psi as psi_g, gamma as gamma_g
#import faiss
from cupy.cuda.nvtx import RangePush, RangePop
from cuml.neighbors import NearestNeighbors


def count_points1(tree_X, tree_XY, X, XY, k, Thei=1):
  N   = X.shape[0]
  eps = np.zeros(N)
  nx  = np.zeros(N)
  for i in range(N):
    distance, _ = tree_XY.query(XY[i], k=k+1, p=np.inf)
    eps[i] = 2.e0 * distance[-1]
    idx = tree_X.query_ball_point(X[i], 0.5e0 * eps[i], p=np.inf)
    idx = [j for j in idx if abs(j - i) >= Thei]
    nx[i] = max(len(idx), 1.e0) # nx must be greater than 0 because psi(0) = -inf
  return nx, eps


def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx


def shannon_entropy_py(X, k=3, Thei=1, Z=None):
  # Kozachenko Leonenko
  N = X.shape[0]
  X, dx = reshape_matrix(X)
  eps = np.zeros(N)
  tree_X = KDTree(X)
  # volume of unit ball in d*n
  Cdx = np.pi**(0.5e0*dx) / gamma_c(1.e0 + 0.5e0 * dx) / 2.e0**dx
  if Z is not None:
    Z, _ = reshape_matrix(Z)
    nx, eps = count_points1(tree_X, KDTree(Z), X, Z, k=k, Thei=Thei)
    H = np.mean(-psi_c(nx + 1.e0) + dx * np.log(eps)) + psi_c(N) + np.log(Cdx)
  else:
    for i in range(N):
      distance, _ = tree_X.query(X[i], k=k+1, p=np.inf)
      eps[i] = 2.e0 * distance[-1]
    H = -psi_c(k) + psi_c(N) + np.log(Cdx) + dx * np.mean(np.log(eps))
  return H

'''
# nvtx
def shannon_entropy_rp(X, k=3, Thei=1, Z=None):
  # Kozachenko Leonenko
  X = cp.asarray(X, dtype=cp.float32)
  X_gpu = X.reshape(-1, 1)
  N = X.shape[0]
  dx = X_gpu.shape[1]

  # 最近傍探索 (L∞距離)
  with nvtx.annotate("NearestNeighbors", color="blue"):
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='chebyshev')

  with nvtx.annotate("nn.kneighbors", color="red"):
    nn.fit(X_gpu)
    dist, _ = nn.kneighbors(X_gpu)

    # k+1番目の距離（自身を除外）
  with nvtx.annotate("eps and Cdx", color="red"):
    eps = 2.0 * dist[:, -1]
    eps = cp.where(eps <= 0, 1e-12, eps)
    Cdx = cp.pi**(0.5e0*dx) / gamma_g(1.e0 + 0.5e0 * dx) / 2.e0**dx 

  with nvtx.annotate("digamma", color="green"):
    H = -psi_g(k) + psi_g(N) + cp.log(Cdx) + dx * cp.mean(cp.log(eps))
  return cp.asnumpy(H) # Convert back to numpy for return value
'''

def shannon_entropy_rp(X, k=3, Thei=1, Z=None):
  # Kozachenko Leonenko
  X = cp.asarray(X, dtype=cp.float32)
  X_gpu = X.reshape(-1, 1)
  N = X.shape[0]
  dx = X_gpu.shape[1]

  # 最近傍探索 (L∞距離)
  RangePush("NearestNeighbors")
  nn = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='chebyshev')
  RangePop()

  RangePush("nn.kneighbors")
  nn.fit(X_gpu)
  dist, _ = nn.kneighbors(X_gpu)
  RangePop()
  
  RangePush("eps and Cdx")
  # k+1番目の距離（自身を除外）
  eps = 2.0 * dist[:, -1]
  eps = cp.where(eps <= 0, 1e-12, eps)
  Cdx = cp.pi**(0.5e0*dx) / gamma_g(1.e0 + 0.5e0 * dx) / 2.e0**dx 
  RangePop()

  H = -psi_g(k) + psi_g(N) + cp.log(Cdx) + dx * cp.mean(cp.log(eps))
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

def shannon_entropy_rp_cupy_batch(X, k=3,batch_size=2000):
    """
    CuPyを用いたKozachenko–Leonenko法によるシャノンエントロピー推定
    （最近傍探索をscikit-learnではなくcp.cdistで自作）
    """
    # --- GPU配列に変換 ---
    X = cp.asarray(X, dtype=cp.float32)
    N, d = X.shape if X.ndim > 1 else (X.size, 1)
    X = X.reshape(N, d)

    kth_dist = cp.full(N, cp.inf, dtype=cp.float32)

    for i in range(0, N, batch_size):
        X_batch = X[i:i+batch_size]
        diff = cp.abs(X_batch[:, None, :] - X[None, :, :])
        dist = cp.max(diff, axis=2)
        cp.fill_diagonal(dist[:min(batch_size, N - i), i:i+min(batch_size, N - i)], cp.inf)
        kth_local = cp.partition(dist, k-1, axis=1)[:, k-1]
        kth_dist[i:i+batch_size] = kth_local
        del diff, dist, X_batch
        cp._default_memory_pool.free_all_blocks()

    # --- eps: ボール半径（L∞距離なので2倍する） ---
    eps = 2.0 * kth_dist
    eps = cp.where(eps <= 0, 1e-12, eps)

    # --- 定数項（単位球体の体積係数）---
    C_d = (np.pi ** (0.5 * d)) / gamma(1.0 + 0.5 * d) / (2.0 ** d)

    # --- Kozachenko–Leonenko 推定式 ---
    H = -psi(k) + psi(N) + np.log(C_d) + d * cp.mean(cp.log(eps))

    # --- 結果をCPU側へ戻す ---
    return float(cp.asnumpy(H))


#versionが合わずうまくいかなかった
def shannon_entropy_rp_cupy_faiss(X, k=3):
    
    #CuPyを用いたKozachenko–Leonenko法によるシャノンエントロピー推定
    #（最近傍探索をscikit-learnではなくcp.cdistで自作）

    # --- GPU配列に変換 ---
    X = cp.asarray(X, dtype=cp.float32)
    N, d = X.shape if X.ndim > 1 else (X.size, 1)
    X = X.reshape(N, d)

    res = faiss.StandardGpuResources()

    cpu_index = faiss.IndexFlatL2(d)  # ユークリッド距離
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # --- データをGPUインデックスに登録（NumPy経由でfloat32） ---
    X_np = cp.asnumpy(X)  # CuPy→NumPy（FAISS GPU版はこれでOK）
    gpu_index.add(X_np)

    # --- k最近傍探索 ---
    D, _ = gpu_index.search(X_np, k + 1)  # 自分自身を含む
    D = D[:, 1:]  # 自分自身を除外
    
    eps = np.sqrt(D[:, -1])  # L2距
    eps = np.where(eps <= 0, 1e-12, eps)

    # --- 定数項（単位球体の体積係数）---
    C_d = (np.pi ** (0.5 * d)) / gamma(1.0 + 0.5 * d)

    # --- Kozachenko–Leonenko 推定式 ---
    H = -psi(k) + psi(N) + np.log(C_d) + d * np.mean(np.log(eps))

    # --- 結果をCPU側へ戻す ---
    return float(H)


N = 20000000
s = 1.e0
x = np.random.normal(0.e0, s, N)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))

t1 = time.time()
Hn1 = shannon_entropy_py(x, k=3)
t2 = time.time()
Hn2 = shannon_entropy_rp(x, k=3)
t3 = time.time()
#Hn3 = shannon_entropy_rp_cupy(x, k=3)
#t4 = time.time()
#Hn4 = shannon_entropy_rp_cupy_batch(x, k=3)
#t5 = time.time()
#Hn5 = shannon_entropy_rp_cupy_faiss(x, k=3)
t6 = time.time()
Hc = shannon_entropy(x.reshape(-1,1), k=3)
t7 = time.time()

print("Number     :", N  )
print("Theoretical:", Ht )
print("python     :", Hn1)
print("python rp  :", Hn2)
#print("python rpc :", Hn3)
#print("python rpcb:", Hn4)
#print("python rpcf:", Hn5)
print("c++        :", Hc )
print("time python            :", t2-t1)
print("time python rapids     :", t3-t2)
#print("time python rapids cp  :", t4-t3)
#print("time python rapids cp b:", t5-t4)
#print("time python rapids cp f:", t6-t5)
print("time c++               :", t7-t6)
