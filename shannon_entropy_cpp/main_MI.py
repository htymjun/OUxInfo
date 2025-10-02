import numpy as np
from scipy.special import psi, gamma
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shannon_entropy_cpp import mutual_info
from shannon_entropy_cpp import mutual_info_noeps
import time
from scipy.special import digamma


def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx

def mutual_info_py(X, Y, k=5, Thei=10):
  X, _ = reshape_matrix(X)
  Y, _ = reshape_matrix(Y)
  N = X.shape[0]
  nX, nY, nN, epsC = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
  eps_all = 0
  for i in range(N):
    idx = np.ones(N, dtype=bool)
    idx[max(0, i - Thei):min(i + Thei + 1, N)] = False
    tree_XY = KDTree(np.hstack((X[idx], Y[idx])))
    dist, _ = tree_XY.query(np.hstack((X[i], Y[i])), k=k, p=np.inf, workers=-1)
    half_epsilon_XYkNN = dist[-1]
    eps_all += half_epsilon_XYkNN
    nX[i] = np.sum(cdist(X[idx], X[i].reshape(1,-1), metric='chebyshev') < half_epsilon_XYkNN)
    nY[i] = np.sum(cdist(Y[idx], Y[i].reshape(1,-1), metric='chebyshev') < half_epsilon_XYkNN)
    nN[i] = np.sum(idx)
    epsC[i] = half_epsilon_XYkNN
  print("eps_mean_py1:",eps_all/N)
  valid_idx = (nX > 0) & (nY > 0)
  I = psi(k) - np.mean(psi(nX[valid_idx] + 1)) - np.mean(psi(nY[valid_idx] + 1)) + np.mean(psi(nN[valid_idx] + 1))
  return I, epsC

def mutual_info_py2(x, y, k=5):
    
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y

    n = x.shape[0]

    # 結合空間 (X,Y)
    xy = np.hstack((x, y))

    # kd-tree
    tree_xy = KDTree(xy)
    tree_x  = KDTree(x)
    tree_y  = KDTree(y)

    # (k+1)-th 最近傍距離を取得 (自身を含むので+1)
    d_xy, _ = tree_xy.query(xy, k+1, p=np.inf, workers=-1)  # Chebyshev 距離 (max norm)
    d = d_xy[:,-1]
    d_mean = np.mean(d)
    print("eps_mean_py2:", d_mean)

    # 近傍数をカウント
        # 1. 結果を保存するための空のリストを用意する
    nx_list = []

    # 2. データ点xを一つずつ取り出しながらループする
    for i, p in enumerate(x):
        
        # 3. 現在の点pを中心として、半径d[i]の範囲内にある点のインデックスを取得する
        #    p=np.inf は最大ノルム（正方形の範囲）を指定
        points_in_ball = tree_x.query_ball_point(p, d[i] - 1e-15, p=np.inf)
        
        # 4. 見つかった点の数を数える（len()はリストの要素数を返す）
        #    この時点では、中心点p自身も含まれている
        count_including_self = len(points_in_ball)
        
        # 5. 自分自身（1個）を引いて、純粋な近傍点の数を計算する
        count_excluding_self = count_including_self - 1
        
        # 6. 計算した近傍点の数をリストに追加する
        nx_list.append(count_excluding_self)

    # 7. 最後に、PythonのリストをNumPy配列に変換する
    nx = np.array(nx_list)

    # 1. 結果を保存するための空のリストを用意する
    ny_list = []

    # 2. データ点yを一つずつ取り出しながらループする
    for i, p in enumerate(y):
        
        # 3. 現在の点pを中心として、半径d[i]の範囲内にある点のインデックスを取得する
        #    このd[i]は、先ほどと同様に同時空間(X,Y)で計算された距離
        points_in_ball = tree_y.query_ball_point(p, d[i] - 1e-15, p=np.inf)
        
        # 4. 見つかった点の数を数える（中心点p自身も含む）
        count_including_self = len(points_in_ball)
        
        # 5. 自分自身（1個）を引いて、純粋な近傍点の数を計算する
        count_excluding_self = count_including_self - 1
        
        # 6. 計算した近傍点の数をリストに追加する
        ny_list.append(count_excluding_self)

    # 7. 最後に、PythonのリストをNumPy配列に変換する
    ny = np.array(ny_list)
    
    #nx = np.array([tree_x.query_ball_point(p, d[i]-1e-15, p=np.inf).__len__() - 1 for i, p in enumerate(x)])
    #ny = np.array([tree_y.query_ball_point(p, d[i]-1e-15, p=np.inf).__len__() - 1 for i, p in enumerate(y)])

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx+1) + digamma(ny+1))
    return mi

var_x = 9.e0
var_y = 25.e0
cov = 10
It = 0.0
In = 0.0
In2 = 0.0
Ic = 0.0
mean = (0.e0, 0.e0)
rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
Cov  = [[var_x, cov], \
            [cov, var_y]]
x = np.random.multivariate_normal(mean, Cov, 10000)
It = -0.5e0 * np.log(1.e0 - rho**2)

X, dx = reshape_matrix(x[:,0])
Y, dy = reshape_matrix(x[:,1])
N = X.shape[0]

print("Theoretical  :", It)
t1 = time.time()
In, eps = mutual_info_py(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
t2 = time.time()
print("python1      :", In)
In2 = mutual_info_py2(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
t3 = time.time()
print("python2      :", In2)
Ic1 = mutual_info(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
t4 = time.time()
Ic2 = mutual_info_noeps(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), eps = eps,k=10)  
t5 = time.time()
print("C++          :", Ic1)
print("C++ eps by py:", Ic2)
print("Elapsed time py1           :", t2-t1)
print("Elapsed time py2           :", t3-t2)
print("Elapsed time C++           :", t4-t3)
print("Elapsed time C++ eps by py :", t5-t4)