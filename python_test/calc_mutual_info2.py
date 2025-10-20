import numpy as np
from scipy.special import psi, gamma
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.special import digamma


def reshape_matrix(X):
  if X.ndim == 1:
    dx = 1.e0
    X  = X.reshape((-1,1))
  else:
    dx = X.shape[1]
  return X, dx

def mutual_info1(X, Y, k=5, Thei=10):
    X, _ = reshape_matrix(X)
    Y, _ = reshape_matrix(Y)
    N = X.shape[0]
    
    nX = np.zeros(N)
    nY = np.zeros(N)
    
    tree_XY = KDTree(np.hstack((X, Y)))
    dist, _ = tree_XY.query(np.hstack((X, Y)), k=k+1, p=np.inf, workers=-1)
    half_epsilon_XYkNN = dist[:, -1]

    for i in range(N):
        idx = np.arange(N) != i
        # 論文の定義通り、距離が "厳密に小さい" 点をカウント
        nX[i] = np.sum(np.abs(X[idx] - X[i]) < half_epsilon_XYkNN[i])
        nY[i] = np.sum(np.abs(Y[idx] - Y[i]) < half_epsilon_XYkNN[i])

    I = psi(k) - np.mean(psi(nX + 1)) - np.mean(psi(nY + 1)) + psi(N)
    
    return I

def mutual_info2(X, Y, k=5, Thei=10):
  X, _ = reshape_matrix(X)
  Y, _ = reshape_matrix(Y)
  N = X.shape[0]
  nX, nY, nN = np.zeros(N), np.zeros(N), np.zeros(N)
  for i in range(N):
    idx = np.ones(N, dtype=bool)
    idx[max(0, i - Thei):min(i + Thei + 1, N)] = False
    tree_XY = KDTree(np.hstack((X[idx], Y[idx])))
    dist, _ = tree_XY.query(np.hstack((X[i], Y[i])), k=k, p=np.inf, workers=-1)
    half_epsilon_XYkNN = dist[-1]
    nX[i] = np.sum(cdist(X[idx], X[i].reshape(1,-1), metric='chebyshev') < half_epsilon_XYkNN)
    nY[i] = np.sum(cdist(Y[idx], Y[i].reshape(1,-1), metric='chebyshev') < half_epsilon_XYkNN)
    nN[i] = np.sum(idx)
  valid_idx = (nX > 0) & (nY > 0)
  I = psi(k) - np.mean(psi(nX[valid_idx] + 1)) - np.mean(psi(nY[valid_idx] + 1)) + np.mean(psi(nN[valid_idx] + 1))
  return I 

def mutual_info3(x, y, k=5):
    
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



def test_mutual_info1():
  # variance
  var_x = 9.e0
  var_y = 25.e0
  covariance = np.linspace(0.e0, 10.e0, 50)
  mean = (0.e0, 0.e0)
  It   = np.zeros(len(covariance))
  In   = np.zeros(len(covariance))
  t1 = time.time()
  for i, cov in enumerate(covariance):
    rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
    Cov  = [[var_x, cov], \
            [cov, var_y]]
    x = np.random.multivariate_normal(mean, Cov, 2000)
    It[i] = -0.5e0 * np.log(1.e0 - rho**2)
    In[i] = mutual_info1(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
  t2 = time.time()
  print("time:",t2-t1)
  plt.figure(figsize=(6,6))
  plt.plot(covariance, It, color='black')
  plt.plot(covariance, In, "o", color='blue')
  plt.xlabel(r'$\sigma_{xy}$', fontsize=18, style='italic')
  plt.ylabel("I(X;Y)", fontsize=18)
  plt.savefig("mutual_info2_1.png")

def test_mutual_info2():
  # variance
  var_x = 9.e0
  var_y = 25.e0
  covariance = np.linspace(0.e0, 10.e0, 50)
  mean = (0.e0, 0.e0)
  It   = np.zeros(len(covariance))
  In   = np.zeros(len(covariance))
  t3 = time.time()
  for i, cov in enumerate(covariance):
    rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
    Cov  = [[var_x, cov], \
            [cov, var_y]]
    x = np.random.multivariate_normal(mean, Cov, 2000)
    It[i] = -0.5e0 * np.log(1.e0 - rho**2)
    In[i] = mutual_info2(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
  t4 = time.time()
  print("time:",t4-t3)
  plt.figure(figsize=(6,6))
  plt.plot(covariance, It, color='black')
  plt.plot(covariance, In, "o", color='blue')
  plt.xlabel(r'$\sigma_{xy}$', fontsize=18, style='italic')
  plt.ylabel("I(X;Y)", fontsize=18)
  plt.savefig("mutual_info2_2.png")

def test_mutual_info3():
  # variance
  var_x = 9.e0
  var_y = 25.e0
  covariance = np.linspace(0.e0, 10.e0, 50)
  mean = (0.e0, 0.e0)
  It   = np.zeros(len(covariance))
  In   = np.zeros(len(covariance))
  t5 = time.time()
  for i, cov in enumerate(covariance):
    rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
    Cov  = [[var_x, cov], \
            [cov, var_y]]
    x = np.random.multivariate_normal(mean, Cov, 2000)
    It[i] = -0.5e0 * np.log(1.e0 - rho**2)
    In[i] = mutual_info2(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
  t6 = time.time()
  print("time:",t6-t5)
  plt.figure(figsize=(6,6))
  plt.plot(covariance, It, color='black')
  plt.plot(covariance, In, "o", color='blue')
  plt.xlabel(r'$\sigma_{xy}$', fontsize=18, style='italic')
  plt.ylabel("I(X;Y)", fontsize=18)
  plt.savefig("mutual_info2_3.png")

test_mutual_info1()
test_mutual_info2()
test_mutual_info3()
