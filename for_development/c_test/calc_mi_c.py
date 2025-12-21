import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from shannon_entropy_cpp import mutual_info

#修正必要　拡張
def test_mi():
  var_x = 1.0
  Var_y = np.linspace(1.01, 10.e0, 50)
  cov   = 1.e0
  mean  = (0.e0, 0.e0)
  MIt   = np.zeros_like(Var_y)
  MI5   = np.zeros_like(Var_y)
  MI10  = np.zeros_like(Var_y)
  MI50  = np.zeros_like(Var_y)
  for i, var_y in enumerate(Var_y):
    rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
    Cov  = [[var_x, cov], \
              [cov, var_y]]
    x = np.random.multivariate_normal(mean, Cov, 10000)
    MIt[i] = -0.5e0 * np.log(1.e0 - rho**2)
    MI5[i] = mutual_info(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=5)
    MI10[i] = mutual_info(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
    MI50[i] = mutual_info(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=50)
  plt.figure(figsize=(6,6))
  plt.plot(Var_y, MIt, color='black')
  plt.plot(Var_y, MI5,  "o", color='blue')
  plt.plot(Var_y, MI10, "o", color='red')
  plt.plot(Var_y, MI50, "o", color='darkgreen')
  plt.xlabel("Var", fontsize=18, style='italic')
  plt.ylabel("mutual info", fontsize=18)
  plt.savefig("mi_c_sigma.png")
test_mi()