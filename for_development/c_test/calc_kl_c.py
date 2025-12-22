import numpy as np
import matplotlib.pyplot as plt
from shannon_entropy_cpp import KL_div

def test_kl_div():
  var_x = 1.e0
  Var_y = np.linspace(1e0, 10.e0, 50)
  cov   = 1.e0
  mean  = (0.e0, 0.e0)
  KLt   = np.zeros_like(Var_y)
  KL5   = np.zeros_like(Var_y)
  KL10  = np.zeros_like(Var_y)
  KL50  = np.zeros_like(Var_y)
  for i, var_y in enumerate(Var_y):
    rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
    Cov  = [[var_x, cov], \
            [cov, var_y]]
    x = np.random.multivariate_normal(mean, Cov, 10000)
    KLt[i] = np.log(np.sqrt(var_y/var_x)) + var_x / (2.e0 * var_y) - 0.5e0
    KL5[i] = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=5)
    KL10[i] = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
    KL50[i] = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=50)
  plt.figure(figsize=(6,6))
  plt.plot(Var_y, KLt, color='black')
  plt.plot(Var_y, KL5,  "o", color='blue')
  plt.plot(Var_y, KL10, "o", color='red')
  plt.plot(Var_y, KL50, "o", color='darkgreen')
  plt.xlabel("Var", fontsize=18, style='italic')
  plt.ylabel("KL div", fontsize=18)
  plt.savefig("kl_c_sigma.png")


test_kl_div()