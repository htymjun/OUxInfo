import numpy as np
from shannon_entropy_cpp import shannon_entropy, KL_div
import time
import matplotlib.pyplot as plt

'''
#シャノン
N = 500000
s = 1.e0
x = np.random.normal(0.e0, s, N)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))

t1 = time.time()
Hn = shannon_entropy(x.reshape(-1,1), k=3)
t2 = time.time()
print("Theoretical:", Ht, " Numerical:", Hn)
print("Elapsed time:", t2-t1)
'''

'''
#KL
N = 500000
var_x = 1.e0
var_y = 9.e0
cov   = 1.e0
mean  = (0.e0, 0.e0)
rho   = cov / (np.sqrt(var_x * var_y)) # corr coef
Cov   = [[var_x, cov], \
         [cov, var_y]]
x = np.random.multivariate_normal(mean, Cov, N)
Dt = np.log(np.sqrt(var_y/var_x)) + var_x / (2.e0 * var_y) - 0.5e0
t1 = time.time()
Dn = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=3)
t2 = time.time()
print("Theoretical:", Dt, " Numerical:", Dn)
print("Elapsed time:", t2-t1)
'''

'''
#KL いろんな分散
N = 500000
mean = (0.0,0.0)

var_list = [1.0,2.0,5.0,9.0,16.0]

for var_x in var_list:
    for var_y in var_list:
        cov = [[var_x,0],
               [0,var_y]]
        x = np.random.multivariate_normal(mean,cov,N)
        Dt = np.log(np.sqrt(var_y/var_x)) + var_x / (2.e0 * var_y) - 0.5e0
        Dn = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=3)
        print(f"{var_x:10.2f} {var_y:10.2f} {Dt:15.6f} {Dn:15.6f}")
'''

def test_kl_div():
  t1 = time.time()
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
    x = np.random.multivariate_normal(mean, Cov, 20000)
    KLt[i] = np.log(np.sqrt(var_y/var_x)) + var_x / (2.e0 * var_y) - 0.5e0
    KL5[i] = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=5)
    KL10[i] = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=10)
    KL50[i] = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=50)
  t2 = time.time()
  print("Elapsed time:", t2-t1)
  plt.figure(figsize=(6,6))
  plt.rcParams.update({'font.size': 16})
  plt.plot(Var_y, KLt, color='black')
  plt.plot(Var_y, KL5,  "o", color='blue')
  plt.plot(Var_y, KL10, "o", color='red')
  plt.plot(Var_y, KL50, "o", color='darkgreen')
  #plt.xlabel("Var", fontsize=18, style='italic')
  #plt.ylabel("KL div", fontsize=18)
  plt.savefig("figure_me.png")


test_kl_div()
