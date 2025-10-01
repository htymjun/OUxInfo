import numpy as np
from shannon_entropy_cpp import shannon_entropy, KL_div
import time


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
Dn = KL_div(x.reshape(-1,1), x.reshape(-1,1), k=3)
t2 = time.time()
print("Theoretical:", Dt, " Numerical:", Dn)
print("Elapsed time:", t2-t1)
'''
