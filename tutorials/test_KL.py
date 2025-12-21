import numpy as np
from ouxinfo import KL_div


N = 10000
s = 1.e0
x = np.random.normal(0.e0, s, N)
var_x = 1.e0
var_y = 9.e0
cov   = 1.e0
mean  = (0.e0, 0.e0)
rho   = cov / (np.sqrt(var_x * var_y)) # corr coef
Cov   = [[var_x, cov], \
         [cov, var_y]]
x = np.random.multivariate_normal(mean, Cov, N)

Dt = np.log(np.sqrt(var_y/var_x)) + var_x / (2.e0 * var_y) - 0.5e0
Dn = KL_div(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=5)
print("Theoretical:", Dt)
print("Numerical  :", Dn)

