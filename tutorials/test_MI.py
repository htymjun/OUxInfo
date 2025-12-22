import numpy as np
from ouxinfo import mutual_info


var_x = 9.e0
var_y = 25.e0
cov  = 10
mean = (0.e0, 0.e0)
rho  = cov / (np.sqrt(var_x * var_y)) # corr coef
Cov  = [[var_x, cov], \
        [cov, var_y]]
x = np.random.multivariate_normal(mean, Cov, 10000)
It = -0.5e0 * np.log(1.e0 - rho**2)

In = mutual_info(x[:,0].reshape(-1,1), x[:,1].reshape(-1,1), k=5)
print("Theoretical:", It)
print("Numerical  :", In)

