import numpy as np
from ouxinfo import shannon_entropy


N = 10000
s = 1.e0
x = np.random.normal(0.e0, s, N)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))
Hn = shannon_entropy(x.reshape(-1,1), k=5)

print("Theoretical:", Ht)
print("Numerical  :", Hn)

