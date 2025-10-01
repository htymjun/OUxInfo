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

