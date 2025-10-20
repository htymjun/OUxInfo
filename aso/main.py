import numpy as np
from shannon import shannon_entropy
from python_test.calc_shannon_entropy import shannon_entropy as shannon_entropy_py
import time


N = 500000
s = 1.e0
x = np.random.normal(0.e0, s, N).astype(np.float32)
x = np.asfortranarray(x)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))

t1 = time.time()
Hf = shannon_entropy(x.reshape(-1,1), k=5)
t2 = time.time()
Hp = shannon_entropy_py(x, k=5)
t3 = time.time()
print("Theoretical:", Ht, " Numerical:", Hf, Hp)
print("Elapsed time:", t2-t1, t3-t2)

