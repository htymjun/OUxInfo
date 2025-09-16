import numpy as np
from shannon import shannon_entropy

N = 500
s = 1.e0
x = np.random.normal(0.e0, s, N).astype(np.float32)
x = np.asfortranarray(x)
Ht = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))
Hn = shannon_entropy(x.reshape(-1,1), k=5)

print("Theoretical:", Ht, " Numerical:", Hn)

