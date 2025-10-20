import matplotlib.pyplot as plt
import numpy as np
from shannon_entropy_cpp import shannon_entropy

def test_sigma():
  sigma = np.linspace(1.e0, 5.e0, 50)
  H_t = np.zeros(len(sigma))
  H_n = np.zeros(len(sigma))
  for i, s in enumerate(sigma):
    x = np.random.normal(0.e0, s, 500)
    H_t[i] = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))
    H_n[i] = shannon_entropy(x.reshape(-1,1), k=5)
  plt.figure(figsize=(6,6))
  plt.plot(sigma, H_t, color='black')
  plt.plot(sigma, H_n, "o", color='blue')
  plt.xlabel(r'$\sigma$', fontsize=18, style='italic')
  plt.ylabel("H(X)", fontsize=18)
  plt.savefig("shan_c_sigma.png")

  
def test_k():
  K = np.arange(1,50)
  s = 1.e0
  x = np.random.normal(0.e0, s, 500)
  H_t = np.zeros(len(K))
  H_n = np.zeros(len(K))
  for i, k in enumerate(K):
    H_t[i] = 0.5e0 * (1.e0 + np.log(2.e0 * np.pi * s**2))
    H_n[i] = shannon_entropy(x.reshape(-1,1), k=k)
  plt.figure(figsize=(6,6))
  plt.plot(K, H_t, color='black')
  plt.plot(K, H_n, "o", color='blue')
  plt.xlabel("k", fontsize=18, style='italic')
  plt.ylabel("H(X)", fontsize=18)
  plt.savefig("shan_c_k.png")


test_sigma()
test_k()
