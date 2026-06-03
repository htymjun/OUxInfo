# Validation

Each estimator is validated against analytical results on synthetic Gaussian data. Black lines show theoretical values; colored markers show KNN estimates (N = 5,000–10,000 samples, k = 5 unless varied).

---

## Shannon Entropy

**Left:** Entropy of a Gaussian \(X \sim \mathcal{N}(0, \sigma^2)\) vs standard deviation \(\sigma\).
The theoretical value is \(H = \tfrac{1}{2}(1 + \ln 2\pi\sigma^2)\).

**Right:** Estimated entropy vs KNN parameter \(k\) at fixed \(\sigma = 1\).

![Shannon entropy validation](results/H.png)

---

## Mutual Information

**Left:** Mutual information of a bivariate Gaussian with correlation \(\rho\) vs \(\rho\).
The theoretical value is \(I = -\tfrac{1}{2}\ln(1 - \rho^2)\).

**Right:** Estimated MI vs \(k\) at fixed \(\rho = 0.75\).

![Mutual information validation](results/MI.png)

---

## KL Divergence

**Left:** KL divergence \(D_{\rm KL}(\mathcal{N}(0,1) \| \mathcal{N}(0,\sigma_y^2))\) vs \(\sigma_y\).
The analytical value is \(D_{\rm KL} = \ln\sigma_y + \tfrac{1}{2\sigma_y^2} - \tfrac{1}{2}\).

**Right:** Estimated \(D_{\rm KL}\) vs \(k\) at fixed \(\sigma_y = 3\).

![KL divergence validation](results/KL.png)

---

## Transfer Entropy

Coupled linear Gaussian AR system where \(x\) drives \(y\):

$$x_{t+1} = a\,x_t + \varepsilon_x, \quad y_{t+1} = b\,y_t + c_{xy}\,x_t + \varepsilon_y$$

with \(a = b = 0.5\) and \(\varepsilon \sim \mathcal{N}(0, 1)\).
The causal direction is \(x \to y\); \(c_{xy}\) controls coupling strength.

Blue circles show \(TE_{x\to y}\) (should exceed the black analytical line); red triangles show \(TE_{y\to x}\) (should remain near zero). The estimator correctly identifies the causal direction and is monotone in coupling strength.

![Transfer entropy validation](results/TE.png)

---

## Information Flow

Same coupled AR system as above.

Blue circles show \(T_{x\to y}\) (Horowitz–Esposito information flow in the causal direction); red triangles show \(T_{y\to x}\) (should remain near zero). The estimator recovers the correct causal asymmetry across all coupling strengths.

![Information flow validation](results/IF.png)
