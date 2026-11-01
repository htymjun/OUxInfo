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

---

## Conditional Information Flow

Passing a conditioning series `Z` to `information_flow` or `information_flow_causal_map` replaces every mutual information by a conditional mutual information (Frenzel–Pompe KSG estimator):

$$T_{x\to y|z} = \frac{I(x_t; y_{t+\tau} \mid z_t) - I(x_t; y_t \mid z_t)}{\tau\,\Delta t}$$

where \(\Delta t\) is the sampling step; every rate in OUxInfo is divided by \(\tau\,\Delta t\).

Linear Gaussian chain \(x \to w \to y\) without a direct link:

$$x_{t+1} = 0.5\,x_t + \varepsilon_x, \quad w_{t+1} = 0.3\,w_t + c\,x_t + \varepsilon_w, \quad y_{t+1} = 0.3\,y_t + c\,w_t + \varepsilon_y$$

with \(\tau = 2\), \(N = 10^4\) and \(k = 5\). Lines are the closed-form values from the stationary covariance; markers are single-realization estimates. Conditioning on the mediator \(w\) removes the part of the flow that passes through \(w\).

![Conditional information flow validation](results/IF_conditional.png)

`tests/test_IF_conditional.py` also checks that

- the estimate agrees with an exhaustive-distance implementation of the same formula to \(10^{-10}\) for every code path (single pair, causal map with and without mask, `full=True`);
- conditioning on the target, \(z = y\), gives the transfer entropy exactly (\(10^{-12}\)), because \(I(x_t; y_t \mid y_t) = 0\); likewise \(z = (y, w)\) gives the conditional transfer entropy, and a constant \(z\) gives the unconditional information flow;
- the 16-trial mean at \(N = 10^4\) lies within 0.01 nats of the closed-form value for unconditional, conditional, transfer entropy and conditional transfer entropy cases of three VAR(1) systems.
