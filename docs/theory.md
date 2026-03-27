# Theory

This document explains the theoretical background of Shannon entropy estimation.

## Shannon Entropy (KSG Estimator)
Shannon entropy quantifies the uncertainty in a random variable. For continuous data, the Kozachenko-Leonenko (k-NN) estimator (base e) is used:

$$
H(X) = \psi(N) - \psi(k) + \log c_d + \frac{d}{N} \sum_{i=1}^N \log \epsilon(i)
$$
where $\psi$ is the digamma function, $N$ is the number of samples, $k$ is the number of neighbors, $c_d$ is the volume of the $d$-dimensional unit ball, and $\epsilon(i)$ is twice the distance to the $k$-th neighbor of $x_i$.

See: Kozachenko & Leonenko (1987), Kraskov et al. (2004)

## KL Divergence (KSG Estimator)
KL divergence measures the difference between two distributions. The k-NN estimator (Pérez-Cruz, 2008) is:

$$
D_{KL}(P \| Q) = \psi(k) - \frac{1}{N} \sum_{i=1}^N \psi(n_q(i)) + \log\frac{M}{N-1}
$$
where $n_q(i)$ is the number of $Q$-samples within the $k$-th neighbor distance of $x_i$ in $P$, $N$ and $M$ are sample sizes, and $\psi$ is the digamma function. All logs are natural (base e).

See: Pérez-Cruz (2008), Kraskov et al. (2004)

## Mutual Information (KSG Estimator)
Mutual information quantifies the shared information between X and Y. The Kraskov-Stögbauer-Grassberger (KSG) estimator (base e) is:

$$
I(X;Y) = \psi(k) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle + \psi(N)
$$
where $n_x$ and $n_y$ are the number of neighbors within the $k$-th distance in $X$ and $Y$, $N$ is the sample size, and $\psi$ is the digamma function.

See: Kraskov et al. (2004)

## Conditional Mutual Information (KSG Estimator)
Conditional mutual information quantifies the information shared between X and Y given Z. The KSG estimator (base e) is:

$$
I(X;Y|Z) = \psi(k) - \langle \psi(n_{xz} + 1) + \psi(n_{yz} + 1) - \psi(n_z + 1) \rangle
$$
where $n_{xz}$, $n_{yz}$, and $n_z$ are the number of neighbors within the $k$-th distance in the joint spaces, and $\psi$ is the digamma function.

See: Kraskov et al. (2004)

## Transfer Entropy (KSG Estimator)
Transfer entropy quantifies the directed information transfer from Y to X. It is estimated using the k-NN (KSG) method (base e):

$$
TE_{Y \to X} = I(X_{t+1}; Y_t^{(l)} | X_t^{(k)})
$$
where $I$ is conditional mutual information estimated via k-NN. See Schreiber (2000), Kraskov et al. (2004).

## Backward Transfer Entropy
Backward transfer entropy is computed on time-reversed series to assess reverse information flow. See Ito et al. (2011) for details.

## Information Flow
Information flow measures the net information transferred between components in a system. The estimator is based on the difference of mutual information at different lags. See Horowitz & Esposito (2014).

$$
\frac{d}{dt} I(X_t;Y_t) = \dot{I}_X(t) + \dot{I}_Y(t) + Leak
$$

## Causal Map Transfer Entropy & Information Flow
Causal maps visualize the structure of information transfer and flow in multivariate systems.

## TEIFL
TEIFL combines transfer entropy, information flow, and leak to provide a comprehensive view of dynamical dependencies.

According to Matsumoto & Sagawa (2018, Phys. Rev. E), there is an analytical relation among these quantities:

$$
sTE > mTE > \dot{I}
$$

where sTE is the single-time step transfer entropy, mTE is the multi-time step transfer entropy, and IF is the information flow. 

## k-NN (KSG) Estimator
All estimators use the k-nearest neighbor (k-NN) approach, specifically the Kraskov-Stögbauer-Grassberger (KSG) estimator for entropy, mutual information, and conditional mutual information. This method is non-parametric and uses the digamma function for bias correction. See Kraskov et al. (2004).

## Applications
- Causal inference
- Complex systems analysis

## References
- Shannon, C. E. (1948). "A Mathematical Theory of Communication." Bell System Technical Journal.
- Schreiber, T. (2000). "Measuring information transfer." Physical Review Letters.
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). "Estimating mutual information." Physical Review E.
- Pérez-Cruz, F. (2008). "Kullback-Leibler divergence estimation of continuous distributions." IEEE International Symposium on Information Theory.
- Ito, S., et al. (2011). "Extending transfer entropy improves identification of effective connectivity in a spiking cortical network model." PLoS One.
- Horowitz, J. M., & Esposito, M. (2014). "Thermodynamics with continuous information flow." Physical Review X.
- Matsumoto, T., Sagawa, T. (2018). "Role of sufficient statistics in stochastic thermodynamics and its implication to sensor adaptation." Physical Review E.
