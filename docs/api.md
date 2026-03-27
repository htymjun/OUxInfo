# API Reference

This document provides an overview of the main API functions in the Shannon Entropy Estimator package.

## shannon_entropy(x, k=5)
- **Description:** Computes the Shannon entropy of the input data using the k-NN (KSG) estimator.
- **Parameters:**
  - `x` (ndarray, shape [N, dim]): Input data sequence.
  - `k` (int, optional): Number of nearest neighbors (default 5).
- **Returns:**
  - `double`: Shannon entropy.

## KL_div(x, y, k=5)
- **Description:** Computes the Kullback-Leibler divergence between two datasets using the k-NN (KSG) estimator (Pérez-Cruz variant).
- **Parameters:**
  - `x` (ndarray, shape [N, dim]): First dataset.
  - `y` (ndarray, shape [N, dim]): Second dataset.
  - `k` (int, optional): Number of nearest neighbors (default 5).
- **Returns:**
  - `double`: KL divergence.

## mutual_info(x, y, k=5, Thei=0)
- **Description:** Computes the mutual information between two datasets using the k-NN (KSG) estimator (Kraskov et al. 2004).
- **Parameters:**
  - `x` (ndarray, shape [N, dim]): First dataset.
  - `y` (ndarray, shape [N, dim]): Second dataset.
  - `k` (int, optional): Number of nearest neighbors (default 5).
  - `Thei` (int, optional): Length of Theiler window (default 0).
- **Returns:**
  - `double`: Mutual information.

## conditional_mutual_info(x, y, z, k=5)
- **Description:** Computes the conditional mutual information between x and y given z using the k-NN (KSG) estimator.
- **Parameters:**
  - `x` (ndarray, shape [N, dim]): First dataset.
  - `y` (ndarray, shape [N, dim]): Second dataset.
  - `z` (ndarray, shape [N, dim]): Conditioning dataset.
  - `k` (int, optional): Number of nearest neighbors (default 5).
- **Returns:**
  - `double`: Conditional mutual information.

## transfer_entropy(x, y, tau=1, m=1, lag=1, dt=1.0, k=5, trial=0)
- **Description:** Computes the transfer entropy from x to y using the k-NN (KSG) estimator.
- **Parameters:**
  - `x` (ndarray, shape [N, dim]): Source time series.
  - `y` (ndarray, shape [N, dim]): Target time series.
  - `tau` (int, optional): Length of time delay (default 1).
  - `m` (int, optional): Embedding dimension for y (default 1).
  - `lag` (int, optional): Time lag for embedding (default 1).
  - `dt` (float, optional): Physical time (default 1.0).
  - `k` (int, optional): Number of nearest neighbors (default 5).
  - `trial` (int, optional): Number of surrogate trials (default 0).
- **Returns:**
  - `double`: Transfer entropy.

## backward_transfer_entropy(x, y, tau=1, m=1, lag=1, dt=1.0, k=5, trial=0)
- **Description:** Computes the backward transfer entropy from x to y.
- **Parameters:**
  - `x` (ndarray, shape [N, dim]): Source time series.
  - `y` (ndarray, shape [N, dim]): Target time series.
  - `tau` (int, optional): Length of time delay (default 1).
  - `m` (int, optional): Embedding dimension for y (default 1).
  - `lag` (int, optional): Time lag for embedding (default 1).
  - `dt` (float, optional): Physical time (default 1.0).
  - `k` (int, optional): Number of nearest neighbors (default 5).
  - `trial` (int, optional): Number of surrogate trials (default 0).
- **Returns:**
  - `double`: The backward transfer entropy value.

## information_flow(x, y, tau=1, dt=1.0, k=5)
- **Description:** Computes the information flow from x to y using the k-NN (KSG) estimator.
- **Parameters:**
  - `x` (ndarray, shape [N, dim]): Source time series.
  - `y` (ndarray, shape [N, dim]): Target time series.
  - `tau` (int, optional): Length of time delay (default 1).
  - `dt` (float, optional): Physical time (default 1.0).
  - `k` (int, optional): Number of nearest neighbors (default 5).
- **Returns:**
  - `double`: Information flow.

## transfer_entropy_causal_map(X, tau, m=1, lag=1, dt=1.0, k=5, trial=0, n_threads=1)
- **Description:** Computes the transfer entropy causal map for multivariate time series using the k-NN (KSG) estimator.
- **Parameters:**
  - `X` (ndarray, shape [N, Nt] or [N, Nt, dim]): Multivariate time series.
  - `tau` (ndarray, shape [N]): Time delay for each variable.
  - `m` (int, optional): Embedding dimension for y (default 1).
  - `lag` (int, optional): Time lag for embedding (default 1).
  - `dt` (float, optional): Physical time (default 1.0).
  - `k` (int, optional): Number of nearest neighbors (default 5).
  - `trial` (int, optional): Number of surrogate trials (default 0).
  - `n_threads` (int, optional): Number of OpenMP threads (default 1).
- **Returns:**
  - `ndarray`: Transfer entropy causal map (N x N).

## information_flow_causal_map(X, tau, dt=1.0, k=5, n_threads=1)
- **Description:** Computes the information flow causal map for multivariate time series using the k-NN (KSG) estimator.
- **Parameters:**
  - `X` (ndarray, shape [N, Nt] or [N, Nt, dim]): Multivariate time series.
  - `tau` (ndarray, shape [N]): Time delay for each variable.
  - `dt` (float, optional): Physical time (default 1.0).
  - `k` (int, optional): Number of nearest neighbors (default 5).
  - `n_threads` (int, optional): Number of OpenMP threads (default 1).
- **Returns:**
  - `tuple(ndarray, ndarray, ndarray)`: (information flow map, leak map, dI map).

## TEIFL(X, tau, dt=1.0, lag=1, max_m=10, tol=0.01, k=5, n_threads=1, result_dir='.')
- **Description:** Computes transfer entropy, information flow, and related measures for a multivariate time series X.
- **Parameters:**
  - `X`: Multivariate time series (numpy array).
  - `tau` (ndarray, shape [N]): Time delay for each variable.
  - `dt` (float, optional): Physical time (default 1.0).
  - `lag` (int, optional): Time lag for embedding (default 1).
  - `max_m` (int, optional): Maximum embedding dimension for y (default 10).
  - `tol` (float, optional): Tolerance for mTE calculation (default 0.01).
  - `k` (int, optional): Number of nearest neighbors (default 5).
  - `n_threads` (int, optional): Number of OpenMP threads (default 1).
  - `result_dir` (str, optional) : The path of directory to save result (default '.').
- **Returns:**
  - Saves results to a compressed file in `result_dir`.
