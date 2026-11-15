# Numerical Optimization & Spectral Solvers in SciPy

A first-principles scientific computing library implementing unconstrained quasi-Newton, nonlinear conjugate gradient, proximal composite solvers, and randomized low-rank matrix decompositions using **NumPy** and **SciPy**.

## Mathematical & Algorithmic Modules

1. **First- & Second-Order Convex/Non-Convex Solvers (`solvers.py`):**
   - **BFGS Quasi-Newton:** Updates symmetric positive-definite inverse Hessian approximations $H_k$ via secant pairs $(s_k, y_k)$ paired with Armijo backtracking line search.
   - **Polak-Ribiere+ Nonlinear Conjugate Gradient:** Exploits conjugate search directions with automatic steepest-descent restarts for high-dimensional quadratic and smooth objectives.
   - **FISTA Proximal Gradient Descent:** Solves non-smooth composite objectives $\min_x f(x) + \lambda \|x\|_1$ using exact Lipschitz step sizes $1/L$, Nesterov momentum extrapolation, and soft-thresholding proximal operators.

2. **Spectral Decomposition & Sparse Linear Algebra (`spectral_decomposition.py`):**
   - **Randomized SVD:** Implements the Halko-Martinsson-Tropp subspace sketching algorithm with power iterations for `scipy.sparse` matrices.
   - **Normalized Power Iteration:** Extracts dominant eigenpairs for spectral norm and Lipschitz constant estimation.

3. **Convergence & Conditioning Benchmarks (`convergence_analysis.py`):**
   - Benchmarks custom solvers against `scipy.optimize.minimize` and `scipy.sparse.linalg.svds` across ill-conditioned quadratic systems ($\kappa(A) = 10^4$), the non-convex Rosenbrock function, and sparse LASSO recovery.

## Installation & Usage

```bash
git clone [https://github.com/p-akinwunmi/scipy-convex-optimizers.git](https://github.com/p-akinwunmi/scipy-convex-optimizers.git)
cd scipy-convex-optimizers
pip install -r requirements.txt
python convergence_analysis.py