import numpy as np
from scipy import linalg, optimize
from solvers import (
    bfgs_minimize,
    nonlinear_conjugate_gradient,
    proximal_gradient_descent,
)
from spectral_decomposition import compare_sparse_svd


def benchmark_ill_conditioned_quadratic(dim: int = 50, cond_number: float = 1e4) -> None:
    """
    Tests solver convergence on a strongly convex quadratic f(x) = 0.5 x^T A x - b^T x
    with controlled spectral condition number kappa(A) = lambda_max / lambda_min.
    """
    rng = np.random.default_rng(123)
    Q, _ = linalg.qr(rng.standard_normal((dim, dim)))
    eigenvalues = np.geomspace(1.0, cond_number, num=dim)
    A = Q @ np.diag(eigenvalues) @ Q.T
    b = rng.standard_normal(dim)
    x_star = linalg.solve(A, b, assume_a="pos")

    f = lambda x: float(0.5 * x @ (A @ x) - b @ x)
    grad_f = lambda x: A @ x - b
    x0 = np.zeros(dim)

    res_cg = nonlinear_conjugate_gradient(f, grad_f, x0, tol=1e-8)
    res_bfgs = bfgs_minimize(f, grad_f, x0, tol=1e-8)
    res_scipy = optimize.minimize(f, x0, jac=grad_f, method="BFGS", tol=1e-8)

    print(f"=== Ill-Conditioned Quadratic (dim={dim}, kappa(A)={cond_number:.0e}) ===")
    print(f"Polak-Ribiere+ CG:  iters={res_cg.nit:<4} | ||x - x*||_2={np.linalg.norm(res_cg.x - x_star):.2e}")
    print(f"Custom BFGS:        iters={res_bfgs.nit:<4} | ||x - x*||_2={np.linalg.norm(res_bfgs.x - x_star):.2e}")
    print(f"SciPy BFGS:         iters={res_scipy.nit:<4} | ||x - x*||_2={np.linalg.norm(res_scipy.x - x_star):.2e}\n")


def benchmark_rosenbrock(dim: int = 10) -> None:
    """Benchmarks custom BFGS against SciPy on the non-convex Rosenbrock banana valley."""
    def rosen(x: np.ndarray) -> float:
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def rosen_grad(x: np.ndarray) -> np.ndarray:
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    x0 = np.full(dim, -1.2)
    res_bfgs = bfgs_minimize(rosen, rosen_grad, x0, tol=1e-7, max_iter=1000)
    res_scipy = optimize.minimize(rosen, x0, jac=rosen_grad, method="BFGS", tol=1e-7)

    print(f"=== Non-Convex Rosenbrock (dim={dim}, global min at 1_n) ===")
    print(f"Custom BFGS: iters={res_bfgs.nit:<4} | f(x)={res_bfgs.fun:.2e} | max|x - 1|={np.max(np.abs(res_bfgs.x - 1.0)):.2e}")
    print(f"SciPy BFGS:  iters={res_scipy.nit:<4} | f(x)={res_scipy.fun:.2e} | max|x - 1|={np.max(np.abs(res_scipy.x - 1.0)):.2e}\n")


def benchmark_sparse_lasso(n_samples: int = 200, n_features: int = 100) -> None:
    """Benchmarks FISTA Proximal Gradient Descent on L1-regularized least squares (LASSO)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)) / np.sqrt(n_samples)
    w_true = np.zeros(n_features)
    w_true[:8] = np.array([3.5, -2.8, 4.1, -1.9, 2.2, -3.0, 1.5, -2.4])
    y = X @ w_true + 0.05 * rng.standard_normal(n_samples)

    l1_lambda = 0.05
    L = float(np.linalg.norm(X, 2) ** 2)  # Exact Lipschitz constant of grad(0.5 ||Xw - y||^2)

    smooth_f = lambda w: float(0.5 * np.sum((X @ w - y) ** 2))
    grad_smooth_f = lambda w: X.T @ (X @ w - y)

    res_fista = proximal_gradient_descent(
        smooth_f, grad_smooth_f, l1_lambda, np.zeros(n_features), lipschitz_const=L, tol=1e-7
    )
    nnz = int(np.sum(np.abs(res_fista.x) > 1e-3))

    print(f"=== Sparse LASSO Recovery (n={n_samples}, p={n_features}, true_nnz=8) ===")
    print(f"FISTA Proximal GD: iters={res_fista.nit:<4} | obj={res_fista.fun:.6f} | recovered_nnz={nnz}\n")


if __name__ == "__main__":
    benchmark_ill_conditioned_quadratic()
    benchmark_rosenbrock()
    benchmark_sparse_lasso()
    compare_sparse_svd()