from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np


@dataclass
class OptimizationResult:
    x: np.ndarray
    fun: float
    grad_norm: float
    nit: int
    converged: bool
    history: List[float]


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    grad: np.ndarray,
    direction: np.ndarray,
    alpha_init: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    max_ls_iter: int = 50,
) -> float:
    """Armijo backtracking line search ensuring sufficient decrease."""
    alpha = alpha_init
    fx = f(x)
    directional_deriv = float(np.dot(grad, direction))

    if directional_deriv >= 0:
        return 1e-6

    for _ in range(max_ls_iter):
        if f(x + alpha * direction) <= fx + c1 * alpha * directional_deriv:
            return alpha
        alpha *= rho
    return alpha


def nonlinear_conjugate_gradient(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> OptimizationResult:
    """
    Polak-Ribiere+ Nonlinear Conjugate Gradient solver with automatic restarts
    and Armijo backtracking line search.
    """
    x = x0.astype(np.float64).copy()
    g = grad_f(x)
    d = -g
    history = [float(f(x))]

    for it in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        if grad_norm <= tol:
            return OptimizationResult(x, history[-1], grad_norm, it - 1, True, history)

        alpha = backtracking_line_search(f, x, g, d)
        x_next = x + alpha * d
        g_next = grad_f(x_next)

        # Polak-Ribiere+ parameter with non-negativity reset
        y = g_next - g
        denom = float(np.dot(g, g))
        beta = max(0.0, float(np.dot(g_next, y)) / denom) if denom > 0 else 0.0

        d = -g_next + beta * d
        if np.dot(d, g_next) >= 0:
            d = -g_next

        x, g = x_next, g_next
        history.append(float(f(x)))

    return OptimizationResult(x, history[-1], float(np.linalg.norm(g)), max_iter, False, history)


def bfgs_minimize(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> OptimizationResult:
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) Quasi-Newton optimizer maintaining
    a symmetric positive-definite inverse Hessian approximation H_k.
    """
    x = x0.astype(np.float64).copy()
    n = x.shape[0]
    H = np.eye(n, dtype=np.float64)
    g = grad_f(x)
    history = [float(f(x))]

    for it in range(1, max_iter + 1):
        grad_norm = float(np.linalg.norm(g))
        if grad_norm <= tol:
            return OptimizationResult(x, history[-1], grad_norm, it - 1, True, history)

        direction = -H @ g
        alpha = backtracking_line_search(f, x, g, direction)

        s = alpha * direction
        x_next = x + s
        g_next = grad_f(x_next)
        y = g_next - g

        sy = float(np.dot(s, y))
        if sy > 1e-14:
            rho = 1.0 / sy
            I = np.eye(n, dtype=np.float64)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        x, g = x_next, g_next
        history.append(float(f(x)))

    return OptimizationResult(x, history[-1], float(np.linalg.norm(g)), max_iter, False, history)


def soft_threshold(v: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal operator for the L1 norm: prox_{lambda ||.||_1}(v)."""
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)


def proximal_gradient_descent(
    smooth_f: Callable[[np.ndarray], float],
    grad_smooth_f: Callable[[np.ndarray], np.ndarray],
    l1_lambda: float,
    x0: np.ndarray,
    lipschitz_const: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> OptimizationResult:
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA / FISTA-style accelerated)
    for composite convex objectives: min_x smooth_f(x) + l1_lambda * ||x||_1.
    """
    x = x0.astype(np.float64).copy()
    y = x.copy()
    t = 1.0
    step_size = (1.0 / lipschitz_const) if lipschitz_const else 1e-2

    def total_obj(w: np.ndarray) -> float:
        return float(smooth_f(w) + l1_lambda * np.sum(np.abs(w)))

    history = [total_obj(x)]

    for it in range(1, max_iter + 1):
        grad_y = grad_smooth_f(y)
        x_next = soft_threshold(y - step_size * grad_y, step_size * l1_lambda)

        # Gradient mapping norm as composite optimality residual
        grad_map_norm = float(np.linalg.norm((y - x_next) / step_size))
        history.append(total_obj(x_next))

        if grad_map_norm <= tol:
            return OptimizationResult(x_next, history[-1], grad_map_norm, it, True, history)

        # Nesterov momentum extrapolation (FISTA)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_next + ((t - 1.0) / t_next) * (x_next - x)
        x, t = x_next, t_next

    return OptimizationResult(x, history[-1], grad_map_norm, max_iter, False, history)