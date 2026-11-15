from typing import Tuple, Union
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import svds


MatrixLike = Union[np.ndarray, sparse.spmatrix]


def power_iteration(
    A: MatrixLike, num_iter: int = 200, tol: float = 1e-9, seed: int = 42
) -> Tuple[float, np.ndarray]:
    """
    Computes the dominant eigenvalue and eigenvector of a symmetric matrix A
    using normalized power iteration.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    eigenvalue = 0.0

    for _ in range(num_iter):
        Av = A @ v
        new_eigenvalue = float(np.dot(v, Av))
        norm_Av = float(np.linalg.norm(Av))
        if norm_Av == 0.0:
            return 0.0, v
        v = Av / norm_Av
        if abs(new_eigenvalue - eigenvalue) <= tol:
            eigenvalue = new_eigenvalue
            break
        eigenvalue = new_eigenvalue

    return eigenvalue, v


def randomized_svd(
    A: MatrixLike,
    rank: int,
    n_oversamples: int = 10,
    n_power_iter: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Halko-Martinsson-Tropp Randomized SVD for fast low-rank matrix approximation
    on dense or scipy.sparse matrices.

    Returns:
        U: (m, rank), S: (rank,), Vt: (rank, n)
    """
    m, n = A.shape
    sketch_dim = min(rank + n_oversamples, min(m, n))
    rng = np.random.default_rng(seed)

    # Stage A: Construct orthonormal basis Q for approximate range of A
    Omega = rng.standard_normal((n, sketch_dim))
    Y = A @ Omega
    Q, _ = linalg.qr(Y, mode="economic")

    # Subspace power iterations to sharpen singular value decay
    for _ in range(n_power_iter):
        Z = A.T @ Q
        Q_z, _ = linalg.qr(Z, mode="economic")
        Y = A @ Q_z
        Q, _ = linalg.qr(Y, mode="economic")

    # Stage B: Project A onto small subspace B = Q^T A and compute exact SVD
    B = Q.T @ A
    if sparse.issparse(B):
        B = B.toarray()

    U_tilde, S, Vt = linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    return U[:, :rank], S[:rank], Vt[:rank, :]


def compare_sparse_svd(m: int = 1000, n: int = 800, density: float = 0.02, rank: int = 15) -> None:
    """Compares Randomized SVD against scipy.sparse.linalg.svds on a sparse matrix."""
    rng = np.random.default_rng(7)
    A = sparse.random(m, n, density=density, format="csr", random_state=rng)

    U_rand, S_rand, Vt_rand = randomized_svd(A, rank=rank, n_oversamples=10, n_power_iter=3)
    U_sp, S_sp, Vt_sp = svds(A, k=rank)

    # Sort SciPy svds singular values in descending order
    idx = np.argsort(S_sp)[::-1]
    S_sp = S_sp[idx]

    A_dense = A.toarray()
    rel_err_rand = np.linalg.norm(A_dense - (U_rand * S_rand) @ Vt_rand, "fro") / np.linalg.norm(A_dense, "fro")
    rel_err_sp = np.linalg.norm(A_dense - (U_sp[:, idx] * S_sp) @ Vt_sp[idx, :], "fro") / np.linalg.norm(A_dense, "fro")

    print(f"Sparse Matrix ({m}x{n}, density={density:.1%}), Target Rank={rank}")
    print(f"Top-5 Singular Values (Randomized SVD): {np.round(S_rand[:5], 4)}")
    print(f"Top-5 Singular Values (SciPy ARPACK):     {np.round(S_sp[:5], 4)}")
    print(f"Relative Frobenius Error (Randomized):  {rel_err_rand:.6f}")
    print(f"Relative Frobenius Error (SciPy ARPACK):{rel_err_sp:.6f}\n")


if __name__ == "__main__":
    compare_sparse_svd()