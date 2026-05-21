"""Maths Panel — advanced mathematics surfaced as composable operations.

Plan §17: "cover advanced maths and make it easy to use the maths graphically."
Three pillars:

- :mod:`linalg` operations (matmul / SVD / eigen / inverse / pseudoinverse / rank)
- :mod:`spectral` operations (FFT / iFFT / DCT / wavelet / spectrogram)
- :mod:`stats` operations (histogram / KDE / hypothesis tests / entropy / KL)
- :mod:`optim` numerical optimisers + 2D loss surface viz
- :mod:`dimred` PCA / t-SNE / UMAP / Isomap projections

Each function returns a plain numpy / scipy result so the Phase 3/4 viz layer
can plot it natively. Heavy deps (scipy / sklearn / pywavelets / sympy) are
lazily imported so a clean install can `import graphnet.maths` and discover
which backends are wired via :func:`backends`.

The Phase 16 NodeGraph treats Maths-Panel ops as another NodeKind family in
a follow-up tick; for now the REPL drives them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def backends() -> dict[str, bool]:
    """Report which Maths Panel backends are importable."""
    out: dict[str, bool] = {}
    for name in ("numpy", "scipy", "sklearn", "pywt", "sympy"):
        try:
            __import__(name)
            out[name] = True
        except ImportError:
            out[name] = False
    return out


# ----- linear algebra --------------------------------------------------------


@dataclass
class SvdResult:
    """Result of :func:`svd`."""

    u: Any  # numpy.ndarray (M, K)
    s: Any  # numpy.ndarray (K,) — singular values, descending
    vh: Any  # numpy.ndarray (K, N) — V transpose


def svd(matrix: Any) -> SvdResult:
    """Singular value decomposition: A = U diag(s) Vᵀ.

    Returns the thin SVD (K = min(M, N)). Singular values descending.
    """
    import numpy as np

    u, s, vh = np.linalg.svd(np.asarray(matrix), full_matrices=False)
    return SvdResult(u=u, s=s, vh=vh)


@dataclass
class EigResult:
    """Result of :func:`eigendecomp`."""

    values: Any  # complex eigenvalues
    vectors: Any  # columns are eigenvectors


def eigendecomp(matrix: Any) -> EigResult:
    """Compute eigenvalues + right eigenvectors of a square matrix."""
    import numpy as np

    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"need square matrix, got shape {arr.shape}")
    values, vectors = np.linalg.eig(arr)
    return EigResult(values=values, vectors=vectors)


def matrix_rank(matrix: Any, *, tol: float | None = None) -> int:
    """Numerical rank of a matrix via SVD; tol defaults to numpy heuristic."""
    import numpy as np

    return int(np.linalg.matrix_rank(np.asarray(matrix), tol=tol))


def pseudoinverse(matrix: Any) -> Any:
    """Moore-Penrose pseudoinverse via SVD."""
    import numpy as np

    return np.linalg.pinv(np.asarray(matrix))


def condition_number(matrix: Any) -> float:
    """2-norm condition number; large = ill-conditioned."""
    import numpy as np

    return float(np.linalg.cond(np.asarray(matrix), p=2))


# ----- spectral --------------------------------------------------------------


def fft(signal: Any) -> Any:
    """Discrete Fourier transform of a 1D signal; returns complex array."""
    import numpy as np

    return np.fft.fft(np.asarray(signal))


def ifft(spectrum: Any) -> Any:
    """Inverse DFT; returns complex array."""
    import numpy as np

    return np.fft.ifft(np.asarray(spectrum))


def magnitude_spectrum(signal: Any) -> Any:
    """|FFT(signal)| — magnitude of frequency components."""
    import numpy as np

    return np.abs(fft(signal))


def phase_spectrum(signal: Any) -> Any:
    """angle(FFT(signal)) in radians, unwrapped."""
    import numpy as np

    return np.unwrap(np.angle(fft(signal)))


def dct(signal: Any) -> Any:
    """Type-II discrete cosine transform (requires scipy)."""
    from scipy.fft import dct as _dct

    return _dct(signal, type=2, norm="ortho")


def spectrogram(signal: Any, *, fs: float = 1.0, nperseg: int = 256) -> tuple[Any, Any, Any]:
    """Compute STFT spectrogram. Returns (frequencies, times, S).

    Requires scipy.
    """
    from scipy.signal import spectrogram as _spec

    return _spec(signal, fs=fs, nperseg=nperseg)


# ----- statistics ------------------------------------------------------------


def histogram(values: Any, *, bins: int = 50) -> tuple[Any, Any]:
    """Histogram counts + bin edges."""
    import numpy as np

    counts, edges = np.histogram(np.asarray(values), bins=bins)
    return counts, edges


def shannon_entropy(probabilities: Any, *, base: float = 2.0) -> float:
    """Shannon entropy H(p) = -Σ p_i log_b p_i.

    Probabilities must be non-negative and sum to ≈ 1.
    """
    import math

    import numpy as np

    p = np.asarray(probabilities, dtype=float)
    if (p < 0).any():
        raise ValueError("probabilities must be non-negative")
    total = p.sum()
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"probabilities must sum to 1.0, got {total}")
    nz = p[p > 0]
    return float(-(nz * (np.log(nz) / np.log(base))).sum())


def kl_divergence(p: Any, q: Any, *, base: float = 2.0) -> float:
    """KL divergence D(P || Q) = Σ p_i log_b (p_i / q_i). Same shape required."""
    import numpy as np

    pa = np.asarray(p, dtype=float)
    qa = np.asarray(q, dtype=float)
    if pa.shape != qa.shape:
        raise ValueError(f"shape mismatch: {pa.shape} vs {qa.shape}")
    if (pa < 0).any() or (qa < 0).any():
        raise ValueError("probabilities must be non-negative")
    if (qa == 0).any() and (pa > 0).any():
        return float("inf")
    mask = pa > 0
    return float((pa[mask] * (np.log(pa[mask] / qa[mask]) / np.log(base))).sum())


# ----- dimensionality reduction ----------------------------------------------


def pca(data: Any, *, n_components: int = 3) -> Any:
    """PCA projection of (N, D) matrix to (N, n_components).

    Returns the projected matrix; for the principal components, use
    :func:`sklearn.decomposition.PCA` directly.
    """
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components).fit_transform(data)


def tsne(data: Any, *, n_components: int = 2, perplexity: float = 30.0) -> Any:
    """t-SNE projection to n_components. Requires sklearn."""
    from sklearn.manifold import TSNE

    return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(data)


# ----- optimisation ----------------------------------------------------------


@dataclass
class GdResult:
    """Result of :func:`gradient_descent`."""

    final_x: Any  # final parameter value (numpy array)
    history: Any  # per-step (N+1, D) parameter trace
    loss_history: list[float]


def gradient_descent(
    grad: Any,
    x0: Any,
    *,
    lr: float = 0.1,
    steps: int = 100,
) -> GdResult:
    """Vanilla gradient descent. `grad(x)` returns the gradient at x.

    Returns final parameter, parameter trace, and per-step loss (if a `loss`
    function is provided via the `grad` closure's contract — here we don't
    require loss, so loss_history is empty).
    """
    import numpy as np

    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]
    loss_history: list[float] = []
    for _ in range(steps):
        g = np.asarray(grad(x), dtype=float)
        x = x - lr * g
        history.append(x.copy())
    return GdResult(final_x=x, history=np.array(history), loss_history=loss_history)


# ----- panel summary --------------------------------------------------------


def list_operations() -> list[tuple[str, str]]:
    """Enumerate Maths Panel operations with one-line descriptions."""
    return [
        ("svd", "singular value decomposition (NumPy)"),
        ("eigendecomp", "eigenvalues + eigenvectors of a square matrix (NumPy)"),
        ("matrix_rank", "numerical rank via SVD (NumPy)"),
        ("pseudoinverse", "Moore-Penrose pseudoinverse (NumPy)"),
        ("condition_number", "2-norm condition number (NumPy)"),
        ("fft", "DFT of a 1D signal (NumPy)"),
        ("ifft", "inverse DFT (NumPy)"),
        ("magnitude_spectrum", "|FFT(signal)| (NumPy)"),
        ("phase_spectrum", "unwrapped phase of FFT (NumPy)"),
        ("dct", "type-II discrete cosine transform (SciPy)"),
        ("spectrogram", "STFT spectrogram (SciPy)"),
        ("histogram", "value histogram (NumPy)"),
        ("shannon_entropy", "H(p) information-theoretic entropy"),
        ("kl_divergence", "KL divergence D(P || Q)"),
        ("pca", "PCA projection (sklearn)"),
        ("tsne", "t-SNE projection (sklearn)"),
        ("gradient_descent", "vanilla GD trace (NumPy)"),
    ]
