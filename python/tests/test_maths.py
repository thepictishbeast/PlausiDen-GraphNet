"""Tests for graphnet.maths — Maths Panel operations (Phase 17 wave 1)."""

from __future__ import annotations

import math

import pytest
from graphnet import maths


def _have(name: str) -> bool:
    return maths.backends().get(name, False)


def test_backends_reports_known_set() -> None:
    b = maths.backends()
    assert "numpy" in b
    assert "scipy" in b
    assert "sklearn" in b


def test_list_operations_covers_three_pillars() -> None:
    ops = {name for name, _ in maths.list_operations()}
    # Linear algebra
    assert "svd" in ops
    assert "eigendecomp" in ops
    assert "pseudoinverse" in ops
    # Spectral
    assert "fft" in ops
    assert "ifft" in ops
    # Stats
    assert "histogram" in ops
    assert "shannon_entropy" in ops
    # Dimensionality reduction
    assert "pca" in ops


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_svd_round_trip() -> None:
    import numpy as np

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    r = maths.svd(a)
    reconstructed = r.u @ np.diag(r.s) @ r.vh
    assert np.allclose(reconstructed, a, atol=1e-8)


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_eigendecomp_identity_yields_ones() -> None:
    import numpy as np

    r = maths.eigendecomp(np.eye(3))
    assert np.allclose(sorted(r.values.real), [1.0, 1.0, 1.0])


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_eigendecomp_rejects_non_square() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="square matrix"):
        maths.eigendecomp(np.zeros((2, 3)))


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_matrix_rank_identity_is_n() -> None:
    import numpy as np

    assert maths.matrix_rank(np.eye(5)) == 5


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_pseudoinverse_round_trip() -> None:
    import numpy as np

    a = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    pinv = maths.pseudoinverse(a)
    # pinv @ a ≈ identity for full-column-rank case.
    assert np.allclose(pinv @ a, np.eye(2), atol=1e-8)


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_condition_number_identity_is_one() -> None:
    import numpy as np

    assert math.isclose(maths.condition_number(np.eye(3)), 1.0, rel_tol=1e-9)


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_fft_ifft_round_trip() -> None:
    import numpy as np

    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    spec = maths.fft(sig)
    recovered = maths.ifft(spec).real
    assert np.allclose(recovered, sig, atol=1e-8)


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_magnitude_spectrum_is_nonnegative() -> None:
    import numpy as np

    sig = np.sin(np.linspace(0.0, 2.0 * np.pi, 64))
    mag = maths.magnitude_spectrum(sig)
    assert (mag >= 0).all()


@pytest.mark.skipif(not _have("scipy"), reason="needs scipy")
def test_dct_returns_same_length() -> None:
    import numpy as np

    sig = np.arange(16.0)
    coeffs = maths.dct(sig)
    assert len(coeffs) == len(sig)


@pytest.mark.skipif(not _have("scipy"), reason="needs scipy")
def test_spectrogram_shape() -> None:
    import numpy as np

    sig = np.sin(np.linspace(0.0, 100.0, 1024))
    f, t, s = maths.spectrogram(sig, nperseg=64)
    assert s.shape == (len(f), len(t))


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_histogram_sums_to_n() -> None:
    import numpy as np

    values = np.random.default_rng(42).normal(size=1000)
    counts, edges = maths.histogram(values, bins=20)
    assert counts.sum() == 1000
    assert len(edges) == 21


def test_shannon_entropy_uniform_distribution() -> None:
    p = [0.5, 0.5]
    h = maths.shannon_entropy(p, base=2.0)
    assert math.isclose(h, 1.0, abs_tol=1e-9)


def test_shannon_entropy_dirac_is_zero() -> None:
    p = [1.0, 0.0, 0.0]
    h = maths.shannon_entropy(p)
    assert h == 0.0


def test_shannon_entropy_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        maths.shannon_entropy([0.5, -0.5, 1.0])


def test_shannon_entropy_rejects_unnormalised() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        maths.shannon_entropy([0.3, 0.3])


def test_kl_divergence_self_is_zero() -> None:
    p = [0.25, 0.25, 0.5]
    assert math.isclose(maths.kl_divergence(p, p), 0.0, abs_tol=1e-9)


def test_kl_divergence_infinite_on_missing_q() -> None:
    assert maths.kl_divergence([1.0, 0.0], [0.0, 1.0]) == float("inf")


def test_kl_divergence_shape_mismatch_errors() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        maths.kl_divergence([0.5, 0.5], [0.3, 0.3, 0.4])


@pytest.mark.skipif(not _have("sklearn"), reason="needs sklearn")
def test_pca_projects_to_requested_dim() -> None:
    import numpy as np

    data = np.random.default_rng(1).normal(size=(30, 50))
    out = maths.pca(data, n_components=3)
    assert out.shape == (30, 3)


@pytest.mark.skipif(not _have("numpy"), reason="needs numpy")
def test_gradient_descent_converges_on_quadratic() -> None:
    import numpy as np

    # Minimise f(x) = (x - 3)^2 + (y + 1)^2; gradient is 2*(x-3, y+1).
    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - np.array([3.0, -1.0]))

    res = maths.gradient_descent(grad, np.array([0.0, 0.0]), lr=0.1, steps=200)
    assert np.allclose(res.final_x, [3.0, -1.0], atol=1e-3)
    assert res.history.shape == (201, 2)
