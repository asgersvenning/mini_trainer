# import numpy as np
# from matplotlib import patches
# from matplotlib import pyplot as plt
# from matplotlib.axes import Axes
# from scipy.special import betainc, loggamma
# from tqdm import trange
# from tqdm.contrib.concurrent import thread_map


# def regular_simplex_points(dim: int) -> np.ndarray:
#     m = dim + 1
#     I = np.eye(m)
#     J = np.ones((m, m))
#     G = (1.0 + 1.0 / dim) * I - (1.0 / dim) * J
#     evals, evecs = np.linalg.eigh(G)
#     idx = np.argsort(evals)[::-1]
#     evals = evals[idx]
#     evecs = evecs[:, idx]
#     X = evecs[:, :dim] * np.sqrt(evals[:dim])
#     X /= np.linalg.norm(X, axis=1, keepdims=True)
#     return X


# def hypercube_points(dim: int) -> np.ndarray:
#     n = 2**dim
#     pts = np.empty((n, dim), dtype=float)
#     for i in range(n):
#         bits = np.array([(i >> k) & 1 for k in range(dim)], dtype=float)
#         pts[i] = 2.0 * bits - 1.0
#     pts /= np.sqrt(dim)
#     return pts


# def equally_spaced_circle_points(n: int) -> np.ndarray:
#     k = np.arange(n)
#     theta = 2 * np.pi * k / n
#     return np.stack([np.cos(theta), np.sin(theta)], axis=1)


# def generate_axis_aligned_equally_spaced_points(dim: int) -> np.ndarray:
#     pts = np.zeros((dim * 2, dim))
#     pts[np.arange(dim), np.arange(dim)] = 1
#     pts[dim + np.arange(dim), np.arange(dim)] = -1
#     return pts


# def deterministic_rotation(dim: int) -> np.ndarray:
#     i = np.arange(dim, dtype=float)[:, None]
#     j = np.arange(dim, dtype=float)[None, :]
#     phi = np.pi * (np.sqrt(5.0) - 1.0) / 2.0
#     A = np.sin(phi * (i + 1.0) * (j + 1.0))
#     Q, _ = np.linalg.qr(A)
#     if np.linalg.det(Q) < 0:
#         Q[:, -1] *= -1.0
#     return Q


# def equal_area_voronoi_points(dim: int, n: int, rotate: bool = True) -> np.ndarray:
#     if dim == 2:
#         pts = equally_spaced_circle_points(n)
#     elif n == 2:
#         pts = np.zeros((2, dim), dtype=float)
#         pts[0, 0] = 1.0
#         pts[1, 0] = -1.0
#     elif n == dim + 1:
#         pts = regular_simplex_points(dim)
#     elif n == 2 * dim:
#         pts = generate_axis_aligned_equally_spaced_points(dim)
#     elif n == 2**dim:
#         pts = hypercube_points(dim)
#     else:
#         raise ValueError(
#             f"No closed-form equal-area Voronoi configuration implemented for {dim=}, {n=}.\n"
#             f"Valid options for {dim=} are [2, {dim + 1}, {2 * dim}, {2 ** dim}]"
#         )

#     if rotate and dim > 2:
#         R = deterministic_rotation(dim)
#         pts = pts @ R.T

#     return pts


# def project_to_span(X: np.ndarray) -> np.ndarray:
#     R, C = X.shape
#     if C <= R:
#         return X
#     U, S, Vt = np.linalg.svd(X, full_matrices=False)
#     Z = (X @ Vt.T)[:, : X.shape[0]]
#     Z /= np.linalg.norm(Z, axis=1, keepdims=True)
#     return Z


# def symmetric_projection_matrix(dim: int) -> np.ndarray:
#     v1 = np.ones(dim, dtype=float)
#     v1 /= np.linalg.norm(v1)

#     v2 = np.zeros(dim, dtype=float)
#     v2[0] = 1.0
#     v2[1] = -1.0
#     v2 = v2 - v1 * np.dot(v1, v2)
#     v2 /= np.linalg.norm(v2)

#     P = np.stack([v1, v2], axis=0)
#     return P


# def hyperspherical_projection2d(X: np.ndarray) -> np.ndarray:
#     D = X.shape[1]
#     if D < 2:
#         raise RuntimeError(f"Unable to project degenerate hypersphere of dim={D}")
#     if D == 2:
#         return X

#     P = symmetric_projection_matrix(D)
#     Y = X @ P.T

#     r2 = np.sum(Y * Y, axis=1)
#     r2 = np.clip(r2, 0.0, 1.0)

#     b = 0.5 * (D - 2)
#     u = 1.0 - (1.0 - r2) ** b
#     r_prime = np.sqrt(u)

#     r = np.sqrt(r2)
#     scale = np.ones_like(r)
#     mask = r > 0
#     scale[mask] = r_prime[mask] / r[mask]

#     Y *= scale[:, None]
#     return Y


# def log_unit_sphere_area(dim: int) -> float:
#     return np.log(2) + (dim / 2) * np.log(np.pi) - loggamma(dim / 2)


# def sample_uniform_on_sphere(n: int, dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
#     if rng is None:
#         rng = np.random.default_rng()
#     x = rng.normal(0.0, 1.0, size=(n, dim))
#     return x / np.linalg.norm(x, axis=1, keepdims=True)


# def _normalize_rows(X: np.ndarray) -> np.ndarray:
#     nrm = np.linalg.norm(X, axis=1, keepdims=True)
#     nrm = np.maximum(nrm, 1e-300)
#     return X / nrm


# def _cap_fraction(dim: int, theta: np.ndarray) -> np.ndarray:
#     """
#     Fraction of surface area of S^{dim-1} inside angular radius theta around a pole.
#     theta in [0, pi].
#     """
#     theta = np.asarray(theta, dtype=float)
#     theta = np.clip(theta, 0.0, np.pi)

#     if dim == 2:
#         return theta / np.pi

#     a = 0.5 * (dim - 1.0)
#     b = 0.5
#     x = np.sin(theta) ** 2
#     I = betainc(a, b, x)

#     out = np.empty_like(theta)
#     mask = theta <= 0.5 * np.pi
#     out[mask] = 0.5 * I[mask]
#     out[~mask] = 1.0 - 0.5 * I[~mask]

#     out = np.where(theta <= 0.0, 0.0, out)
#     out = np.where(theta >= np.pi, 1.0, out)
#     return out


# def _sample_tangent_directions(
#     center: np.ndarray, n: int, rng: np.random.Generator, min_norm: float = 1e-12
# ) -> np.ndarray:
#     """
#     Sample u ~ Unif(S^{d-2}) in the tangent space at 'center' (u ⟂ center, ||u||=1).
#     """
#     d = center.shape[0]
#     c = center / np.linalg.norm(center)

#     U = np.empty((n, d), dtype=float)
#     filled = 0
#     while filled < n:
#         m = (n - filled) * 2
#         g = rng.normal(0.0, 1.0, size=(m, d))
#         g = g - (g @ c)[:, None] * c[None, :]
#         nrm = np.linalg.norm(g, axis=1)
#         ok = nrm > min_norm
#         g = g[ok]
#         if g.shape[0] == 0:
#             continue
#         take = min(g.shape[0], n - filled)
#         U[filled : filled + take] = g[:take] / np.linalg.norm(g[:take], axis=1, keepdims=True)
#         filled += take

#     return U


# def _circle_voronoi_proportions(points2: np.ndarray) -> np.ndarray:
#     """
#     Exact Voronoi proportions on S^1 using angular midpoints.
#     Works for dot-product Voronoi on the unit circle.
#     """
#     P = _normalize_rows(points2)
#     phi = np.arctan2(P[:, 1], P[:, 0])
#     order = np.argsort(phi)
#     phi_sorted = phi[order]

#     phi_ext = np.concatenate([phi_sorted, [phi_sorted[0] + 2 * np.pi]])
#     mids = 0.5 * (phi_ext[:-1] + phi_ext[1:])  # right boundaries
#     mids_left = np.concatenate([[mids[-1] - 2 * np.pi], mids[:-1]])  # left boundaries

#     lengths = mids - mids_left
#     props_sorted = lengths / (2 * np.pi)

#     props = np.empty_like(props_sorted)
#     props[order] = props_sorted
#     return props


# def estimate_voronoi_area_ray(
#     center_idx: int,
#     centers: np.ndarray,
#     dim: int,
#     n_samples: int,
#     rng: np.random.Generator,
#     debug: bool = False,
# ) -> tuple[float, float, dict]:
#     """
#     Unbiased estimator of a single cell's area proportion p_i on S^{dim-1}.

#     Returns:
#         mean: E[Y]
#         var:  sample variance of Y
#         dbg:  optional debug dict
#     """
#     N = centers.shape[0]
#     C = _normalize_rows(centers)
#     ci = C[center_idx]

#     if dim == 2:
#         props = _circle_voronoi_proportions(C[:, :2])
#         mean = float(props[center_idx])
#         var = 0.0
#         dbg = {"mode": "exact_circle"} if debug else {}
#         return mean, var, dbg

#     alpha = C @ ci  # (N,)
#     A = 1.0 - alpha
#     A[center_idx] = 0.0

#     U = _sample_tangent_directions(ci, n_samples, rng=rng)

#     S = U @ C.T  # (n_samples, N)
#     A_row = A[None, :].repeat(n_samples, axis=0)
#     T = np.arctan2(A_row, S)  # (n_samples, N) in [0, pi] since A>=0
#     T[:, center_idx] = np.pi
#     t_max = np.min(T, axis=1)

#     Y = _cap_fraction(dim, t_max)
#     mean = float(np.mean(Y))
#     var = float(np.var(Y, ddof=1)) if n_samples >= 2 else 0.0

#     dbg: dict = {}
#     if debug:
#         dbg = {
#             "t_max": t_max,
#             "Y": Y,
#             "alpha": alpha,
#             "A": A,
#             "mode": "ray_exit",
#         }
#     return mean, var, dbg


# def estimate_areas_ray(
#     points: np.ndarray,
#     pilot: int = 256,
#     rel_se_target: float = 0.02,
#     p_floor: float = 1e-10,
#     n_min: int = 256,
#     n_max: int = 200_000,
#     seed: int = 0,
#     debug_cell: int | None = 0,
#     out_path: str = "test.jpeg",
# ) -> dict:
#     """
#     Two-stage unbiased estimation:
#       - pilot run allocates per-cell sample sizes
#       - final run uses fresh independent samples only

#     Returns dict with proportions, standard errors, total proportion, and allocation stats.
#     """
#     N, D = points.shape
#     points_strict = project_to_span(_normalize_rows(points))
#     Ds = points_strict.shape[1]

#     rng = np.random.default_rng(seed)
#     rng_pilot = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
#     rng_main = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))

#     pilot_means = np.zeros(N, dtype=float)
#     pilot_vars = np.zeros(N, dtype=float)

#     dbg_payload: dict | None = None

#     def _pilot(i):
#         m, v, dbg = estimate_voronoi_area_ray(
#             i,
#             points_strict,
#             Ds,
#             n_samples=pilot,
#             rng=rng_pilot,
#             debug=(debug_cell is not None and i == debug_cell),
#         )
#         pilot_means[i] = m
#         pilot_vars[i] = v
#         if dbg and i == debug_cell:
#             dbg_payload = dbg
#     thread_map(_pilot, range(N), leave=False, desc="Pilot")

#     denom = np.maximum(pilot_means, p_floor) ** 2
#     n_alloc = np.ceil(pilot_vars / (rel_se_target**2 * denom)).astype(int)
#     n_alloc = np.clip(n_alloc, n_min, n_max)

#     proportions = np.zeros(N, dtype=float)
#     ses = np.zeros(N, dtype=float)

#     def _final(i):
#         m, v, _ = estimate_voronoi_area_ray(
#             i,
#             points_strict,
#             Ds,
#             n_samples=int(n_alloc[i]),
#             rng=rng_main,
#             debug=False,
#         )
#         proportions[i] = m
#         ses[i] = np.sqrt(v / max(int(n_alloc[i]), 1))
#     thread_map(_final, range(N), leave=False, desc="Final")

#     total_proportion = float(np.sum(proportions))
#     rel_error = abs(total_proportion - 1.0)

#     fig, axs = plt.subplots(1, 3 if dbg_payload is not None else 2, figsize=(16 if dbg_payload is not None else 12, 4))
#     if dbg_payload is None:
#         ax1, ax2 = axs.ravel()
#         ax3 = None
#     else:
#         ax1, ax2, ax3 = axs.ravel()

#     ax1: Axes
#     ax2: Axes

#     ax1.bar(range(N), proportions)
#     ax1.errorbar(range(N), proportions, yerr=ses, fmt="none", capsize=3)
#     ax1.axhline(1 / N, color="red", linestyle="--", label="Equal region area")
#     ax1.set_title(f"Voronoi Areas (Ray-Exit)\nΣp = {total_proportion:.6f} | Rel. Error = {rel_error:.2%}")
#     ax1.set_xticks(range(N))
#     ax1.set_xlabel("Region Index")
#     ax1.set_ylabel("Estimated Area Proportion")
#     ax1.legend()

#     proj = hyperspherical_projection2d(_normalize_rows(points))
#     ax2.add_patch(patches.Circle((0, 0), 1))
#     ax2.scatter(*proj.T, c="white", ec="black", s=220)
#     for i, (x, y) in enumerate(proj):
#         ax2.text(x, y, f" {i} ", horizontalalignment="center", verticalalignment="center_baseline")
#     ax2.set_title("Projection")
#     ax2.set_xlabel("x")
#     ax2.set_ylabel("y")
#     ax2.axis("equal")

#     if dbg_payload is not None and ax3 is not None and dbg_payload.get("mode", None) not in ["exact_circle", None]:
#         print(f'{type(dbg_payload)=} : {list(dbg_payload.keys())}')
#         t_max = dbg_payload["t_max"]
#         Y = dbg_payload["Y"]
#         ax3.hist(t_max, bins=40)
#         ax3.set_title(f"Debug cell {debug_cell}: exit angles\nmean(Y)={np.mean(Y):.4f} | n={len(Y)}")
#         ax3.set_xlabel("t_max (radians)")
#         ax3.set_ylabel("count")

#     plt.tight_layout()
#     plt.savefig(out_path)

#     return {
#         "proportions": proportions,
#         "ses": ses,
#         "sum": total_proportion,
#         "rel_error": rel_error,
#         "n_alloc": n_alloc,
#         "pilot_means": pilot_means,
#         "pilot_vars": pilot_vars,
#         "span_dim": Ds,
#         "out_path": out_path,
#     }


# def run_tests_and_examples(seed: int = 0) -> None:
#     cases: list[tuple[str, np.ndarray]] = []

#     cases.append(("random_on_S^99 (N=10)", sample_uniform_on_sphere(10, 100, rng=np.random.default_rng(seed))))
#     cases.append(("simplex in d=9 (N=10)", equal_area_voronoi_points(9, 10, rotate=True)))
#     cases.append(("axis-aligned in d=10 (N=20)", equal_area_voronoi_points(10, 20, rotate=True)))
#     cases.append(("hypercube in d=6 (N=64)", equal_area_voronoi_points(6, 64, rotate=True)))
#     cases.append(("circle equally spaced (N=12)", equal_area_voronoi_points(2, 12, rotate=False)))

#     for name, pts in cases:
#         res = estimate_areas_ray(
#             pts,
#             pilot=256,
#             rel_se_target=0.03,
#             n_min=256,
#             n_max=150_000,
#             seed=seed,
#             debug_cell=0,
#             out_path=f"test_{name.replace(' ', '_').replace('^', '').replace('(', '').replace(')', '').replace('=', '')}.jpeg",
#         )
#         p = res["proportions"]
#         se = res["ses"]
#         rel_error = res["rel_error"]
#         print(f"{name:28s} | span_dim={res['span_dim']:2d} | Σp={res['sum']:.6f} | rel_err={rel_error:.2%}")
#         print(f"  mean(p)={np.mean(p):.6f} vs 1/N={1/len(p):.6f} | max se={np.max(se):.6f} | min se={np.min(se):.6f}")


# if __name__ == "__main__":
#     run_tests_and_examples(seed=0)

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm.auto import trange


@torch.jit.script
def _betacf_jit(
    a: float, 
    b: float, 
    x: torch.Tensor, 
    max_iter: int = 200, 
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Continued fraction for incomplete beta function. 
    JIT-compiled for performance.
    """
    # Force float32 processing to match input, or promote to double for precision if needed.
    # Here we stick to input dtype but ensure stability with a tiny constant.
    dtype = x.dtype
    device = x.device
    
    # Pre-compute constants
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    
    tiny = 1e-30
    
    # Initialize c, d, h
    c = torch.ones_like(x)
    d = 1.0 - qab * x / qap
    
    # Avoid zero in denominator
    # Note: TorchScript requires explicit masking logic usually
    mask_d_small = torch.abs(d) < tiny
    d = torch.where(mask_d_small, torch.tensor(tiny, dtype=dtype, device=device), d)
    d = 1.0 / d
    h = d.clone()
    
    # Lentz's method loop
    for m in range(1, max_iter + 1):
        m_float = float(m)
        m2 = 2.0 * m_float
        
        # Even step
        aa = (m_float * (b - m_float) * x) / ((qam + m2) * (a + m2))
        
        d = 1.0 + aa * d
        d = torch.where(torch.abs(d) < tiny, torch.tensor(tiny, dtype=dtype, device=device), d)
        
        c = 1.0 + aa / c
        c = torch.where(torch.abs(c) < tiny, torch.tensor(tiny, dtype=dtype, device=device), c)
        
        d = 1.0 / d
        h = h * d * c
        
        # Odd step
        aa = -((a + m_float) * (qab + m_float) * x) / ((a + m2) * (qap + m2))
        
        d = 1.0 + aa * d
        d = torch.where(torch.abs(d) < tiny, torch.tensor(tiny, dtype=dtype, device=device), d)
        
        c = 1.0 + aa / c
        c = torch.where(torch.abs(c) < tiny, torch.tensor(tiny, dtype=dtype, device=device), c)
        
        d = 1.0 / d
        del_val = d * c
        h = h * del_val
        
        # Check convergence (using max error across batch)
        # We cannot break easily in a JIT vector loop without checking all, 
        # but typically we just run fixed iter or check 'all converged'.
        # For GPU efficiency, simpler to run all or break if max diff is small.
        if torch.max(torch.abs(del_val - 1.0)) < eps:
            break
            
    return h

def _betainc_reg_torch(
    a: float,
    b: float,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Regularized incomplete beta I_x(a,b). 
    Wraps the JIT kernel with symmetry logic.
    """
    x = x.clamp(0.0, 1.0)
    
    # Edge cases
    out = torch.empty_like(x)
    out = torch.where(x <= 0.0, torch.zeros_like(out), out)
    out = torch.where(x >= 1.0, torch.ones_like(out), out)
    
    mask_interior = (x > 0.0) & (x < 1.0)
    if not mask_interior.any():
        return out
        
    xi = x[mask_interior]
    
    # Symmetry threshold
    thresh = (a + 1.0) / (a + b + 2.0)
    use_sym = xi > thresh
    
    def _eval_log_beta(a_: float, b_: float, x_: torch.Tensor) -> torch.Tensor:
        # ln(beta(a,b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
        # Term: x^a * (1-x)^b / (a * beta(a,b))
        
        lbeta = torch.lgamma(torch.tensor(a_, device=x_.device)) + \
                torch.lgamma(torch.tensor(b_, device=x_.device)) - \
                torch.lgamma(torch.tensor(a_ + b_, device=x_.device))
        
        log_term = -lbeta + a_ * torch.log(x_) + b_ * torch.log1p(-x_)
        return torch.exp(log_term) / a_

    results = torch.empty_like(xi)
    
    # Case 1: x <= thresh
    if (~use_sym).any():
        x_lo = xi[~use_sym]
        factor = _eval_log_beta(a, b, x_lo)
        cf = _betacf_jit(a, b, x_lo)
        results[~use_sym] = factor * cf

    # Case 2: x > thresh (use symmetry I_x(a,b) = 1 - I_{1-x}(b,a))
    if use_sym.any():
        x_hi = xi[use_sym]
        x_sym = 1.0 - x_hi
        factor = _eval_log_beta(b, a, x_sym)
        cf = _betacf_jit(b, a, x_sym)
        results[use_sym] = 1.0 - (factor * cf)

    out[mask_interior] = results
    return out


def _cap_fraction_torch(dim: int, theta: torch.Tensor) -> torch.Tensor:
    """
    Fraction of surface area of S^{dim-1} inside angular radius theta around a pole.
    theta in [0, pi]. Works for all theta by using the correct piecewise form.
    """
    theta = theta.clamp(0.0, math.pi)

    if dim == 2:
        return theta / math.pi

    a = 0.5 * (dim - 1.0)
    b = 0.5
    x = torch.sin(theta) ** 2

    I = _betainc_reg_torch(a, b, x)

    out = torch.empty_like(theta)
    mask = theta <= 0.5 * math.pi
    out[mask] = 0.5 * I[mask]
    out[~mask] = 1.0 - 0.5 * I[~mask]

    out = torch.where(theta <= 0.0, torch.zeros_like(out), out)
    out = torch.where(theta >= math.pi, torch.ones_like(out), out)
    return out


# ----------------------------
# Your existing helpers (kept)
# ----------------------------

def project_to_span(X: torch.Tensor) -> torch.Tensor:
    R, C = X.shape
    if C <= R:
        return X
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    Z = (X @ Vt.T)[:, : X.shape[0]]
    Z /= Z.norm(2, 1, True)
    return Z

def symmetric_projection_matrix(dim: int) -> torch.Tensor:
    v1 = torch.ones(dim, dtype=torch.float32)
    v1 /= v1.sum().sqrt()

    v2 = torch.zeros(dim, dtype=torch.float32)
    v2[0] = 1.0
    v2[1] = -1.0
    v2 = v2 - v1 * (v1 @ v2.unsqueeze(-1))
    v2 /= v2.norm(2)

    P = torch.stack([v1, v2], dim=0)
    return P

def hyperspherical_projection2d(X: torch.Tensor) -> torch.Tensor:
    D = X.shape[1]
    if D < 2:
        raise RuntimeError(f"Unable to project degenerate hypersphere of dim={D}")
    if D == 2:
        return X

    P = symmetric_projection_matrix(D)
    Y = X @ P.T.to(X.device)

    r2 = (Y * Y).sum(1)
    r2 = torch.clip(r2, 0.0, 1.0)

    b = 0.5 * (D - 2)
    u = 1.0 - (1.0 - r2) ** b
    r_prime = u.sqrt()

    r = r2.sqrt()
    scale = torch.ones_like(r)
    mask = r > 0
    scale[mask] = r_prime[mask] / r[mask]

    Y *= scale.unsqueeze(-1)
    return Y

def sample_uniform_on_sphere(n: int, dim: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ----------------------------
# PyTorch core implementation
# ----------------------------

import torch


def _normalize_rows_torch(X: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    nrm = torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(eps)
    return X / nrm


@dataclass(frozen=True)
class RayExitConfig:
    pilot: int = 64
    rel_se_target: float = 0.02
    p_floor: float = 1e-10
    n_min: int = 128
    n_max: int = 200_000
    seed: int = 0

    batch_pilot: int = 32
    batch_final: int = 16

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    debug_cell: int | None = 0
    plot: bool = False
    out_path: str = "test.jpeg"

    # If set (e.g. 128, 256), uses only the K most similar competitors (approximate, not exact)
    max_competitors: int | None = None


def _estimate_cells_batch(
    indices: torch.Tensor,
    C: torch.Tensor,
    n_samples: int,
    dim: int,
    seeds: torch.Tensor,
    dtype_acc: torch.dtype = torch.float32,
    max_competitors: int | None = None,
    max_tensor_elements: int = 50_000_000,  # Conservative default (~200MB float32)
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Memory-safe batched estimator.
    Splits 'n_samples' into chunks to keep temporary tensors small.
    """
    device = C.device
    B = indices.numel()
    N, d = C.shape
    
    ci = C[indices]  # (B, d)

    # --- 1. Identify Competitors (Once per batch) ---
    if max_competitors is None:
        comp_idx = None
        C_comp = C
        K = N
    else:
        # Approximate: Select K nearest neighbors by dot product
        K = max(int(max_competitors), 2)
        sim = ci @ C.T  # (B, N)
        # Select top K. Note: this includes self.
        top_k = torch.topk(sim, k=min(K, N), dim=1, largest=True, sorted=False)
        comp_idx = top_k.indices # (B, K)
        
        # Gather C_comp: (B, K, d)
        # Flatten indices to use fancy indexing
        flat_indices = comp_idx.view(-1)
        C_comp = C[flat_indices].view(B, K, d)

    # --- 2. Pre-calculate static geometry terms ---
    if comp_idx is None:
        # Full N case
        alpha = ci @ C.T  # (B, N)
    else:
        # Top-K case: dot product between (B, d) and (B, K, d) -> (B, K)
        alpha = torch.einsum("bd,bkd->bk", ci, C_comp)
        
    A = (1.0 - alpha).clamp_min(0.0).to(dtype_acc)
    
    # --- 3. Setup Generators ---
    # We create a list of generators, one per batch element, to ensure 
    # the sequence of random numbers is identical to the non-chunked version.
    generators = []
    for b in range(B):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seeds[b].item()))
        generators.append(gen)

    # --- 4. Chunking Setup ---
    # We need to store S (B, chunk, K). 
    # Elements = B * chunk * K.
    # Solve for chunk: chunk = max_elements / (B * K)
    chunk_size = max_tensor_elements // (B * K)
    chunk_size = max(min(chunk_size, n_samples), 1)

    # Accumulators (using float64 for stability if possible)
    total_sum_Y = torch.zeros(B, device=device, dtype=torch.float64)
    total_sq_sum_Y = torch.zeros(B, device=device, dtype=torch.float64)
    
    debug_vals = None
    
    # --- 5. Execution Loop ---
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        current_n = end - start
        
        # Generate noise u (B, current_n, d)
        # We must iterate B to use the per-cell generators
        # This python loop is B=16 or 32, so it's negligible.
        g_list = []
        for b in range(B):
            g_b = torch.randn(current_n, d, generator=generators[b], device=device, dtype=C.dtype)
            g_list.append(g_b)
        g = torch.stack(g_list, dim=0) # (B, current_n, d)

        # Project to tangent space
        # proj = (g . ci) * ci
        proj = (g * ci[:, None, :]).sum(dim=2, keepdim=True)
        u = g - proj * ci[:, None, :]
        u = u / torch.linalg.norm(u, dim=2, keepdim=True).clamp_min(1e-30)

        # Compute max angle
        if comp_idx is None:
            # S: (B, current_n, N)
            S = (u.reshape(B * current_n, d) @ C.T).reshape(B, current_n, N).to(dtype_acc)
            
            # T = atan2(A, S)
            # We broadcast A (B, N) -> (B, 1, N)
            T = torch.atan2(A[:, None, :], S)
            
            # Mask self (index matching batch index)
            # indices is (B,) mapping batch_row -> global_index
            # We need to set T[:, :, indices[b]] = pi for row b
            # Create a mask or use scatter. Scatter is cleaner.
            
            # T is (B, n, N). We want to mask T[b, :, indices[b]]
            # Construct index tensor for scatter
            mask_indices = indices[:, None, None].expand(B, current_n, 1) # (B, n, 1)
            pi_tensor = torch.tensor(math.pi, device=device, dtype=dtype_acc)
            T.scatter_(2, mask_indices, pi_tensor)
            
        else:
            # S: (B, current_n, K)
            S = torch.einsum("bnd,bkd->bnk", u, C_comp).to(dtype_acc)
            T = torch.atan2(A[:, None, :], S)
            
            # Mask self in Top-K
            # comp_idx is (B, K). indices is (B,).
            # Identify where comp_idx[b, k] == indices[b]
            is_self = (comp_idx == indices[:, None]) # (B, K)
            if is_self.any():
                # Broadcast mask to (B, n, K)
                mask_broad = is_self[:, None, :].expand(B, current_n, K)
                T.masked_fill_(mask_broad, math.pi)

        # Min angle over neighbors
        t_max, _ = torch.min(T, dim=2) # (B, current_n)
        
        # Cap fraction
        Y_chunk = _cap_fraction_torch(dim, t_max) # (B, current_n)
        
        # Accumulate
        Y_sum = Y_chunk.sum(dim=1).to(torch.float64)
        total_sum_Y += Y_sum
        total_sq_sum_Y += (Y_chunk.to(torch.float64) ** 2).sum(dim=1)
        
        # Capture debug info from first chunk of first batch item if needed
        if debug_vals is None:
             debug_vals = {
                 "t_max": t_max[0].detach(), # Store only first in batch
                 "Y": Y_chunk[0].detach()
             }

    # --- 6. Finalize Statistics ---
    means = (total_sum_Y / n_samples).to(torch.float32)
    
    if n_samples >= 2:
        # Variance = (SumSq - (Sum^2)/n) / (n-1)
        mean_sq = total_sq_sum_Y / n_samples
        var_num = total_sq_sum_Y - (total_sum_Y**2 / n_samples)
        # Numerical stability clamp
        var_num = var_num.clamp_min(0.0)
        vars_ = (var_num / (n_samples - 1)).to(torch.float32)
    else:
        vars_ = torch.zeros_like(means)
        
    return means, vars_, {"chunked_debug": debug_vals}


def estimate_areas_ray_torch(points: np.ndarray, cfg: RayExitConfig) -> dict:
    """
    Two-stage allocation (pilot -> final), batched on GPU.
    Unbiased per cell and unbiased sum in expectation. No renormalization is applied.
    """
    X = points.float() if isinstance(points, torch.Tensor) else torch.tensor(points).clone().float()
    X /= X.norm(2, 1, True)

    Xs = project_to_span(X)
    N, Ds = Xs.shape

    device = torch.device(cfg.device)
    C = Xs.to(device, cfg.dtype)
    C = _normalize_rows_torch(C)

    base = np.random.default_rng(cfg.seed)
    pilot_seeds = torch.as_tensor(base.integers(0, 2**63 - 1, size=N, dtype=np.int64), device=device)
    final_seeds = torch.as_tensor(base.integers(0, 2**63 - 1, size=N, dtype=np.int64), device=device)

    pilot_means = torch.empty(N, device=device, dtype=torch.float32)
    pilot_vars = torch.empty(N, device=device, dtype=torch.float32)

    dbg_payload: dict | None = None

    # ---- Pilot (batched) ----
    for i0 in trange(0, N, cfg.batch_pilot, desc="Pilot", leave=False):
        i1 = min(i0 + cfg.batch_pilot, N)
        idx = torch.arange(i0, i1, device=device, dtype=torch.long)
        seeds = pilot_seeds[idx]

        debug_in_batch = None
        if cfg.debug_cell is not None and i0 <= cfg.debug_cell < i1:
            debug_in_batch = int(cfg.debug_cell - i0)

        means, vars_, dbg = _estimate_cells_batch(
            indices=idx,
            C=C,
            n_samples=cfg.pilot,
            dim=Ds,
            seeds=seeds,
            dtype_acc=torch.float32,
            max_competitors=cfg.max_competitors
        )
        pilot_means[idx] = means
        pilot_vars[idx] = vars_

        if dbg and cfg.debug_cell is not None and i0 <= cfg.debug_cell < i1:
            dbg_payload = dbg

    # allocation based on pilot (on device)
    denom = torch.clamp(pilot_means, min=cfg.p_floor) ** 2
    n_alloc = torch.ceil(pilot_vars / ((cfg.rel_se_target ** 2) * denom)).to(torch.int64)
    n_alloc = torch.clamp(n_alloc, min=cfg.n_min, max=cfg.n_max)

    # ---- Final (group by n_alloc; batched per group) ----
    proportions = torch.empty(N, device=device, dtype=torch.float32)
    ses = torch.empty(N, device=device, dtype=torch.float32)

    # group indices by allocated sample count (reduces kernel launch variety)
    n_alloc_cpu = n_alloc.detach().cpu().numpy()
    buckets: dict[int, list[int]] = {}
    for i, ni in enumerate(n_alloc_cpu):
        buckets.setdefault(int(ni), []).append(i)

    for ni, inds in buckets.items():
        inds_t = torch.as_tensor(inds, device=device, dtype=torch.long)
        for j0 in trange(0, inds_t.numel(), cfg.batch_final, desc=f"Final n={ni}", leave=False):
            j1 = min(j0 + cfg.batch_final, int(inds_t.numel()))
            idx = inds_t[j0:j1]
            seeds = final_seeds[idx]
            means, vars_, _ = _estimate_cells_batch(
                indices=idx,
                C=C,
                n_samples=int(ni),
                dim=Ds,
                seeds=seeds,
                dtype_acc=torch.float32,
                max_competitors=cfg.max_competitors
            )
            proportions[idx] = means
            ses[idx] = torch.sqrt(vars_ / max(int(ni), 1))

    proportions_cpu = proportions.detach().cpu().numpy()
    ses_cpu = ses.detach().cpu().numpy()
    total_proportion = float(np.sum(proportions_cpu))
    rel_error = abs(total_proportion - 1.0)

    if cfg.plot:
        # ---- Plotting ----
        fig, axs = plt.subplots(1, 3 if dbg_payload is not None else 2, figsize=(16 if dbg_payload is not None else 12, 4))
        if dbg_payload is None:
            ax1, ax2 = axs.ravel()
            ax3 = None
        else:
            ax1, ax2, ax3 = axs.ravel()

        ax1: Axes
        ax2: Axes

        ax1.bar(range(N), proportions_cpu)
        ax1.errorbar(range(N), proportions_cpu, yerr=ses_cpu, fmt="none", capsize=3)
        ax1.axhline(1 / N, color="red", linestyle="--", label="Equal region area")
        ax1.set_title(f"Voronoi Areas (Ray-Exit / Torch)\nΣp = {total_proportion:.6f} | Rel. Error = {rel_error:.2%}")
        ax1.set_xticks(range(N))
        ax1.set_xlabel("Region Index")
        ax1.set_ylabel("Estimated Area Proportion")
        ax1.legend()

        proj = hyperspherical_projection2d(X)
        ax2.add_patch(patches.Circle((0, 0), 1))
        ax2.scatter(*proj.cpu().T, c="white", ec="black", s=220)
        for i, (x, y) in enumerate(proj):
            ax2.text(x.item(), y.item(), f" {i} ", horizontalalignment="center", verticalalignment="center_baseline")
        ax2.set_title("Projection")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.axis("equal")

        if dbg_payload is not None and ax3 is not None:
            t_max = dbg_payload["t_max"].detach().cpu().numpy()
            Y = dbg_payload["Y"].detach().cpu().numpy()
            ax3.hist(t_max, bins=40)
            ax3.set_title(f"Debug cell {cfg.debug_cell}: exit angles\nmean(Y)={np.mean(Y):.4f} | n={len(Y)}")
            ax3.set_xlabel("t_max (radians)")
            ax3.set_ylabel("count")

        plt.tight_layout()
        plt.savefig(cfg.out_path)

    return {
        "proportions": proportions_cpu,
        "ses": ses_cpu,
        "sum": total_proportion,
        "rel_error": rel_error,
        "n_alloc": n_alloc_cpu,
        "pilot_means": pilot_means.detach().cpu().numpy(),
        "pilot_vars": pilot_vars.detach().cpu().numpy(),
        "span_dim": Ds,
        "out_path": cfg.out_path,
        "device": str(device),
        "dtype": str(cfg.dtype),
        "max_competitors": cfg.max_competitors,
    }


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    pts = sample_uniform_on_sphere(200, 100, seed=0)

    cfg = RayExitConfig(
        pilot=64,
        rel_se_target=0.03,
        n_min=128,
        n_max=50_000,
        seed=0,
        batch_pilot=32,
        batch_final=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        debug_cell=0,
        out_path="test_torch.jpeg",
        max_competitors=None,  # set to e.g. 256 for a fast approximate mode
    )

    res = estimate_areas_ray_torch(pts, cfg)
    print({k: res[k] for k in ["span_dim", "sum", "rel_error", "device", "max_competitors"]})

