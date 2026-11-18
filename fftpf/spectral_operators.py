"""Shared k-space operator layer: the ONE implementation of every spectral
symbol, Nyquist mask and E_ext / static-eigenstrain normaliser that the three
engines (solver.py 2D periodic, bulk3d.py 3D periodic, film3d.py in-plane
rfft2 x z finite differences) and the sector modules (kittel_afe, octa_tilt,
grad4, polycrystal) use. The engines keep their old private names as thin
aliases / delegating wrappers, so nothing that imports them moves; what they
can no longer do is carry a private copy that drifts (the film once failed to
mask A12 on even Nyquist lines while solver.py did; the ensemble normaliser
once had to be patched in five places). docs/OPERATORS.md: what lives where,
how to add a symbol.

This module is a LEAF: numpy only, no engine import, no FFT. Transforms stay
in the engines because each owns a deliberately different FFT_WORKERS knob.
Every function here reproduces, operation for operation, the code it
replaced, so routing an engine through it is bit-for-bit neutral.

=============================================================================
CONVENTIONS (stated once; every docstring below relies on them)
=============================================================================
GRID AND SPECTRUM LAYOUT. Real fields are component-first, space last:
(3, [S,] [Nz,] Ny, Nx), axis order z, y, x. Every engine FFT is real, so the
spectral arrays are the rfft half-spectrum: the LAST axis (x) is halved to
Nxr = Nx//2 + 1 bins (rfftfreq, kx >= 0) and y (and z) keep their full length
in fftfreq order (negative frequencies in the upper half). The grid dict
carries the half-spectrum wavevector arrays KXr, KYr[, KZr], each already
broadcast to the full spectral shape (Ny, Nxr) or (Nz, Ny, Nxr), plus the
full-spectrum KX, KY[, KZ] (shape (Ny, Nx) / (Nz, Ny, Nx)) that only the
real-space IC generators read. Wavenumbers are angular: k = 2*pi*f/h.
`dim` is the number of FOURIER axes: 2 for solver.py and for the film (whose
in-plane grid IS solver.make_grid's, and whose z axis is a node grid with no
Fourier index), 3 for bulk3d.py. Every k-space multiplier is even under the
joint flip k -> -k, which is what makes a real elementwise multiplier
Hermitian-consistent on the half-spectrum (the WP4 §10b argument).

NYQUIST RULE (one rule, every engine, every symbol). On an axis of EVEN
length N the bin at index N//2 -- in the full (fftfreq) and the half
(rfftfreq) layout alike -- is the single self-conjugate Nyquist wavenumber
+-pi/h. A symbol ODD in a wavevector component k_j has no single real value
there, so it is DECLARED zero on the Nyquist line/plane of axis j whenever
N_j is even:
  * first-derivative wavevectors (masked_wavevectors_*, derivative_symbols_*):
    component j on its own axis j;
  * mixed second-order symbols odd x odd in (i, j) -- the gradient A_ij and
    the depolarisation D_ij, i != j -- on both the i and the j plane (on an
    intersection both rules give 0, so no corner case exists);
  * the film's mixed z-terms, which carry i*k (derivative_symbols_2d).
EVEN symbols are never masked: A_ii, D_ii, |k|^2, the H11 sum k_j^4, second
derivatives -k_j^2. An odd axis has no Nyquist bin, so the rule is the
identity there and odd-grid numbers never depend on it. This is a spectral
CONVENTION, not a filter: force, implicit operator, Green operator and energy
all use the same masked symbols, so they stay adjoint. The ENERGY side of the
rule is add_nyquist_correction: a collocation density built from masked
first derivatives drops each even axis' alternating mode; adding back
0.5*a_ij*(pi/h_j)^2*(NyquistProject_j f_i)^2 makes its grid mean the
quadratic form whose gradient is the unmasked-diagonal k^2 force.

VOIGT / TENSOR SHEAR CONVENTION. Stiffness is always given as the cubic Voigt
triple (C11, C12, C44) and expanded by elastic.build_C_tensor with
C_1212 = C44, i.e. sigma_12 = 2*C44*e_12 on the TENSOR strain e_12 =
gamma_12/2. The Green operator here takes that full tensor C_ijkl. The
eigenstrain is NOT shared, on purpose (SPEC §37.5's named trap): elastic.py
and film_elastic use the Voigt-flavoured Q44 (e_12 = 0.5*Q44*P1*P2), bulk3d
the tensor Q1212 (e_12 = 2*Q1212*P1*P2), so Q44 = 4*Q1212. No function in this
module ever receives a Q.

ENSEMBLE AXIS. A batched state carries the realization axis S right after the
component axes: P (3,S,...), E_ext field (3,S,...), static eigenstrain
(3,3,S,...). A P-independent per-realization field ((3,...) or (3,3,...))
meeting a batched quantity gets S inserted (shared by all members); anything
else is refused, never left to numpy's left-padding (ensemble_field).
"""
import numpy as np

# axis name -> (grid size key, spacing key, half-spectrum K key, array axis)
_AXIS = {
    "x": ("Nx", "hx", "KXr", -1),
    "y": ("Ny", "hy", "KYr", -2),
    "z": ("Nz", "hz", "KZr", -3),
}
AXES = ("x", "y", "z")


def _seqsum(terms):
    """terms[0] + terms[1] + ... evaluated left to right, exactly as the
    written-out sums it replaces (a one-term sum is the term itself)."""
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def _axes(dim):
    if dim not in (2, 3):
        raise ValueError(f"operators: dim must be 2 or 3, got {dim!r}")
    return AXES[:dim]


# ---------------------------------------------------------------------------
# Grid / wavevector construction
# ---------------------------------------------------------------------------
def k_axis(N, h):
    """Angular wavenumbers of an N-point axis of spacing h, fftfreq order."""
    return 2 * np.pi * np.fft.fftfreq(N, d=h)


def k_axis_half(N, h):
    """Angular wavenumbers of the halved (rfft) axis: N//2+1 bins, k >= 0."""
    return 2 * np.pi * np.fft.rfftfreq(N, d=h)


def k_mesh(*k1d):
    """Broadcast 1D wavenumber axes given in (x, y[, z]) order to full
    arrays of shape ([Nz,] Ny, Nx'), returned in the same (x, y[, z]) order.
    Values and memory layout equal np.meshgrid(kx, ky) in 2D and
    np.meshgrid(kz, ky, kx, indexing='ij') in 3D (both C-contiguous copies)."""
    return tuple(reversed(np.meshgrid(*reversed(k1d), indexing="ij")))


def half_spectrum(grid, dim):
    """The grid's half-spectrum wavevector arrays (KXr, KYr[, KZr]) -- the
    grid's own arrays, not copies."""
    return tuple(grid[_AXIS[a][2]] for a in _axes(dim))


# ---------------------------------------------------------------------------
# The Nyquist rule
# ---------------------------------------------------------------------------
def nyquist_index(grid, axis):
    """Index of `axis`' Nyquist bin (N//2) if its length N is even, else None."""
    N = grid[_AXIS[axis][0]]
    return N // 2 if N % 2 == 0 else None


def mask_nyquist(sym, grid, axes):
    """Zero `sym` IN PLACE on the Nyquist line/plane of every EVEN axis in
    `axes` (names from 'x', 'y', 'z'); `sym` is indexed with negative array
    axes, so leading axes pass through. Returns `sym`."""
    for a in axes:
        n = nyquist_index(grid, a)
        if n is not None:
            idx = [slice(None)] * sym.ndim
            idx[_AXIS[a][3]] = n
            sym[tuple(idx)] = 0.0
    return sym


def masked_wavevectors(grid, dim):
    """Real skew-adjoint first-derivative wavevectors: copies of
    (KXr, KYr[, KZr]), each zeroed on its OWN even-axis Nyquist line/plane.
    The derivative operator i*k_j is then real skew-adjoint (no +-pi/h sign
    ambiguity), which is what the collocation gradient density, the Green
    operator and the film's odd mixed terms need."""
    out = tuple(K.copy() for K in half_spectrum(grid, dim))
    for K, a in zip(out, _axes(dim)):
        mask_nyquist(K, grid, (a,))
    return out


def masked_wavevectors_2d(grid):
    """masked_wavevectors(grid, 2): (KX, KY) on the rfft2 half-spectrum.
    Was solver._modified_wavevectors_2d and elastic._modified_wavevectors_2d."""
    return masked_wavevectors(grid, 2)


def masked_wavevectors_3d(grid):
    """masked_wavevectors(grid, 3): (KX, KY, KZ) on the rfftn half-spectrum.
    Was bulk3d._modified_wavevectors_3d."""
    return masked_wavevectors(grid, 3)


def derivative_symbols(grid, dim):
    """(i*kx, i*ky[, i*kz]) with the masked wavevectors: the first-derivative
    symbols, odd, Nyquist bin zeroed on each one's own axis."""
    return tuple(1j * K for K in masked_wavevectors(grid, dim))


def derivative_symbols_2d(grid):
    """derivative_symbols(grid, 2) -- the film's in-plane odd symbols (its
    mixed z-terms, the Poisson source/recovery, film_elastic's strains).
    Was film3d._odd_k."""
    return derivative_symbols(grid, 2)


# ---------------------------------------------------------------------------
# Even symbols: |k|^2, isotropic c*|k|^2, the H11 k^4 sum
# ---------------------------------------------------------------------------
def k_squared(grid, dim):
    """|k|^2 = kx^2 + ky^2 [+ kz^2] on the half-spectrum, unmasked (even)."""
    return _seqsum([K**2 for K in half_spectrum(grid, dim)])


def isotropic_symbol(grid, coeff, dim):
    """coeff*|k|^2, the scalar isotropic gradient symbol shared by every
    component (kittel_afe's Aq, octa_tilt's Atheta)."""
    return coeff * k_squared(grid, dim)


def hog_symbol(grid, dim):
    """kx^4 + ky^4 [+ kz^4]: the component-wise H11 multiplier (NC2025 eq.3,
    grad4's module docstring) -- NOT |k|^4. Even, never masked."""
    return _seqsum([K**4 for K in half_spectrum(grid, dim)])


def add_hog(diagonals, H11, k4):
    """(A11 + H11*k4, A22 + H11*k4, A33 + H11*k4): H11 joins the three
    diagonal gradient multipliers, the off-diagonals and D(k) untouched."""
    return tuple(A + H11 * k4 for A in diagonals)


# ---------------------------------------------------------------------------
# Gradient symbols A(k) (SPEC §1.2)
# ---------------------------------------------------------------------------
def grad_symbols(grid, G11, G12, G44, G44p, dim):
    """The base-G gradient operator A(k) as a dict {(i, j): array}, i <= j,
    i, j in 0..2 (components), on the half-spectrum:
        A_ii = G11*k_i^2 + (G44+G44p)*sum_{m != i} k_m^2   (i < dim)
        A_33 = (G44+G44p)*(kx^2 + ky^2)                     (dim 2: no k3)
        A_ij = (G12+G44-G44p)*k_i*k_j, Nyquist-masked on axes i and j.
    In 2D the off-diagonals involving component 3 are identically zero and
    are not returned (k3 == 0). Diagonals are even and unmasked."""
    K = half_spectrum(grid, dim)
    axes = _axes(dim)
    A = {}
    for i in range(3):
        if i < dim:
            A[i, i] = (G11 * K[i]**2
                       + (G44 + G44p) * _seqsum([K[m]**2 for m in range(dim) if m != i]))
        else:
            A[i, i] = (G44 + G44p) * _seqsum([K[m]**2 for m in range(dim)])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if j < dim:
            A[i, j] = mask_nyquist((G12 + G44 - G44p) * K[i] * K[j], grid, (axes[i], axes[j]))
    return A


def grad_symbols_2d(grid, G11, G12, G44, G44p):
    """(A11, A22, A12, A33) on the rfft2 half-spectrum -- the 2D engine's and
    the film's in-plane A(k). Was solver._grad_symbols (and the body of
    film3d._grad_A2d)."""
    A = grad_symbols(grid, G11, G12, G44, G44p, 2)
    return A[0, 0], A[1, 1], A[0, 1], A[2, 2]


def grad_symbols_3d(grid, G11, G12, G44, G44p):
    """(A11, A22, A33, A12, A13, A23) on the rfftn half-spectrum. Was
    bulk3d._grad_symbols_3d."""
    A = grad_symbols(grid, G11, G12, G44, G44p, 3)
    return A[0, 0], A[1, 1], A[2, 2], A[0, 1], A[0, 2], A[1, 2]


def grad_nyquist_coefficients(G11, G44, G44p, dim):
    """a[i, j], the coefficient of the (P_i, axis j) Nyquist energy term of
    the G-gradient: G11 on the diagonal i == j, G44+G44p elsewhere (the P3
    row of the 2D slice included, matching A33). Shape (3, dim)."""
    a = np.full((3, dim), G44 + G44p)
    for i in range(dim):
        a[i, i] = G11
    return a


# ---------------------------------------------------------------------------
# Depolarisation symbols D(k) (SPEC §1.3): -E_d = (1/kappa0) D(k) P(k)
# ---------------------------------------------------------------------------
def depolarization_symbols(grid, dim):
    """D_ij = k_i k_j / |k|^2 as a dict {(i, j): array}, i <= j < dim, with
    D(k=0) = 0 and the off-diagonals Nyquist-masked on axes i and j. In 2D
    (k3 == 0) every entry involving component 3 is identically zero and is
    not returned."""
    K = half_spectrum(grid, dim)
    axes = _axes(dim)
    k2 = _seqsum([Kj**2 for Kj in K])
    k2safe = np.where(k2 == 0.0, 1.0, k2)
    D = {}
    for i in range(dim):
        D[i, i] = np.where(k2 == 0.0, 0.0, K[i] * K[i] / k2safe)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if j < dim:
            D[i, j] = mask_nyquist(np.where(k2 == 0.0, 0.0, K[i] * K[j] / k2safe),
                                   grid, (axes[i], axes[j]))
    return D


def depolarization_symbols_2d(grid):
    """(D11, D22, D12) on the rfft2 half-spectrum (solver.build_operators)."""
    D = depolarization_symbols(grid, 2)
    return D[0, 0], D[1, 1], D[0, 1]


def depolarization_symbols_3d(grid):
    """(D11, D22, D33, D12, D13, D23) on the rfftn half-spectrum
    (bulk3d.build_operators_3d)."""
    D = depolarization_symbols(grid, 3)
    return D[0, 0], D[1, 1], D[2, 2], D[0, 1], D[0, 2], D[1, 2]


# ---------------------------------------------------------------------------
# Homogeneous elastic Green operator Omega(k) = [C_ijkl k_j k_l]^-1 (SPEC §1.4)
# ---------------------------------------------------------------------------
# The historical einsum spellings of the two engines. They are numerically the
# same contraction (measured byte-identical); each dim keeps its own spelling
# so the zero-change claim does not rest on that measurement.
_GREEN_SUBSCRIPTS = {2: "ijkl,jyx,lyx->ikyx", 3: "ijkl,j...,l...->ik..."}


def green_operator(grid, C, dim):
    """Omega(k) = [C_ijkl k_j k_l]^-1 built from the full (3,3,3,3) tensor C
    and the MASKED wavevectors (so the even quadratic-in-k elastic operator
    is a real quadratic form whose variation is the force). Every zero-K mode
    -- DC plus each intersection of even-axis Nyquist lines/planes where all
    surviving components vanish -- is regularised with I before the batched
    inverse and set to 0 after. Returns (Omega (...,3,3), K (3,...), zero_k);
    in 2D K's third row is exact zeros (k3 == 0)."""
    Km = masked_wavevectors(grid, dim)
    if dim == 2:
        K = np.stack([Km[0], Km[1], np.zeros_like(Km[0])], axis=0)
    else:
        K = np.stack(Km, axis=0)
    Omega_inv = np.einsum(_GREEN_SUBSCRIPTS[dim], C, K, K)    # (3,3,...)
    Omega_inv = np.moveaxis(Omega_inv, (0, 1), (-2, -1))     # (...,3,3)
    k2 = _seqsum([Kj**2 for Kj in Km])
    zero_k = (k2 == 0.0)
    Omega_inv_safe = Omega_inv.copy()
    Omega_inv_safe[zero_k] = np.eye(3)
    Omega = np.linalg.inv(Omega_inv_safe)
    Omega[zero_k] = 0.0  # placeholder is irrelevant: multiplies k_j = 0 downstream
    return Omega, K, zero_k


# ---------------------------------------------------------------------------
# Energy side of the Nyquist rule
# ---------------------------------------------------------------------------
def nyquist_project_axis(field, axis, N):
    """Project one real field onto the alternating (Nyquist) pattern along a
    single axis: alternating-sign axial mean, expanded back to the full
    shape. This extracts pattern_j f (pattern_j = (-1)^index_j), NOT a
    low-pass filter -- every other Fourier mode is orthogonal to the pattern
    and contributes zero. `axis` is the (negative) spatial axis, so an
    ensemble axis in front broadcasts untouched. Was
    solver._nyquist_project_axis."""
    pat = (-1.0) ** np.arange(N)
    shape = [1] * field.ndim
    shape[axis] = N
    pat = pat.reshape(shape)
    return pat * (pat * field).mean(axis=axis, keepdims=True)


def add_nyquist_correction(fg, fields, grid, coeff, dim):
    """fg plus, for every EVEN axis j (x, y[, z] order) and component i,
    0.5*a_ij*(pi/h_j)^2*(nyquist_project_axis(fields[i], j))^2 -- the
    alternating mode a masked-derivative collocation density drops (module
    docstring, Nyquist rule). `coeff` is a scalar (isotropic: the same a for
    every (i, j)) or a (n_comp, dim) table (grad_nyquist_coefficients).
    Odd axes add nothing, so on an all-odd grid fg is returned unchanged."""
    for j, a in enumerate(_axes(dim)):
        Nkey, hkey, _K, axis = _AXIS[a]
        N = grid[Nkey]
        if N % 2 == 0:
            k2 = (np.pi / grid[hkey]) ** 2
            for i in range(len(fields)):
                c = coeff if np.ndim(coeff) == 0 else coeff[i, j]
                fg = fg + 0.5 * c * k2 * nyquist_project_axis(fields[i], axis, N) ** 2
    return fg


# ---------------------------------------------------------------------------
# E_ext / static-eigenstrain normalisation
# ---------------------------------------------------------------------------
def ensemble_field(v, n_lead, ndim, what):
    """Align a P-INDEPENDENT field with a BATCHED quantity (ensemble axis S
    right after the n_lead component axes): eps_star_static (n_lead=2,
    (3,3,...)) and a field-shaped E_ext (n_lead=1, (3,...)). `ndim` is the
    batched quantity's own ndim, so a per-realization v has ndim-1 axes and
    gets S inserted at n_lead (broadcast to every member); an already
    ensemble-shaped v (ndim axes, member-resolved) is returned as given.
    Anything else raises: numpy would left-pad it against the component axes
    instead -- at S=3 member s silently got v[j,s] in slot [i,j,s], at S=1 the
    output grew an axis, other S errored. Callers use it ONLY when P carries
    the ensemble axis, so every unbatched path is untouched. Was
    elastic._ensemble_field."""
    v = np.asarray(v)
    if v.ndim in (ndim - 1, ndim) and v.shape[:n_lead] == (3,) * n_lead:
        return np.expand_dims(v, n_lead) if v.ndim == ndim - 1 else v
    raise ValueError(f"{what} of shape {v.shape} cannot meet a batched (ensemble) field of "
                     f"ndim {ndim}: give it the per-realization field shape "
                     f"({', '.join(['3'] * n_lead)}, *space) or the ensemble shape "
                     f"({', '.join(['3'] * n_lead)}, S, *space)")


def vec3_field(v, ndim, ensemble=False):
    """The periodic engines' E_ext normaliser (solver, bulk3d, the sectors).
    A (3,) vector becomes (3,1,...,1) with ndim-1 trailing singleton axes
    (plain .reshape(3,1,1) would broadcast WRONG once an ensemble axis makes
    the field 4-D: numpy left-pads, misaligning the component axis against S).
    Any other v is a field: unbatched (ensemble=False) it is passed through
    as given (assumed broadcastable); with ensemble=True (the caller's P
    carries S: P.ndim == 4 in 2D, 5 in 3D) a per-realization (3,*space) field
    gets S inserted, a (3,S,*space) one is kept, anything else raises
    (ensemble_field). Was solver._reshape_vec3 / bulk3d._reshape_vec3_3d."""
    v = np.asarray(v, dtype=float)
    if v.shape == (3,):
        return v.reshape((3,) + (1,) * (ndim - 1))
    if ensemble:
        return ensemble_field(v, 1, ndim, "E_ext")
    return v


def vec3_field_checked(v, shape):
    """The film engine's E_ext normaliser -- a STRICTER policy than
    vec3_field, on purpose: the film has no ensemble axis and validates a
    field-shaped E_ext up front. A (3,) vector becomes (3,1,...,1) against a
    field of `shape` ((3,Nz,Ny,Nx) -> (3,1,1,1)); any other v must be
    component-first with the same ndim as `shape` and every axis 1 or equal
    to it; the result must be finite. Refusals raise ValueError. Was inline in
    film3d.solve_poisson_factored."""
    Eext = np.asarray(v, dtype=float)
    if Eext.shape == (3,):
        Eext = Eext.reshape((3,) + (1,) * (len(shape) - 1))
    elif (Eext.ndim != len(shape) or Eext.shape[0] != 3
          or any(a not in (1, b) for a, b in zip(Eext.shape[1:], shape[1:]))):
        raise ValueError("E_ext must be (3,) or a component-first field broadcastable to P")
    if not np.all(np.isfinite(Eext)):
        raise ValueError("E_ext must be finite")
    return Eext
