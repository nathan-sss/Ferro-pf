"""SPEC S30 (WP22/E2 machinery leg): static composite phase-label field
layer. L(r) in {0=host, 1=filler}, generated with the EXISTING
solvers/disorder.py machinery (random_field_gaussian -- WP3's own
spectral-Gaussian-filter generator, correlation length xi_AN, reused
unchanged) then thresholded to an EXACT target volume fraction x_AN
(rank-and-cut construction, not a z-score cutoff which only hits x_AN on
AVERAGE across realizations -- SPEC S30.2 G24a "exact-count construction
permitted and preferred").

Label-dependent Landau coefficient FIELDS: alpha_i(r) = host_i*(1-L) +
filler_i*L for every one of solver.ALPHA_KEYS (plus any 8th-order
solver.ALPHA_KEYS_8TH key a phase provides), over the SAME
"array-valued alpha into solver.bulk_force/bulk_energy_density" contract
SPEC S1.1 already establishes ("alpha1 may be a spatial array (disorder /
dielectric regions)") and WP9's pzt_field.py already exercises in
production (ch1 random-bond disorder) -- this module does not touch
solver.py, it only BUILDS the array alpha dicts solver.py's existing
functions already accept.

Filler menu (Yang SI Table 2, reports/WP20_STAGE0_NOTE.md line 9, SPEC
S30 preamble): PE (non-polar) linear dielectric, alpha1=5.645e8/eps_r
SI (eps_r the free dial, "approximately 50 was sufficient" per Yang's
main text -- correctly attributed to the FILLER, SPEC S27.3(b)/S28.7's
own name-collision correction, NOT kappa_b), all higher-order terms
zero -- first-class, used in G24e's demo. FE (polar) filler: full 6th-
order Table 2 values, hook built, demo-deferred (SPEC S30.1) -- not
used in any gate/demo this WP, present so a future WP need not
re-derive it.

Elastic OFF, gradient SHARED (single G, not label-dependent, flagged),
L(r) STATIC (generated once, never evolves) -- SPEC S30.1 scope."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from fftpf import pf2d_engine as solver, disorder_fields as disorder, units, random_field_mechanisms as quenched_fields

ALPHA_KEYS = solver.ALPHA_KEYS  # ("a1","a11","a12","a111","a112","a123")

# --------------------------------------------------------------------------
# Yang SI Table 2 filler coefficients (SI units), reports/WP20_STAGE0_NOTE.md
# line 9 -- transcribed there from the SI PDF directly, reused verbatim here.
# --------------------------------------------------------------------------
def pe_filler_landau_si(eps_r):
    """PE (non-polar) filler: pure linear dielectric. alpha1=5.645e8/eps_r
    N.m^2/C^2 SI; all higher-order/off-axis terms zero (Table 2's own PE
    row -- a single coefficient, matching G23a's own "pure-quadratic deck"
    pattern exactly, deliberately reused so G24b's x_AN=1 check can call
    G23a's own analytics unmodified)."""
    return dict(a1=5.645e8 / eps_r, a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0)


def pe_filler_landau_si_corrected(eps_r):
    """B1 coordinator-review rider (SPEC S26.0(b), physics-derived, NOT
    tuning): pe_filler_landau_si's alpha1=5.645e8/eps_r is a literal
    transcription of Yang SI Table 2; for a PURE LINEAR DIELECTRIC (Table
    2's own PE row -- a11=a111=0, unchanged here) the standard Landau <->
    permittivity relation is alpha1=1/(2*eps0*eps_r) (the SAME functional
    form pmn_field.py's own Curie-Weiss alpha1(T)=(T-T0)/(2*eps0*C_cw)
    already uses in this repo). At eps_r=50 this gives 1.1294e9 SI --
    ~100x the printed 5.645e8/50=1.129e7, with the first four significant
    digits agreeing (5.645 vs 5.6467) -- consistent with a base-10-exponent
    misprint in the SI PDF, not a different physical model. Discovered when
    A-N's filler |P| grew to 3.26 C/m^2 (WP22/B1 Stage 1 FINDING,
    reports/B1_REPORT.md), far outside Yang's own printed 0.05-0.55 C/m^2
    colorbar range -- this function is the corrected replacement used by
    cases/b1_yang_an_corrected.py; pe_filler_landau_si above is kept
    UNCHANGED (as-printed record, also needed for the companion "as-printed
    (suspected misprint)" panel)."""
    return dict(a1=1.0 / (2.0 * quenched_fields.EPS0_SI * eps_r), a11=0.0, a12=0.0, a111=0.0, a112=0.0, a123=0.0)


# FE (polar) filler, Table 2's own row -- HOOK, not used in any gate/demo
# this WP (SPEC S30.1: "FE filler hook built but demo-deferred").
FE_FILLER_LANDAU_SI = dict(
    a1=-2.0875e8, a11=-2.5288e10, a12=1.259765625e10,
    a111=2.0498e12, a112=1.1440e12, a123=1.2550e12,
)


def si_to_deck_alpha(alpha_si):
    """{alpha_name: SI value} -> {alpha_name: deck value}, uniform units.si_to_deck_landau for every P-power."""
    return {k: units.si_to_deck_landau(v) for k, v in alpha_si.items()}


# --------------------------------------------------------------------------
# Label field L(r)
# --------------------------------------------------------------------------
def make_label_field(grid, x_AN, xi_AN, seed):
    """Static binary phase-label field, EXACT target volume fraction x_AN
    (SPEC S30.2 G24a). Continuous correlated field from
    disorder.random_field_gaussian (WP3, unmodified; correlation length
    xi_AN, grid units matching hx/hy i.e. nm at Delta=1nm); the n_filler =
    round(x_AN*N) cells with the LARGEST field value become label=1 --
    rank-and-cut, not a fixed threshold value, so the realized fraction is
    exact up to +-1 cell (<<0.5pp for any grid used here)."""
    field = disorder.random_field_gaussian(grid, rms=1.0, xi=xi_AN, seed=seed)
    Ny, Nx = grid["Ny"], grid["Nx"]
    N = Ny * Nx
    n_filler = int(round(x_AN * N))
    flat = field.ravel()
    L_flat = np.zeros(N)
    if 0 < n_filler < N:
        top_idx = np.argpartition(flat, -n_filler)[-n_filler:]
        L_flat[top_idx] = 1.0
    elif n_filler >= N:
        L_flat[:] = 1.0
    L = L_flat.reshape(Ny, Nx)
    return L, field


def _present_8th_keys(decks):
    """The 8th-order keys (solver.ALPHA_KEYS_8TH order) that at least one
    of `decks` provides. Empty for 6th-order-only inputs, which therefore
    get the same dict, built by the same arithmetic, as before the
    8th-order keys were carried. (Adding explicit zero fields would not
    change bulk_force's bits -- it skips the 8th-order groups when they are
    all 0.0 -- but it would change the dict's keys and cost four arrays.)"""
    return tuple(k for k in solver.ALPHA_KEYS_8TH if any(k in d for d in decks))


def alpha_field(L, host_alpha_deck, filler_alpha_deck):
    """alpha_i(r) = host_i*(1-L) + filler_i*L for every ALPHA_KEYS entry.
    The same blend covers each 8th-order key (a1111..a1123) that either
    phase provides; a phase lacking that key contributes 0.0
    (solver.bulk_force's own default for an absent 8th-order key). Without
    this an 8th-order deck (e.g. su_landis, a1112=5.74) silently lost those
    terms.
    At L==0 identically (x_AN=0), this is EXACTLY host_alpha_deck[k]*1.0 +
    filler_alpha_deck[k]*0.0 -- floating-point-exact (1-0=1.0, x*0.0=0.0
    for finite x), so G24b's bit-for-bit zero-leakage check holds by
    construction, not a special-cased branch."""
    one_minus_L = 1.0 - L
    out = {k: host_alpha_deck[k] * one_minus_L + filler_alpha_deck[k] * L for k in ALPHA_KEYS}
    for k in _present_8th_keys((host_alpha_deck, filler_alpha_deck)):
        out[k] = host_alpha_deck.get(k, 0.0) * one_minus_L + filler_alpha_deck.get(k, 0.0) * L
    return out


def half_slab_label(grid, axis=1):
    """Deterministic (non-random) step-function label field for G24d's
    single-interface analytic check: L=0 for the first half of the box
    along `axis` (1=x, matching grid['KX']/['Nx'] convention), L=1 for the
    second half. axis=1 (x) is the default, matching the closed form's own
    "layers along x" derivation below."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    L = np.zeros((Ny, Nx))
    if axis == 1:
        L[:, Nx // 2:] = 1.0
    else:
        L[Ny // 2:, :] = 1.0
    return L


def circular_label(grid, cx_nm, cy_nm, radius_nm):
    """Deterministic (non-random) SINGLE circular inclusion label field --
    SAME "deterministic, not random" spirit as half_slab_label above, built
    for the Fig.1-style single-inclusion comparison (Yang's own Fig.1b/c
    cartoon is ONE hand-placed inclusion, not the multi-blob ED Fig.2/3
    composite that composite.make_label_field targets). Real-space (x,y) in
    nm, cell-origin convention matching every other L(r)/imshow(extent=...)
    caller in this repo (x=i*hx, y=j*hy, i,j the array indices)."""
    Ny, Nx = grid["Ny"], grid["Nx"]
    x = np.arange(Nx) * grid["hx"]
    y = np.arange(Ny) * grid["hy"]
    X, Y = np.meshgrid(x, y)  # shape (Ny,Nx)
    return (((X - cx_nm) ** 2 + (Y - cy_nm) ** 2) <= radius_nm ** 2).astype(float)


# --------------------------------------------------------------------------
# SPEC S34.2 (B2-A): binary -> K-ary (ternary, K=3) generalization. PURE
# ADDITION -- new functions only, make_label_field/alpha_field above are
# UNTOUCHED (G24 gate zero regression, SPEC S34.2's own requirement).
#
# Placement rule (dossier iii states none -- UNDICTATED, so this is a
# declared dial, SPEC S26.0(a)): SAME correlated Gaussian field
# (disorder.random_field_gaussian) as the binary case, generalized from a
# single threshold (rank-and-cut into 2 exact-count blocks) to K-1
# thresholds (rank-and-cut into K exact-count contiguous blocks, ascending
# field value -> label 0,1,...,K-1). This is the K=2 rank-and-cut algorithm
# read as its own K-ary generalization, not a different construction:
# make_label_field_multi(grid, (1-x_AN, x_AN), xi_AN, seed) reproduces
# make_label_field(grid, x_AN, xi_AN, seed)'s own L bit-for-bit (label 0 <->
# L=0, label 1 <-> L=1), verified in gates/g26a_ternary_label_statistics.py.
# Still exact-count (Sum_k n_k = N cells exactly, matching target fractions
# up to +-1 cell rounding per block boundary) and still correlated-blob (the
# underlying continuous field is spatially correlated, so nearby cells tend
# to land in the same rank-block -- spatially clustered regions of each
# label, matching dossier (iii)'s own "regions" language and this WP's
# ~5-10nm nanoregion visual-calibration target for xi, SPEC S34.2).
# --------------------------------------------------------------------------
def make_label_field_multi(grid, fractions, xi, seed):
    """Static K-ary phase-label field, EXACT target fractions (generalizes
    make_label_field's own exact-count/rank-and-cut construction, SPEC
    S30.2 G24a's "preferred" choice, to K>2 labels). `fractions`: sequence
    of K target fractions, must sum to 1.0 (checked, tol 1e-9). Returns
    (labels, field): labels is (Ny,Nx) int array in {0,...,K-1}; field is
    the underlying continuous correlated field (same return-shape contract
    as make_label_field, for the same REPORTED-not-gated correlation-length
    sanity check, e.g. disorder.fit_correlation_length)."""
    fractions = list(fractions)
    K = len(fractions)
    total = sum(fractions)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fractions} (sum={total})")
    field = disorder.random_field_gaussian(grid, rms=1.0, xi=xi, seed=seed)
    Ny, Nx = grid["Ny"], grid["Nx"]
    N = Ny * Nx
    order = np.argsort(field.ravel())  # ascending field value
    counts = [int(round(f * N)) for f in fractions]
    # exact-count rounding can drift the total by a few cells (SPEC S30.2's
    # own "<<0.5pp for any grid used here" tolerance) -- absorb any residual
    # into the LAST block so Sum(counts) == N exactly, never dropping cells.
    counts[-1] = N - sum(counts[:-1])
    labels_flat = np.empty(N, dtype=np.int8)
    pos = 0
    for k, n_k in enumerate(counts):
        n_k = max(n_k, 0)
        labels_flat[order[pos:pos + n_k]] = k
        pos += n_k
    labels = labels_flat.reshape(Ny, Nx)
    return labels, field


def alpha_field_multi(labels, material_alpha_decks):
    """alpha_i(r) = material_alpha_decks[labels[r]][i] for every ALPHA_KEYS
    entry -- WHOLE-CELL assignment (dossier iii: "each cell is assigned
    wholesale to one fixed end-member identity, not a per-cell blend"), NOT
    alpha_field's own affine host*(1-L)+filler*L blend (that formula was
    designed for a continuous L in [0,1]; here L is categorical). Exact at
    EVERY cell by construction (np.where copies the source scalar/array
    verbatim under a boolean mask, no arithmetic blur at any boundary), a
    strictly stronger zero-leakage guarantee than alpha_field's own
    at-the-endpoints-only exactness -- verified for the K=1-pure-label case
    in gates/g26a_ternary_label_statistics.py (G24b precedent).
    8th-order keys (a1111..a1123): built the same way for each one that at
    least one material provides; a material lacking it gets 0.0 there
    (see alpha_field). Absent from every material -> absent from the output.
    A label outside 0..K-1 (no deck for it) raises ValueError -- it used to
    get alpha = 0 at every key, a silent zero-Landau cell."""
    K = len(material_alpha_decks)
    unknown = np.setdiff1d(np.unique(labels), np.arange(K))
    if unknown.size:
        raise ValueError(f"alpha_field_multi: labels {unknown.tolist()} have no deck "
                         f"(got {K} material_alpha_decks, labels must be 0..{K - 1})")
    out = {}
    for key in ALPHA_KEYS:
        field = np.zeros(labels.shape, dtype=float)
        for k in range(K):
            field = np.where(labels == k, material_alpha_decks[k][key], field)
        out[key] = field
    for key in _present_8th_keys(material_alpha_decks):
        field = np.zeros(labels.shape, dtype=float)
        for k in range(K):
            field = np.where(labels == k, material_alpha_decks[k].get(key, 0.0), field)
        out[key] = field
    return out


def layered_dielectric_closed_form(E_ext, f_filler, chi_host, chi_filler, kappa):
    """Independent closed-form derivation (SPEC S30.2 G24d), periodic
    bilayer (BOTH layers linear-dielectric: f_bulk=alpha1*P^2 only, chi=
    1/(2*alpha1)), uniform background kappa, no free charge, layers along
    x with filler volume fraction f_filler.

    1D Gauss's law (d D_x/dx = 0, no free charge) => D_x = D0, a SINGLE
    constant across BOTH layers (D continuity, no interface free charge --
    the periodic-engine analog of gates/g8a_layered_closed_form.py's own
    D-continuity argument, generalized from "one frozen-P0 layer + one
    passive layer" to "two self-consistently-linear layers", so g8a's own
    formula is not directly reusable, re-derived fresh here):
      D0 = kappa*E_host + chi_host*E_host = (kappa+chi_host)*E_host  (layer A)
      D0 = (kappa+chi_filler)*E_filler                                (layer B)
    Constraint: the periodic engine's own convention has <E_depol>(k=0)=0,
    so the CELL-AVERAGE field equals E_ext exactly:
      (1-f_filler)*E_host + f_filler*E_filler = E_ext
    Substituting E_host=D0/(kappa+chi_host), E_filler=D0/(kappa+chi_filler)
    and solving for D0 gives the closed form below."""
    denom = (1 - f_filler) / (kappa + chi_host) + f_filler / (kappa + chi_filler)
    D0 = E_ext / denom
    E_host = D0 / (kappa + chi_host)
    E_filler = D0 / (kappa + chi_filler)
    P_host = chi_host * E_host
    P_filler = chi_filler * E_filler
    sigma_bound = P_host - P_filler  # bound sheet charge at the host->filler interface (normal +x)
    return dict(D0=D0, E_host=E_host, E_filler=E_filler, P_host=P_host, P_filler=P_filler,
                sigma_bound=sigma_bound)


# --------------------------------------------------------------------------
# SPEC S34.5 (B2-C) deliverable 2: K-ary label field, 3D. PURE ADDITION --
# make_label_field_multi/alpha_field_multi above are UNTOUCHED.
#
# alpha_field_multi (above) needs NO 3D counterpart: it is already shape-
# agnostic (np.zeros(labels.shape)/np.where(labels==k,...) never reference
# Ny/Nx by name), so it is reused UNCHANGED on a (Nz,Ny,Nx) labels array --
# verified in gates/g27b. Only the label-field GENERATOR itself needs a new
# function, because make_label_field_multi calls disorder.random_field_
# gaussian, which is 2D-hardcoded (grid['KX']/['KY'], a (Ny,Nx) noise draw).
# --------------------------------------------------------------------------
def make_label_field_multi_3d(grid, fractions, xi, seed):
    """3D generalization of make_label_field_multi -- SAME rank-and-cut K-ary
    construction (exact target fractions, SPEC S30.2 G24a's own "preferred"
    choice) over a spectrally-correlated Gaussian field, now on a genuinely
    3D (Nz,Ny,Nx) grid. Self-contained: does NOT call disorder.
    random_field_gaussian (2D-hardcoded) -- duplicates its own small recipe
    (white noise -> exp(-k^2*xi^2/8) amplitude filter -> zero mean -> exact
    RMS=1.0) instead, the same "duplicate rather than reach into a shared
    file" discipline solver.py's own _lin_matrix docstring already states
    for an analogous situation. `grid`: any dict carrying Nz/Ny/Nx and
    hz/hy/hx (bulk3d.make_grid_3d's own keys) -- no import of bulk3d itself,
    keeping this module's existing zero solver-engine-file dependency
    beyond solver/disorder/units/quenched_fields."""
    fractions = list(fractions)
    K = len(fractions)
    total = sum(fractions)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fractions} (sum={total})")
    Nz, Ny, Nx = grid["Nz"], grid["Ny"], grid["Nx"]
    hz, hy, hx = grid["hz"], grid["hy"], grid["hx"]
    kx1d = 2 * np.pi * np.fft.fftfreq(Nx, d=hx)
    ky1d = 2 * np.pi * np.fft.fftfreq(Ny, d=hy)
    kz1d = 2 * np.pi * np.fft.fftfreq(Nz, d=hz)
    KZ, KY, KX = np.meshgrid(kz1d, ky1d, kx1d, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    amp_filter = np.exp(-k2 * xi**2 / 8.0)  # SAME filter form as disorder.random_field_gaussian
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((Nz, Ny, Nx))
    nk = np.fft.fftn(noise) * amp_filter
    nk[0, 0, 0] = 0.0  # zero mean
    field = np.fft.ifftn(nk).real
    field -= field.mean()  # exact zero mean (guards residual fp noise)
    std = field.std()
    if std > 0:
        field = field / std  # exact RMS=1.0 (disorder.random_field_gaussian's own fixed rms=1.0 call convention, SPEC S34.2 precedent)

    N = Nz * Ny * Nx
    order = np.argsort(field.ravel())  # ascending field value
    counts = [int(round(f * N)) for f in fractions]
    counts[-1] = N - sum(counts[:-1])  # absorb rounding residual into the LAST block (make_label_field_multi's own precedent)
    labels_flat = np.empty(N, dtype=np.int8)
    pos = 0
    for k, n_k in enumerate(counts):
        n_k = max(n_k, 0)
        labels_flat[order[pos:pos + n_k]] = k
        pos += n_k
    labels = labels_flat.reshape(Nz, Ny, Nx)
    return labels, field
