"""SPEC S36 (WP25 Phase 2, stage 2B) -- quenched_fields.py: the PHYSICS layer
for quenched random-field disorder MECHANISMS. Answers "what field does THIS
disorder mechanism make", given chemistry/geometry as plain numeric
arguments -- never "what field does PMN make" (that question is answered by
calling one of the functions below WITH PRESET_PMN's numbers; PMN is a named
preset, not a module, SPEC S36.2 R3's own stated goal).

LAYER BOUNDARY (R3, binding -- the two layers must not be blurred):
  solvers/disorder.py (UNTOUCHED by this stage) is the NUMERICAL layer:
    "generate a field with a given RMS and correlation length" --
    disorder.random_field_gaussian(grid, rms, xi, seed) neither knows nor
    cares whether rms/xi came from a screened-Coulomb derivation below or a
    bare declared dial.
  quenched_fields.py (THIS module) is the PHYSICS layer, one level up:
    "derive those RMS/xi statistics FROM CHEMISTRY" (charge concentration,
    charge mismatch, screening length, region volume fraction/size). Every
    DERIVED function below ultimately calls disorder.random_field_gaussian
    (via gaussian_random_field, immediately below) to realize the field once
    its own statistics have been derived -- it never duplicates disorder.py's
    own spectral-filter machinery.
This module never generates a field with a hand-picked rms/xi of its own
invention; disorder.py never sees a chemistry-shaped argument (conc,
delta_q, eps_r, ...). Confusing the two layers -- e.g. teaching disorder.py
about charge mismatch, or having this module re-implement the spectral
filter -- is exactly the anti-pattern SPEC S36.2 R3 forbids.

DERIVED vs DECLARED-DIAL (SPEC S26.0(a)/(b), made STRUCTURAL, not just a
documentation convention -- the function name itself says which regime it
is): every function below whose name starts with charge_disorder_/
region_charge_disorder_/bond_order_ DERIVES its target statistics from
chemistry/geometry arguments (S26.0(b) "derived-first where derivable").
gaussian_random_field is the one explicit exception: a DECLARED-DIAL entry
point -- sigma/xi_nm are whatever the caller hands it, dialed or already
derived upstream (S26.0(a) "calibration to experiment... must simply be
methodical" -- legitimate, just labeled). The W-lane's Wang-style E_rand
(a calibrated/dialed field) and every DERIVED function below (which calls
gaussian_random_field only AFTER computing its own target rms/xi from
physics) now share this ONE realization primitive -- SPEC S36.3's own
"E_rand and E_d finally sit in one module, the derived/dialed distinction
explicit in the API" instruction.

TWO PHYSICALLY DISTINCT MECHANISMS BELOW, TWO DIFFERENT ENGINE CHANNELS --
do not conflate them (this distinction is the SPEC S36.1 target-script
correction flagged in reports/WP25_2B_REPORT.md's "spec inconsistency"
section):
  (1) CHARGE-IMBALANCE disorder (charge_disorder_stats/_field,
      region_charge_disorder_stats/_field below): heterovalent point
      defects, or charged nanoregions, create a random ELECTRIC FIELD
      E_d(r) via screened electrostatics (Poisson's equation). This is a
      FIELD-VALUED E_ext (WP9 SPEC S15.1 ch3, "generalize E_ext to a
      field"; solver.py's own _reshape_vec3; G8c's consistency anchor) --
      consumed as `solver.relax(..., E_ext=E_d)` / `solver.total_energy(...,
      E_ext=E_d, ...)`. NEVER used to modify alpha/Landau coefficients.
  (2) BOND-ORDER / chemically-ordered-region disorder
      (bond_order_alpha1_field below): a region geometry that carries NO
      net charge (by construction -- the charge-neutral "random site"
      picture, SPEC S21.3) instead shifts the LOCAL Curie-Weiss T0, hence
      the local Landau alpha1. This is an ARRAY-VALUED LANDAU COEFFICIENT
      (deck["landau"]["a1"] becoming a (Ny,Nx) array) -- consumed via
      solver.bulk_force/bulk_energy_density's existing array-alpha support
      (exactly what PZT's ch1 random-bond disorder already relies on).
      NEVER fed to E_ext.
  The two mechanisms may share the SAME region-geometry parameters
  (f_region, xi_region_nm -- e.g. PMN's own 2-3nm chemically-ordered
  regions, Davies & Akbas 2000) while producing DIFFERENT physical
  consequences and feeding DIFFERENT engine arguments. Writing one function
  that takes a Landau deck AND charge-mismatch-shaped arguments (conc,
  delta_q, lambda_s, eps_r) and returns something bound to "alpha" -- as an
  early SPEC S36.1 draft line did -- is physically incoherent: those
  parameters derive an E-FIELD, not a Landau-coefficient shift. See the
  report for the corrected S36.1 script line.

ABSORBS (SPEC S36.3, logic migrated + GENERALIZED -- chemistry/geometry that
was a hard-coded module constant becomes a plain argument -- not
copy-pasted; the old modules become thin parameter-binding shims, SPEC
S36.2 R4):
  - solvers/pmn_field.py: derive_ed_stats_substitutional/_cor,
    defect_field_substitutional/_cor, bond_field_cor_alpha1. PMN's own
    numbers (Mg2+/Nb5+ charge statistics, screening length, COR geometry,
    Curie-Weiss constants) are preserved below as PRESET_PMN -- a named
    preset, NOT a function default (SPEC S36.2 R3's explicit instruction:
    "PMN's values moved into a named preset, not hard-coded defaults").
    pmn_field.py's own module-level derivation constants (P_MG/DQ_MG/P_NB/
    DQ_NB/MEAN_DQ/VAR_DQ/DQ_COR and their import-time sanity asserts) are
    UNCHANGED and remain the authoritative provenance record for PRESET_PMN's
    numbers; the now-redundant PRIVATE computational helpers that duplicated
    what this module does (_layer1_k_integral, _cor_k_integral,
    cor_matrix_charge, their _verify_* self-checks) were deleted from
    pmn_field.py, not left as dead orphans -- same "one implementation, not
    two live copies" precedent as WP25_2A_REPORT.md's own deletion of
    literature_params._bto_298_mechanics.
  - solvers/pzt_field.py: defect_field (a fully material-agnostic
    declared-dial stack-of-3 field -- it never depended on any PZT-specific
    number in the first place, so it belongs in the mechanism layer, not
    the composition/geometry layer -- see composition.py's own docstring
    for why it stayed OUT of that module).

EQUIVALENCE: gates/g29c_quenched_field_equivalence.py.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy import integrate

from fftpf import units
from fftpf import disorder_fields as disorder

# --------------------------------------------------------------------------
# Shared physical constants (SI). Deliberately independent literals, not
# imported from pmn_field.py/solvers.materials -- the SAME "isolated
# duplicate of a CODATA constant" discipline solvers/materials.py's own
# _EPS0_SI_CODATA already documents (WP25_2A_REPORT.md): pmn_field.py's
# shims import THIS module (old depends on new), so the reverse import
# would be circular; both copies are CODATA constants, not expected to ever
# drift.
# --------------------------------------------------------------------------
E_CHARGE_SI = 1.602176634e-19   # C, exact (2019 SI redefinition)
EPS0_SI = 8.8541878128e-12      # F/m, CODATA


# ==========================================================================
# Declared-dial primitive (SPEC S26.0(a)/(b) made structural -- module
# docstring). Every DERIVED function below calls this ONCE it has computed
# its own target rms/xi; a caller wanting a bare, undirected, calibrated
# field (W-lane E_rand) calls it directly.
# ==========================================================================
def gaussian_random_field(grid, sigma, xi_nm, seed):
    """Single-component (Ny,Nx) zero-mean, spectrally-correlated Gaussian
    random field, EXACT target RMS=sigma, correlation length xi_nm -- a
    thin, explicitly-labeled DECLARED-DIAL pass-through to
    disorder.random_field_gaussian (the NUMERICAL layer, module docstring
    R3 boundary). `sigma`/`xi_nm` are accepted as given, dialed or already
    derived by the caller -- this function performs no derivation of its
    own. Stack 3 independently-seeded calls (seed+i, i=0,1,2) for a
    vector-valued field, exactly as every _field function below does."""
    return disorder.random_field_gaussian(grid, sigma, xi_nm, seed)


def gaussian_random_field_3d(grid, sigma, xi_nm, seed):
    """SPEC V2-03 A6. 3D twin of gaussian_random_field: the same DECLARED-DIAL
    pass-through, now to disorder.random_field_gaussian_3d, on a
    bulk3d.make_grid_3d grid -- shape (Nz,Ny,Nx). Same layer boundary (module
    docstring R3): sigma/xi_nm are accepted as given, no derivation here.

    NO 3D VARIANCE RE-NORMALISATION, and this is a statement about BOTH
    layers, checked not assumed (SPEC V2-03 A6's own question):
      * the NUMERICAL layer rescales every realization to EXACTLY the target
        RMS, so the realized per-component std is `sigma` in 2D and 3D alike
        (disorder.random_field_gaussian_3d's docstring derives why the
        exp(-k^2 xi^2/4) power spectrum <-> exp(-r^2/xi^2) correlation pair
        is dimension-independent too);
      * the PHYSICS layer's k-integrals below (_shot_noise_k_integral,
        _correlated_source_k_integral) were ALWAYS the 3D SI integrals --
        d^3k/(2pi)^3 with n_B = 1/a_lattice^3 -- describing a real crystal,
        never the simulation grid. They do not know or care whether the field
        is later realized on a 2D plane or a 3D box.
    So the 3D field functions below are the SAME derivation feeding the SAME
    exact-RMS primitive, with only the realization shape changed; a 2D and a
    3D run of the same chemistry target the SAME sigma and the SAME xi."""
    return disorder.random_field_gaussian_3d(grid, sigma, xi_nm, seed)


# --------------------------------------------------------------------------
# Shared statistics primitive: standard deviation of a two-level
# ("Bernoulli"/two-point-distribution) field -- value=step with probability
# f, value=0 with probability (1-f). The SAME identity appears three times
# below (substitutional two-species shot noise, charge-neutral region/
# matrix compensation, bond-order T0 step) -- ONE shared primitive, not
# three independent re-derivations (this is itself part of what "physics
# layer, not three copies of the same algebra" means).
# --------------------------------------------------------------------------
def _bernoulli_mixture_std(f, step):
    return abs(step) * np.sqrt(f * (1.0 - f))


# ==========================================================================
# Screened-Coulomb k-space integral, WHITE-NOISE (uncorrelated point-charge/
# shot-noise) source. I = integral_0^kmax k^4/(k^2+kappa^2)^2 dk, exact
# closed form (hand-derived, verified against direct numerical quadrature
# at import time below). Relocated from pmn_field.py's _layer1_k_integral
# (SPEC S36 stage 2B) -- that helper already took only kmax/kappa, no
# PMN-specific numbers, so this is a pure move, not a generalization.
# Limits sanity-checked in the original derivation: kappa->0 (unscreened)
# gives I->kmax (variance set purely by the UV/lattice cutoff, independent
# of screening -- correct, an unscreened random field's variance is
# dominated by short-distance granularity); kmax/kappa->0 (kappa->infinity,
# extreme screening) gives I -> kmax^3/(3*kappa^2) -> 0 (perfect screening
# kills the field everywhere -- correct).
# ==========================================================================
def _shot_noise_k_integral(kmax, kappa):
    return kmax - 1.5 * kappa * np.arctan(kmax / kappa) + kappa**2 * kmax / (2.0 * (kmax**2 + kappa**2))


def _verify_shot_noise_k_integral():
    kmax, kappa = 7.85e9, 5.0e8  # representative SI wavenumbers (see charge_disorder_stats)
    closed_form = _shot_noise_k_integral(kmax, kappa)
    numeric, _ = integrate.quad(lambda k: k**4 / (k**2 + kappa**2)**2, 0.0, kmax)
    relerr = abs(closed_form - numeric) / numeric
    assert relerr < 1e-8, f"shot-noise k-integral closed form mismatches numeric quad: relerr={relerr:.2e}"


_verify_shot_noise_k_integral()


# ==========================================================================
# Screened-Coulomb k-space integral, CORRELATED (finite-size source, e.g. a
# charged nanoregion) source. I = integral_0^inf k^4*exp(-k^2*xi^2/4)/
# (k^2+kappa^2)^2 dk. No closed form (Gaussian x rational function) --
# evaluated by numeric quadrature via the SAME dimensionless substitution
# u=k*xi (raw SI wavenumbers, ~1e8-1e9 rad/m, are ~9 orders of magnitude
# away from the O(1) scale scipy.integrate.quad's infinite-interval
# transform assumes; verified during pmn_field.py's original development
# that the un-rescaled SI integrand returns a silently WRONG answer
# ~1e14x too small, no exception raised -- the substitution fixes the root
# cause). Relocated verbatim from pmn_field.py's _cor_k_integral (already
# fully general: xi_source_m/kappa are plain arguments, no PMN-specific
# numbers).
# ==========================================================================
def _correlated_source_k_integral(xi_source_m, kappa, u_bound=60.0):
    kappa_t = kappa * xi_source_m

    def integrand_u(u):
        return u**4 * np.exp(-u**2 / 4.0) / (u**2 + kappa_t**2)**2

    I_u, _ = integrate.quad(integrand_u, 0.0, u_bound, limit=200)
    return I_u / xi_source_m


def _verify_correlated_source_k_integral():
    xi_source_m, kappa = 2.5e-9, 5.0e8  # representative SI values (see region_charge_disorder_stats)
    I1 = _correlated_source_k_integral(xi_source_m, kappa, u_bound=60.0)
    I2 = _correlated_source_k_integral(xi_source_m, kappa, u_bound=120.0)
    relerr = abs(I1 - I2) / I2
    assert relerr < 1e-10, f"correlated-source k-integral not converged at u_bound=60: relerr={relerr:.2e}"
    assert I1 > 0.0


_verify_correlated_source_k_integral()


# ==========================================================================
# MECHANISM (1a): substitutional point-charge disorder -> screened random
# E-field E_d(r). Feeds the engine's E_ext channel (module docstring,
# mechanism (1)) -- NEVER alpha. Generalizes pmn_field.py's Layer 1 (SPEC
# S21.1 bullet 1).
# ==========================================================================
def charge_disorder_stats(conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm):
    """DERIVE (charge statistics -> screening -> field) substitutional
    point-charge E_d(r) statistics. Two co-existing species share each
    lattice site: fraction `conc` carries a charge deviation `delta_q`
    [units of e] relative to the other species (fraction 1-conc, deviation
    0 relative to the first -- `delta_q` is the MISMATCH between the two,
    not an absolute deviation from some third reference). Chain: shot-noise
    charge-density power spectrum (Campbell's theorem for an uncorrelated
    point process, S_rho0=e^2*Var[dq]*n_B, n_B=1/a_lattice^3 one site per
    formula unit) -> screened-Coulomb (Thomas-Fermi/Yukawa) transfer
    function, screening length lambda_s_nm, LOCAL relative permittivity
    eps_r -> closed-form <|E|^2> via _shot_noise_k_integral, UV cutoff
    kmax=pi/a_lattice_nm (one lattice spacing is the finest physically
    meaningful scale for a continuum "charge density"). Var[dq] =
    conc*(1-conc)*delta_q**2, the standard two-point/Bernoulli-mixture
    variance identity -- PRESET_PMN's own Mg2+(prob 1/3, deviation -2)/
    Nb5+(prob 2/3, deviation +1) pair is the conc=1/3, delta_q=-3 special
    case of this, reproducing PMN's historical shot-noise variance (2.0
    e^2) EXACTLY: (1/3)(2/3)(3^2)=2.0. The MEAN of the charge distribution
    never enters this derivation (Campbell's theorem depends only on the
    VARIANCE of the point-charge distribution -- PMN's own neutral-on-
    average Mg/Nb model is a special case of a more general statement that
    does not itself require neutrality).

    Returns (std_vector_SI, std_component_SI, xi_ed_nm). Correlation length
    of the OUTPUT field is the screening length itself lambda_s_nm (the
    ONLY length scale in a white-source + Yukawa-kernel problem). Vector-
    vs-component: std_component = std_vector/sqrt(3) (the isotropic k-space
    integral already sums all 3 Cartesian components; by isotropy each
    carries equal variance)."""
    var_dq = conc * (1.0 - conc) * delta_q**2
    a_m = a_lattice_nm * 1e-9
    lam_m = lambda_s_nm * 1e-9
    kappa = 1.0 / lam_m
    kmax = np.pi / a_m
    n_B = 1.0 / a_m**3
    S_rho0 = E_CHARGE_SI**2 * var_dq * n_B
    eps = EPS0_SI * eps_r
    I = _shot_noise_k_integral(kmax, kappa)
    E2 = S_rho0 / eps**2 * I / (2.0 * np.pi**2)
    std_vector = float(np.sqrt(E2))
    std_component = std_vector / np.sqrt(3.0)
    return std_vector, std_component, lambda_s_nm


def charge_disorder_field(grid, conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm, seed):
    """Substitutional-defect E_d(r), DECK units, shape (3,Ny,Nx) -- a
    field-valued E_ext (module docstring mechanism (1)), ready for
    solver.relax(..., E_ext=this)/solver.total_energy(..., E_ext=this,
    ...). Each Cartesian component independently seeded (seed+i)."""
    _, std_component_SI, xi_nm = charge_disorder_stats(conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm)
    # V2-02.1 U4: this is an E-FIELD, so it converts with si_to_deck_field.
    # BYTE-IDENTICAL re-point -- both helpers are the same x1e-9 factor
    # (units.py's V2-02.1 block); only the name was wrong.
    std_component_deck = units.si_to_deck_field(std_component_SI)
    return np.stack([gaussian_random_field(grid, std_component_deck, xi_nm, seed + i) for i in range(3)])


def charge_disorder_field_3d(grid, conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm, seed):
    """SPEC V2-03 A6. 3D twin of charge_disorder_field: IDENTICAL derivation
    (charge_disorder_stats, unchanged) and identical per-component seeding
    (seed+i), only the realization is 3D -- shape (3,Nz,Ny,Nx), DECK units,
    ready for the 3D engines' field-valued E_ext. See gaussian_random_field_3d
    for why no 3D variance re-normalisation is owed."""
    _, std_component_SI, xi_nm = charge_disorder_stats(conc, delta_q, lambda_s_nm, eps_r, a_lattice_nm)
    std_component_deck = units.si_to_deck_field(std_component_SI)
    return np.stack([gaussian_random_field_3d(grid, std_component_deck, xi_nm, seed + i) for i in range(3)])


# ==========================================================================
# MECHANISM (1b): charged-nanoregion disorder -> screened random E-field
# E_d(r), a SECOND independent charge-imbalance channel with its own finite
# correlation length. Feeds E_ext, same as (1a) -- NEVER alpha. Generalizes
# pmn_field.py's Layer 2 space-charge option (SPEC S21.1 bullet 2, kept
# default-OFF per SPEC S21.3's coordinator ruling as a labeled historical/
# sensitivity path).
# ==========================================================================
def _region_matrix_charge(f_region, delta_q_region):
    """Compensating-matrix charge deviation enforcing GLOBAL charge
    neutrality at fixed region/matrix volume split: f_region*delta_q_region
    + (1-f_region)*dq_matrix = 0. This is the genuine physical difference
    from charge_disorder_stats above: TWO independently-specified
    co-existing species there; here ONE specified region species plus a
    matrix charge DERIVED from neutrality, not independently specified."""
    return -f_region * delta_q_region / (1.0 - f_region)


def region_charge_disorder_stats(f_region, xi_region_nm, delta_q_region, lambda_s_nm, eps_r, a_lattice_nm):
    """DERIVE charged-nanoregion E_d(r) statistics. Regions of volume
    fraction f_region and correlation length xi_region_nm carry a net
    charge deviation delta_q_region [e]; the surrounding matrix's
    compensating charge is DERIVED from global neutrality
    (_region_matrix_charge), not independently specified (contrast
    charge_disorder_stats's two INDEPENDENTLY specified species). Same
    screened-Coulomb kernel as charge_disorder_stats, but the source now
    has its own finite correlation length xi_region_nm (a literature region
    size, not point-charge shot noise) -- no closed form for the resulting
    <|E|^2> (Gaussian x rational function), evaluated by numeric quadrature
    (_correlated_source_k_integral, an exact evaluation of an exact
    integral expression, not a fit). Output correlation length: quadrature
    combination sqrt(xi_region_nm^2+lambda_s_nm^2), the standard combination
    of a source width and a convolving screening kernel's width (reduces to
    xi_region_nm when xi_region_nm>>lambda_s_nm, to lambda_s_nm when the
    reverse). PRESET_PMN's f_region=0.20/xi_region_nm=2.5nm/
    delta_q_region=-0.5 reproduce pmn_field.py's original COR space-charge
    numbers (Davies & Akbas 2000 geometry, SPEC S21.1's stated -1/2/site)."""
    dq_matrix = _region_matrix_charge(f_region, delta_q_region)
    step = delta_q_region - dq_matrix
    std_q = _bernoulli_mixture_std(f_region, step)
    a_m = a_lattice_nm * 1e-9
    n_B = 1.0 / a_m**3
    std_rho = E_CHARGE_SI * n_B * std_q

    xi_region_m = xi_region_nm * 1e-9
    lam_m = lambda_s_nm * 1e-9
    kappa = 1.0 / lam_m
    eps = EPS0_SI * eps_r
    S0 = std_rho**2 * (np.sqrt(np.pi) * xi_region_m)**3  # S_rho(k) = S0 * exp(-k^2 xi_region_m^2/4)
    I = _correlated_source_k_integral(xi_region_m, kappa)
    E2 = S0 / eps**2 * I / (2.0 * np.pi**2)
    std_vector = float(np.sqrt(E2))
    std_component = std_vector / np.sqrt(3.0)
    xi_ed_nm = float(np.sqrt(xi_region_nm**2 + lambda_s_nm**2))
    return std_vector, std_component, xi_ed_nm


def region_charge_disorder_field(grid, f_region, xi_region_nm, delta_q_region, lambda_s_nm, eps_r, a_lattice_nm, seed):
    """Charged-nanoregion E_d(r), DECK units, shape (3,Ny,Nx) -- same
    conventions/consumption (field-valued E_ext) as charge_disorder_field."""
    _, std_component_SI, xi_nm = region_charge_disorder_stats(
        f_region, xi_region_nm, delta_q_region, lambda_s_nm, eps_r, a_lattice_nm)
    # V2-02.1 U4: this is an E-FIELD, so it converts with si_to_deck_field.
    # BYTE-IDENTICAL re-point -- both helpers are the same x1e-9 factor
    # (units.py's V2-02.1 block); only the name was wrong.
    std_component_deck = units.si_to_deck_field(std_component_SI)
    return np.stack([gaussian_random_field(grid, std_component_deck, xi_nm, seed + i) for i in range(3)])


def region_charge_disorder_field_3d(grid, f_region, xi_region_nm, delta_q_region,
                                    lambda_s_nm, eps_r, a_lattice_nm, seed):
    """SPEC V2-03 A6. 3D twin of region_charge_disorder_field -- same
    derivation (region_charge_disorder_stats, unchanged), same conventions,
    shape (3,Nz,Ny,Nx)."""
    _, std_component_SI, xi_nm = region_charge_disorder_stats(
        f_region, xi_region_nm, delta_q_region, lambda_s_nm, eps_r, a_lattice_nm)
    std_component_deck = units.si_to_deck_field(std_component_SI)
    return np.stack([gaussian_random_field_3d(grid, std_component_deck, xi_nm, seed + i) for i in range(3)])


# ==========================================================================
# MECHANISM (2): bond-order / chemically-ordered-region disorder -> local
# alpha1(r) shift. Feeds the engine's ARRAY-VALUED LANDAU coefficient
# channel (module docstring mechanism (2)) -- NEVER E_ext. Generalizes
# pmn_field.py's DEFAULT COR model (SPEC S21.3 ruling 1): a CHARGE-NEUTRAL
# region/matrix geometry (Davies & Akbas's own "random site" conclusion --
# zero E_d contribution, unlike region_charge_disorder_* above, which is
# the labeled alternative charged-nanoregion picture) instead shifts the
# LOCAL Curie-Weiss T0.
# ==========================================================================
def bond_order_alpha1_field(grid, T, T0_K, C_cw_K, f_region, xi_region_nm, dT0_region_K, seed):
    """DERIVE (region geometry -> T0(r) -> alpha1(r) via Curie-Weiss), DECK
    units, shape (Ny,Nx) -- a two-phase (region/matrix) T0-shift field
    realized as a Gaussian-correlated continuous proxy (gaussian_random_
    field; rms from the SAME Bernoulli-mixture identity as
    charge_disorder_stats/region_charge_disorder_stats above: rms_dT0 =
    |dT0_region_K|*sqrt(f_region*(1-f_region))), fed through the CLEAN
    Curie-Weiss map alpha1(T) = (T-T0)/(2*eps0*C_cw) (factor-of-2
    convention: this project's own already-validated Landau/susceptibility
    relation, e.g. solvers.materials.pmn_clean_alpha1/
    literature_params.pzt_haun_x_landau_si's identical a1=(T-T0)/
    (2*eps0*C0) form).

    T0_K/C_cw_K: the material's OWN clean-limit Curie-Weiss constants --
    REQUIRED arguments, no default (the entire point of this
    generalization: they are chemistry, not a hard-coded number; PRESET_PMN
    below carries T0_K=398.0/C_cw_K=1.2e5, Viehland et al. 1992).
    f_region/xi_region_nm: the SAME region-geometry parameters
    region_charge_disorder_stats uses -- may literally be the SAME
    nanoregions viewed through a DIFFERENT physical consequence (module
    docstring). dT0_region_K: the local T0-shift AMPLITUDE inside a region
    -- a bracketed/dialed number in every material this repo has used so
    far (there is no charge lever to derive it from under the
    charge-neutral model; PRESET_PMN's own dT0_region_K=25.0, bracket
    (-50,50) recorded in pmn_field.DT0_COR_BRACKET_K)."""
    rms_dT0 = _bernoulli_mixture_std(f_region, dT0_region_K)
    dT0_field = gaussian_random_field(grid, rms_dT0, xi_region_nm, seed)
    T0_field = T0_K + dT0_field
    a1_si = (T - T0_field) / (2.0 * EPS0_SI * C_cw_K)
    return units.si_to_deck_landau(a1_si)


def bond_order_alpha1_field_3d(grid, T, T0_K, C_cw_K, f_region, xi_region_nm,
                               dT0_region_K, seed):
    """SPEC V2-03 A6. 3D twin of bond_order_alpha1_field -- the SAME
    Bernoulli-mixture rms and the SAME Curie-Weiss map, realized on a 3D box;
    shape (Nz,Ny,Nx), DECK units, an array-valued alpha1 for the 3D engines'
    existing array-alpha support (solver.bulk_force/bulk_energy_density are
    elementwise and already accept a (Nz,Ny,Nx) alpha, bulk3d's own docstring
    "REUSED UNCHANGED" note)."""
    rms_dT0 = _bernoulli_mixture_std(f_region, dT0_region_K)
    dT0_field = gaussian_random_field_3d(grid, rms_dT0, xi_region_nm, seed)
    T0_field = T0_K + dT0_field
    a1_si = (T - T0_field) / (2.0 * EPS0_SI * C_cw_K)
    return units.si_to_deck_landau(a1_si)


# ==========================================================================
# PRESET_PMN -- SPEC S36.2 R3/S36.3: PMN's own numbers, as a NAMED PRESET,
# not hard-coded into the general functions above. Every value here is an
# INDEPENDENT literal (not imported from pmn_field.py) -- the same
# "isolated duplicate, dependency-leaf" discipline as solvers/materials.py's
# own _EPS0_SI_CODATA (WP25_2A_REPORT.md): pmn_field.py's shims import THIS
# module, so the reverse import would be circular; both copies are
# literature/CODATA constants, not expected to drift. pmn_field.py's own
# provenance essay (module docstring; P_MG/DQ_MG/P_NB/DQ_NB/MEAN_DQ/VAR_DQ/
# DQ_COR and their import-time asserts) remains the derivation record for
# every number restated here -- gates/g29c_quenched_field_equivalence.py
# cross-checks the two copies agree.
# ==========================================================================
PRESET_PMN = dict(
    # Mechanism (1a), charge_disorder_stats/_field: B-site Mg2+ (prob 1/3,
    # deviation -2e) / Nb5+ (prob 2/3, deviation +1e) -- conc=P_MG,
    # delta_q=DQ_MG-DQ_NB=-3 reproduces pmn_field.VAR_DQ=2.0 exactly:
    # (1/3)(2/3)(3^2)=2.0 (see gates/g29c's own independent check).
    conc_substitutional=1.0 / 3.0,
    delta_q_substitutional=-3.0,
    # Shared screening (mechanisms 1a and 1b): Westphal/Kleemann/Glinchuk
    # 1992 + Glinchuk/Farhi 1996 lineage; eps_r bracketed per the WP16/SPEC
    # S22 empirical-exclusion ruling (pmn_field.py module docstring).
    lambda_s_nm=2.0,
    eps_r=1000.0,
    a_lattice_nm=0.4049,
    # Mechanism (1b), region_charge_disorder_stats/_field (DEFAULT OFF,
    # SPEC S21.3 kept-as-sensitivity-option ruling): Davies & Akbas 2000
    # chemically-ordered-region geometry, SPEC S21.1's original "net
    # -1/2/site" space-charge picture.
    f_region=0.20,
    xi_region_nm=2.5,
    delta_q_region=-0.5,
    # Mechanism (2), bond_order_alpha1_field (DEFAULT model, SPEC S21.3
    # ruling 1): SAME f_region/xi_region_nm geometry as mechanism (1b)
    # above, viewed as charge-neutral bond order instead.
    dT0_region_K=25.0,
    # Clean-limit Curie-Weiss backbone (SPEC S21.3 ruling 2; Viehland, Jang,
    # Cross & Wuttig, Phys. Rev. B 46, 8003 (1992), cross-checked against 2
    # further independent sources -- see solvers.materials.pmn_clean_alpha1
    # and pmn_field.T0_PMN_CW_K/C_CW_PMN_K).
    T0_K=398.0,
    C_cw_K=1.2e5,
)
