"""SPEC WP15 (R3) SS21: PMN heterovalent chemical-disorder statistics.

pzt_field.py lineage (SPEC S21 file-domain restriction: new files + this
lineage only, no touching dynsus.py/solver.py/film*/wp13_*/wp11c_*/g17*).
Same discipline as WP9 (WP9_REPORT.md): every field amplitude and
correlation length below is DERIVED from charge statistics + a screening
length, never tuned; every literature input carries a citation; genuinely
unresolved provenance gaps are flagged for coordinator ruling, not
self-decided.

======================================================================
PHYSICS: two independent quenched random-field LAYERS, both consumed as
a field-valued E_ext (reusing solver.py's WP9-generalized elec_energy_density
path -- pzt_field.defect_field's own precedent, just with DERIVED instead of
placeholder rms/xi).
======================================================================

Layer 1 -- random B-site substitutional charge disorder (SPEC S21.1 bullet 1):
  PMN = Pb(Mg1/3Nb2/3)O3. Each B-site independently Mg2+ (deviation from the
  +4 average = -2, probability 1/3) or Nb5+ (deviation +1, probability 2/3);
  1:2 stoichiometry is neutral on average (mean deviation = 0, asserted
  below -- this is the SPEC-stated sanity check, not assumed).
  Random point charges -> "shot noise" charge-density power spectrum
  (Campbell's theorem: S_rho(k) = n_B * <q^2>, white/k-independent for an
  uncorrelated point process) -> screened-Coulomb (Thomas-Fermi form,
  Green's function 1/(eps0*eps_r*(k^2+kappa^2)), kappa=1/lambda_s) ->
  E = -grad(phi) -> exact closed-form k-space integral for <|E|^2> (derived
  and verified below, see _layer1_k_integral's docstring). A finite
  UV cutoff kmax=pi/a_lattice is required (point charges are not literally
  a continuum below one lattice spacing) -- the exact same "floor at the
  lattice constant" philosophy pzt_field.formula_units_per_cell documents
  for PZT's x(r) generator, applied here in k-space instead of real space.

Layer 2 -- 1:1 chemically-ordered regions, COR (SPEC S21.1 bullet 2, UPDATED
  by SPEC S21.3's ruling of 2026-08-23 -- see "RULING" note below): a
  SECOND, independent quenched layer, geometry set by COR volume fraction
  f_cor and size xi_cor (Davies & Akbas 2000, below). TWO implementations
  coexist here, matching the ruling's "keep both, label the default":
    (i) bond_field_cor / pmn_clean_alpha1's COR term -- DEFAULT as of
        S21.3: a charge-NEUTRAL "random site" picture (Davies & Akbas's
        OWN conclusion), so COR carries zero net charge and contributes
        NOTHING to E_d; instead it is a BOND-type disorder -- a local
        Landau/T0 shift inside COR regions, amplitude BRACKETED (not
        derived -- there is no charge lever left to derive it from under
        this model), same f_cor/xi_cor geometry, reusing the existing
        spatial-Landau-disorder machinery (solver.bulk_energy_density
        already accepts array-valued alpha, exactly as PZT's ch1 does).
    (ii) derive_ed_stats_cor / defect_field_cor -- the ORIGINAL SPEC
        S21.1 "space charge" (charged-nanoregion) picture: net charge
        -1/2/site inside COR, Nb-rich compensating matrix (SPEC's
        original wording), screened exactly like Layer 1's kernel but
        against a finite-correlation-length source. KEPT, default OFF,
        as a labeled historical/sensitivity option (S21.3).

Both layers are derived in genuine 3D electrostatics (screening and
Poisson's equation are 3D phenomena); the resulting AMPLITUDE statistics
(std, correlation length) are then used directly as targets for the
existing 2D-slice field generator (disorder.random_field_gaussian), exactly
as pzt_field.defect_field already treats its (3,Ny,Nx) E_ext field as a
z=const slice of a conceptually 3D field -- not a new modeling choice, the
same one this lineage has used since WP9.

SI -> deck conversion for the field amplitude: f_elec's "-P.E_ext" term is
LINEAR in P (P unscaled SI<->deck, per units.py's own general argument for
si_to_deck_landau: "every term's coefficient rescales by the SAME 1e-9
regardless of n [the power of P it multiplies]"). E_ext plays the role of
that coefficient here (power 1), so units.si_to_deck_landau applies
directly and is reused UNCHANGED (units.py is a shared foundational module
outside this WP's file domain -- not edited, only called, same as
pzt_field.py already does for Q/alpha/C). NOTE: this does NOT require
resolving the separate kappa_b(background permittivity)-in-deck-units
puzzle (WP9's own flagged, still-open question) -- E_ext is an independent
ADDITIVE term in f_elec (SPEC S1.3), never routed through kappa/E_d at all.

Vector-vs-component RMS: the k-space integral below yields <|E_vector|^2>
(isotropic sum over all 3 Cartesian components, since |E(k)|^2 = k^2|phi(k)|^2
already sums the full gradient). By isotropy the 3 components carry equal
variance, so the PER-COMPONENT rms fed to disorder.random_field_gaussian
(which independently seeds each of the 3 stacked components, exactly like
pzt_field.defect_field) is std(|E|)/sqrt(3) -- an explicit, derived factor,
not an unexplained one.

HONESTY NOTE (Layer 2 model tension, RESOLVED by SPEC S21.3 2026-08-23):
SPEC S21.1 originally specified the classic "space charge" picture (net
charge inside COR, Nb-rich compensating matrix). The SAME paper SPEC cites
for the COR volume-fraction/size numbers (Davies & Akbas, J. Phys. Chem.
Solids 61 (2000) 159 -- read directly, see below) is the primary source
that REFUTES the space-charge picture in favour of a charge-BALANCED
"random site" model (one B-sublattice 100% Nb, the other a random 2:1
Mg:Nb mixture -- zero net charge by construction). This was surfaced, not
silently resolved, in WP15_REPORT.md's first submission; the coordinator's
ruling (SPEC S21.3) adopted the random-site picture as the new default --
"the agent was right to surface it" -- reflected below: Layer 2's E_d
contribution now defaults to ZERO, COR re-enters as a bond-type disorder,
and the original space-charge code is KEPT as a labeled, default-OFF
sensitivity option (not deleted -- SPEC's explicit instruction).

RULING 2 (pure-PMN clean-limit potential, SPEC S21.3, hybrid option a'):
pmn_clean_alpha1(T) below implements alpha1(T) from PMN's OWN measured
high-temperature Curie-Weiss fit (Viehland, Jang, Cross & Wuttig, Phys.
Rev. B 46, 8003 (1992), cross-checked against 2 further independent
sources -- see that function's docstring), with higher-order terms
bracketed (Zhang et al. 2017's numbers remain unreachable after a second,
independent attempt via arXiv/institutional-repository/companion-paper
routes -- see WP15_REPORT.md).

SPEC S36 (WP25 Phase 2, stage 2B, 2026-08-30): derive_ed_stats_substitutional/
_cor, defect_field_substitutional/_cor and bond_field_cor_alpha1 (Layers 1/2
and the default COR bond-order model above) are now thin PARAMETER-BINDING
shims onto solvers/quenched_fields.py's generalized mechanism-layer functions
(charge_disorder_stats/_field, region_charge_disorder_stats/_field,
bond_order_alpha1_field) -- every chemistry number that was a module constant
here (P_MG/DQ_MG/P_NB/DQ_NB via VAR_DQ, DQ_COR, lambda_s/eps_r/f_cor/xi_cor/
dT0_cor/T0/C_cw) is now passed THROUGH explicitly as an argument to a material-
agnostic general function; PMN's own numbers are ALSO recorded independently
as quenched_fields.PRESET_PMN (SPEC S36.2 R3: "a named preset, not a hard-
coded default"). The module constants below (P_MG/DQ_MG/P_NB/DQ_NB/MEAN_DQ/
VAR_DQ/DQ_COR and every provenance/bracket constant) are UNCHANGED and remain
this function's own derivation record AND the values several gates/cases read
directly (e.g. gates/g18a_pmn_field_statistics.py's pf.F_COR_DEFAULT/
XI_COR_DEFAULT_NM/DT0_COR_DEFAULT_K, gates/g19a_clean_control.py's pf.
T0_PMN_CW_K/C_CW_PMN_K/EPS0_SI) -- SPEC S35 ruling (3), never deleted. The
now-redundant PRIVATE computational helpers that duplicated what
quenched_fields.py's own k-space-integral machinery does (_layer1_k_integral,
_cor_k_integral, cor_matrix_charge, _cor_source_std_si, and their
_verify_*_integral self-checks) were deleted here, not left as dead code --
same "one implementation, not two live copies" precedent as
reports/WP25_2A_REPORT.md's own deletion of
literature_params._bto_298_mechanics; the self-checks now run at
quenched_fields.py's own import time instead (still transitively exercised
every time this module is imported, since it imports quenched_fields).
Bit-for-bit equivalence: gates/g29c_quenched_field_equivalence.py.
pmn_clean_alpha1 (stage 2A, unaffected by this stage) and every module
constant below remain otherwise UNCHANGED. pmn_pt_landau_placeholder, the
other stage-2A shim, had no caller and was archived 2026-11 to
solvers/materials_archive.py.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fftpf import mat_registry as materials  # SPEC S36 stage 2A -- pmn_clean_alpha1 shim
from fftpf import random_field_mechanisms as quenched_fields  # SPEC S36 stage 2B -- mechanism-layer shim, see module docstring below

# ==========================================================================
# Charge model (SPEC S21.1 bullet 1): Mg2+/Nb5+ B-site valence deviation
# from the +4 perovskite-B-site average, 1:2 Mg:Nb stoichiometry.
# ==========================================================================
P_MG, DQ_MG = 1.0 / 3.0, -2.0   # Mg2+: probability 1/3, deviation -2e
P_NB, DQ_NB = 2.0 / 3.0, 1.0    # Nb5+: probability 2/3, deviation +1e
MEAN_DQ = P_MG * DQ_MG + P_NB * DQ_NB          # = 0.0 -- SPEC's own stated check
VAR_DQ = P_MG * DQ_MG**2 + P_NB * DQ_NB**2      # = 2.0 (units of e^2)
assert abs(MEAN_DQ) < 1e-12, f"PMN B-site charge model not neutral on average: mean={MEAN_DQ}"
assert abs(VAR_DQ - 2.0) < 1e-12

DQ_COR = 0.5 * DQ_MG + 0.5 * DQ_NB   # = -0.5, SPEC's own stated "net -1/2 per B-site"
assert abs(DQ_COR - (-0.5)) < 1e-12

E_CHARGE_SI = 1.602176634e-19   # C, exact (2019 SI redefinition)
EPS0_SI = 8.8541878128e-12      # F/m, CODATA

# --------------------------------------------------------------------------
# Provenance constants
# --------------------------------------------------------------------------
# PMN pseudocubic perovskite lattice constant at room temperature -- standard
# crystallographic value (same status as pzt_field.A_LATTICE_PZT_NM: basic
# structural crystallography, not a disputed literature number).
A_LATTICE_PMN_NM = 0.4049  # nm

# lambda_s (Thomas-Fermi/Debye-type screening length, SPEC S21.1): the
# theoretical basis for a charge-disorder-driven quenched random field in
# PMN is Westphal, Kleemann & Glinchuk, Phys. Rev. Lett. 68, 847 (1992)
# ("Diffuse phase transitions and random-field-induced domain states of the
# 'relaxor' ferroelectric PbMg1/3Nb2/3O3"), made quantitative by Glinchuk &
# Farhi, J. Phys.: Condens. Matter 8, 6985 (1996) ("A random field theory
# based model for ferroelectric relaxors"). Genuine research (WebSearch/
# WebFetch, multiple queries) did not surface an explicit PMN screening-
# length NUMBER from either paper (both paywalled beyond the abstract) --
# and, tellingly, even a follow-on COMPUTATIONAL implementation of this
# exact theory (Wang, Yi & Xu, Int. J. Solids Struct. 2016, phase-field
# model citing Glinchuk-Farhi by name) treats the random field as a
# phenomenological Gaussian input (fitted variance), not a screening-length
# derivation -- confirming no ready-made number exists to transcribe.
# Per SPEC S21.1's own explicit fallback ("if a range, parameterize +
# sensitivity note"): PMN has no free carriers (it is a good insulator),
# so a literal metallic Thomas-Fermi length does not apply; the physically
# motivated range used here is anchored to the ONLY relevant, independently
# measured short-range structural scales in PMN itself -- the COR size
# (2-3 nm, Davies & Akbas 2000, below) and the polar-nanoregion correlation
# length from neutron/X-ray diffuse scattering (~1.5 nm at high T, growing
# to ~7 nm at 10 K). lambda_s is swept over this range; no single value is
# asserted as "the" answer.
LAMBDA_S_RANGE_NM = (1.0, 5.0)
LAMBDA_S_DEFAULT_NM = 2.0

# eps_r (background/optical relative permittivity screening the LOCAL
# charge-charge interaction): the giant low-frequency relaxor permittivity
# (~20000-30000) is itself the soft-mode response we are trying to model
# the pinning of -- using it to screen the defects that help create the
# random field would be circular (same reasoning WP9's kappa_bracket used
# to reject importing BTO's 66x eps0 ruling for PZT's kappa_b, itself
# FLAGGED genuinely unresolved there). No PMN-specific "background-only"
# permittivity number was found either (same search difficulty as WP9's
# kappa_b). Bracketed, not asserted, per the same SPEC fallback as lambda_s.
#
# SUPERSEDED 2026-08-24 (SPEC S22, WP16/R4 coordinator ruling 2, on the
# WP16 pilot's own pre-flight finding -- see reports/WP16_PILOT_NOTE.md and
# cases/wp16_pmn_pilot.py's module docstring): the original (10,50,100)
# bracket/50-default is EMPIRICALLY EXCLUDED, not merely a weaker choice --
# PMN experimentally SHOWS Curie-Weiss behavior above the Burns temperature,
# so a parameterization that gives NO reachable Curie-Weiss window at ANY
# credible T (eps_r=50's field-saturated regime, confirmed both
# analytically and by direct short Langevin simulation in the WP16 pilot)
# is inconsistent with the data itself, independent of any fit quality
# argument. Physical anchor for the new scale: the host crystal's OWN
# dielectric response at the T range where screening acts is O(10^2-10^3)
# (relaxor permittivity magnitude at other-than-the-giant-soft-mode-peak
# conditions), making eps_r~10^3 the defensible screening scale (NOT the
# giant ~20000-30000 peak value the ORIGINAL comment above correctly warns
# against -- that circularity concern stands; 10^3 sits below it).
# eps_r_screen is now a bracketed PHYSICAL INPUT: default 1000, bracket
# [500,2000], sensitivity legs run at each edge in the WP16 R4 full run.
EPS_R_BRACKET = (500.0, 1000.0, 2000.0)
EPS_R_DEFAULT = 1000.0

# COR volume fraction and size (SPEC S21.1 bullet 2, "Davies/Akbas
# lineage"): Davies, P.K. & Akbas, M.A., "Chemical order in PMN-related
# relaxors: structure, stability, modification, and impact on properties,"
# J. Phys. Chem. Solids 61 (2000) 159-166 -- read directly (PDF fetched and
# read page-by-page). p.160: "The microstructures of the as sintered forms
# (1225C, 1h) of all these ceramics were similar to all PMN-type relaxors
# and contained ~20 vol% of 2-3 nm 1:1 ordered domains dispersed in a
# disordered perovskite matrix." p.161: pure PMN resists further
# modification regardless of anneal (unlike PMT), attributed to PMN's own
# 1:1 order-disorder transition lying below ~900C (extrapolated <800C in
# the paper's own Fig. 4) -- domains are a KINETICALLY QUENCHED remnant,
# consistent with SPEC's "quenched random field" framing.
F_COR_DEFAULT = 0.20
XI_COR_DEFAULT_NM = 2.5   # midpoint of the paper's stated 2-3 nm

# ==========================================================================
# Layer 1: random substitutional charge disorder -> screened E_d(r)
#
# SHIM (SPEC S36 stage 2B, module docstring): calls
# quenched_fields.charge_disorder_stats/_field with THIS module's own
# already-verified charge-model constants (P_MG/DQ_MG/DQ_NB, giving
# conc=P_MG, delta_q=DQ_MG-DQ_NB=-3 -- reproduces VAR_DQ=2.0 exactly). The
# k-space-integral machinery (formerly _layer1_k_integral/
# _verify_layer1_integral here) now lives only in quenched_fields.py.
# ==========================================================================
def derive_ed_stats_substitutional(lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT,
                                    a_lattice_nm=A_LATTICE_PMN_NM):
    """DERIVE (charge statistics -> screening -> field), SPEC S21.1: returns
    (std_vector_SI, std_component_SI, xi_ed_nm). All in SI (V/m); caller
    converts to deck units (see defect_field_substitutional). Full physics
    chain documented at quenched_fields.charge_disorder_stats (this
    function's shim target) -- unchanged since the original submission."""
    return quenched_fields.charge_disorder_stats(
        conc=P_MG, delta_q=(DQ_MG - DQ_NB),
        lambda_s_nm=lambda_s_nm, eps_r=eps_r, a_lattice_nm=a_lattice_nm)


def defect_field_substitutional(grid, lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT,
                                 seed=0, a_lattice_nm=A_LATTICE_PMN_NM):
    """Layer 1 E_d(r), DECK units, shape (3,Ny,Nx) -- ready for solver's
    E_ext (same field-valued-E_ext path WP9 generalized for pzt_field.
    defect_field). Each of the 3 Cartesian components independently seeded
    (seed+i), matching defect_field's own convention exactly."""
    return quenched_fields.charge_disorder_field(
        grid, conc=P_MG, delta_q=(DQ_MG - DQ_NB),
        lambda_s_nm=lambda_s_nm, eps_r=eps_r, a_lattice_nm=a_lattice_nm, seed=seed)


# ==========================================================================
# Layer 2: chemically-ordered regions (COR) -> second screened E_d(r) layer
#
# SHIM (SPEC S36 stage 2B, module docstring): calls
# quenched_fields.region_charge_disorder_stats/_field with THIS module's
# own DQ_COR constant passed through explicitly as delta_q_region. The
# neutrality-enforcing matrix-charge algebra (formerly cor_matrix_charge/
# _cor_source_std_si here) and the k-space-integral machinery (formerly
# _cor_k_integral/_verify_cor_integral here) now live only in
# quenched_fields.py (SAME "one implementation, not two live copies"
# precedent as the Layer-1 shim above).
# ==========================================================================
def derive_ed_stats_cor(f_cor=F_COR_DEFAULT, xi_cor_nm=XI_COR_DEFAULT_NM,
                         lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT,
                         a_lattice_nm=A_LATTICE_PMN_NM):
    """SPEC S21.3 (2026-08-23): historical/sensitivity option, DEFAULT OFF --
    the original SPEC S21.1 "space charge" picture, superseded as the
    default by the charge-neutral random-site model (see
    bond_field_cor_alpha1 below). KEPT per the coordinator's explicit
    instruction, unchanged from the original submission -- not the code
    path any current gate exercises by default. Full physics chain
    documented at quenched_fields.region_charge_disorder_stats (this
    function's shim target)."""
    return quenched_fields.region_charge_disorder_stats(
        f_region=f_cor, xi_region_nm=xi_cor_nm, delta_q_region=DQ_COR,
        lambda_s_nm=lambda_s_nm, eps_r=eps_r, a_lattice_nm=a_lattice_nm)


def defect_field_cor(grid, f_cor=F_COR_DEFAULT, xi_cor_nm=XI_COR_DEFAULT_NM,
                      lambda_s_nm=LAMBDA_S_DEFAULT_NM, eps_r=EPS_R_DEFAULT,
                      seed=0, a_lattice_nm=A_LATTICE_PMN_NM):
    """SPEC S21.3: historical/sensitivity option, DEFAULT OFF -- see
    derive_ed_stats_cor's docstring. Layer 2 E_d(r), DECK units, shape
    (3,Ny,Nx) -- same conventions as defect_field_substitutional.
    Independent seed stream from Layer 1 (the two are added by the
    caller/case script to form the total quenched field, same
    "independently seeded layers" pattern as WP9's 3 channels)."""
    return quenched_fields.region_charge_disorder_field(
        grid, f_region=f_cor, xi_region_nm=xi_cor_nm, delta_q_region=DQ_COR,
        lambda_s_nm=lambda_s_nm, eps_r=eps_r, a_lattice_nm=a_lattice_nm, seed=seed)


# ==========================================================================
# SPEC S21.3 RULING 1 (2026-08-23, DEFAULT model): COR as a BOND-type
# disorder -- charge-neutral "random site" picture (Davies & Akbas's own
# conclusion), so E_d from COR is exactly zero; the quenched COR geometry
# instead shifts the LOCAL Landau T0 (a bulk/f_bulk effect, not an
# elec/f_ext effect). Reuses the SAME f_cor/xi_cor statistics as the
# space-charge option above (same COR geometry either way -- only the
# CONSEQUENCE of that geometry differs between the two models) and the
# SAME binary-mixture-variance identity as _cor_source_std_si (generalized
# from a charge step to a T0 step), via disorder.random_field_gaussian --
# the "existing random-bond machinery": solver.bulk_energy_density/
# bulk_force already accept array-valued alpha coefficients (exactly what
# PZT's ch1, x(r)->local alpha1 via T0(x), already relies on) -- no new
# solver-side plumbing needed, consistent with the SPEC S21 file-domain
# restriction (solver.py untouched, only its existing array-alpha support
# is exercised).
#
# Shift amplitude: BRACKETED, not derived (SPEC S21.3: "PARAMETERIZED with
# a stated bracket (like lambda_s) pending R4 sensitivity") -- under the
# charge-neutral model there is no charge-statistics lever left to derive
# an amplitude from; "how much does local 1:1 order shift the local
# transition tendency" is a genuinely open question with no PMN-specific
# number found. Bracketed SIGN-AGNOSTIC (does not presume COR raises or
# lowers the local T0 -- literature on related 1:1-ordered Pb-perovskites
# does not give a consistent sign either way for this specific comparison,
# so asserting one here would be an unexplained-factor-style overreach).
# ==========================================================================
DT0_COR_BRACKET_K = (-50.0, 50.0)  # K, local T0 shift inside a COR region (bracket, not derived)
DT0_COR_DEFAULT_K = 25.0


def bond_field_cor_alpha1(grid, T, f_cor=F_COR_DEFAULT, xi_cor_nm=XI_COR_DEFAULT_NM,
                           dT0_cor_K=DT0_COR_DEFAULT_K, T0_pmn_K=None, C_cw_K=None, seed=0):
    """DEFAULT COR model (SPEC S21.3 ruling 1): alpha1(r), DECK units, shape
    (Ny,Nx) -- a two-phase (COR/matrix) T0 shift field realized as a
    Gaussian-correlated continuous proxy, feeding solver.bulk_energy_
    density's array-alpha path instead of elec_energy_density's E_ext path.
    T0_pmn_K/C_cw_K default to this module's own T0_PMN_CW_K/C_CW_PMN_K
    Curie-Weiss constants (SPEC S21.3 ruling 2) if not given -- the two
    rulings share the same T0(PMN) baseline by construction, not by
    coincidence.

    SHIM (SPEC S36 stage 2B, module docstring): calls
    quenched_fields.bond_order_alpha1_field with T0_pmn_K/C_cw_K (resolved
    above) and this module's own f_cor/xi_cor_nm/dT0_cor_K passed through
    explicitly -- full physics chain documented at that function."""
    if T0_pmn_K is None:
        T0_pmn_K = T0_PMN_CW_K
    if C_cw_K is None:
        C_cw_K = C_CW_PMN_K
    return quenched_fields.bond_order_alpha1_field(
        grid, T=T, T0_K=T0_pmn_K, C_cw_K=C_cw_K,
        f_region=f_cor, xi_region_nm=xi_cor_nm, dT0_region_K=dT0_cor_K, seed=seed)


# ==========================================================================
# PMN-PT Landau backbone (SPEC S21.1 bullet 3): "PMN-PT single-domain
# literature set as a function of x_PT (source and transcribe with
# provenance, WP9-style)".
#
# SOURCED, NOT TRANSCRIBED -- documented, not silent (WP9's own precedent
# for kappa_b(PZT): "GENUINELY UNRESOLVED... FLAGGED"). Unlike PZT (Rabe/
# Ahn/Triscone App A gives one continuous polynomial-in-x fit spanning
# x in [0,1], WP9's basis for pzt_haun_x_landau_si), PMN-PT's literature
# does NOT offer an equivalent continuous function of x_PT:
#  - Hangbo Zhang et al., "Phase coexistence and Landau expansion
#    parameters for a 0.70Pb(Mg1/3Nb2/3)O3-0.30PbTiO3 single crystal,"
#    Phys. Rev. B 96, 054109 (2017) -- the best single-composition anchor
#    found (x_PT=0.30, near the MPB), citation confirmed by multiple
#    independent search hits (consistent title/journal/volume/year), but
#    the actual numeric alpha_i table sits behind the APS paywall --
#    genuine research (WebSearch + WebFetch on journals.aps.org and a
#    ResearchGate mirror) hit HTTP 403 both times; not fabricated here.
#    RETRIED per SPEC S21.3's explicit "try once more via arXiv/author
#    copy": no arXiv preprint exists for this paper (none of the authors'
#    other PMN-PT papers on arXiv match it either); Penn State's PURE
#    institutional-repository page for it (Wenwu Cao is PSU faculty) links
#    only the paywalled DOI, no deposited PDF; the companion follow-on
#    paper (same group, Phys. Rev. B 97, 054114 (2018)) that might have
#    restated the table is ALSO paywalled (403). Still unreachable --
#    bracketed below (pmn_clean_alpha1_higher_order_bracket), not guessed.
#  - Peng Lv, PhD dissertation (UCLA, Mechanical Engineering, 2018),
#    "Energetics of Domain Engineered Rhombohedral Ferroelectric Single
#    Crystals" -- read directly (PDF, Ch.3). Devoted an entire chapter to
#    building a 10th-order Landau-Devonshire potential for PIN-PMN-PT and
#    states explicitly (p.44): "Finding a consistent set of coefficients
#    for the energy function requires addressing LARGE UNCERTAINTIES in
#    the data reported in the literature... the composition has a degree
#    of uncertainty not reported, and the measured properties can lie
#    within a fairly broad range for the same reported composition" --
#    independent, primary-source confirmation (from a specialist thesis
#    doing exactly this task) that no clean, off-the-shelf composition-
#    continuous PMN-PT set exists to transcribe.
# This is a research-task outcome honestly reported, not the pure-PMN
# clean-limit gap (that one was a genuine physical-modeling question,
# RULED on by the coordinator -- SPEC S21.3, see pmn_clean_alpha1 below);
# this one has a clear resolution path (institutional journal access to
# Zhang et al. 2017) just not one reachable here. It was returned as an
# explicit REQUIRED/None placeholder (pmn_pt_landau_placeholder), matching
# the exact convention params/archive/pzt_Haun_x0.6.toml already uses for
# its own un-sourced G110/permittivity fields -- never a fabricated number.
# ARCHIVED 2026-11: nothing called pmn_pt_landau_placeholder, so it moved
# to solvers/materials_archive.py (same output) with the pmn_pt_placeholder
# registry entry it reshaped (snapshot: params/archive/pmn_pt_landau_x0.30.toml).
# ==========================================================================


# ==========================================================================
# SPEC S21.3 RULING 2 (2026-08-23, hybrid option a'): pure-PMN (x_PT=0)
# clean-limit potential. alpha1(T) from PMN's OWN MEASURED high-temperature
# Curie-Weiss fit (not a PMN-PT extrapolation -- SPEC's own stated
# rationale: "anchors the bare potential to PMN's own measured response").
#
# T0_PMN_CW_K / C_CW_PMN_K: VERIFIED against 3 independent, mutually
# consistent sources (genuine research, not a single unchecked hit):
#   1. PRIMARY: Viehland, D., Jang, S.J., Cross, L.E. & Wuttig, M.,
#      "Deviation from Curie-Weiss behavior in relaxor ferroelectrics,"
#      Phys. Rev. B 46, 8003 (1992) -- the classic, purpose-built
#      measurement of exactly this quantity for PMN (high-T linear
#      1/eps-vs-T regime, above the Burns temperature ~620K, where the
#      diffuse/relaxor deviation has not yet set in): C~1.2e5 K, T0~398K.
#   2. Al-Barakaty, Prosandeev, Wang, Dkhil & Bellaiche, "Finite-
#      temperature properties of the relaxor PbMg1/3Nb2/3O3 from atomistic
#      simulations," Phys. Rev. B 91, 214117 (2015) -- quotes C=1.25e5 K,
#      T0=396K for the same high-T fit (search-corroborated, not
#      independently re-derived here).
#   3. A third search-corroborated source gives C=1.27e5 K, T0=400K.
# All three cluster tightly (T0 within 4K, C within 6%); used as a range,
# T0/C's midpoints as defaults -- NOT a single unchecked number.
#
# Factor-of-2 note (SPEC S5's own warning: "never insert unexplained
# factors... say which you checked"): SPEC S21.3's relayed ruling wrote
# "alpha1(T) = (T-T0)/(eps0*C_CW)" (no factor of 2). This project's OWN
# ALREADY-VALIDATED Curie-Weiss-to-alpha1 convention (literature_params.
# pzt_haun_x_landau_si: a1=(T-T0)/(2*eps0*C0), same functional form used
# for BTO) INCLUDES a factor of 2 -- the standard Landau result
# a1=1/(2*eps0*chi) with chi=C/(T-T0) (paraelectric susceptibility from
# f=a1*P^2+..., dF/dP=2*a1*P=E). Implemented HERE WITH the factor of 2,
# for self-consistency with the SAME, G6a-gated convention already used
# elsewhere in this project rather than introducing a second, unverified
# one -- explicitly flagged (not silently chosen either way); trivial to
# drop the 2 if the coordinator meant the formula literally as relayed.
# ==========================================================================
T0_PMN_CW_K = 398.0                    # K, Viehland et al. 1992 (primary source)
T0_PMN_CW_RANGE_K = (396.0, 400.0)     # K, spread across all 3 sources
C_CW_PMN_K = 1.2e5                     # K, Viehland et al. 1992 (primary source)
C_CW_PMN_RANGE_K = (1.2e5, 1.3e5)      # K, spread across all 3 sources


def pmn_clean_alpha1(T, T0_K=T0_PMN_CW_K, C_cw_K=C_CW_PMN_K):
    """Pure-PMN clean-limit alpha1(T), DECK units -- SPEC S21.3 ruling 2,
    hybrid option a'. alpha1 = (T-T0)/(2*eps0*C_CW) (factor-of-2 note
    above). At T=298K (room temperature, well below T0~398K) this alpha1
    is NEGATIVE -- i.e. the bare/clean potential DOES cross zero and would
    support a homogeneous ferroelectric minimum below ~398K if nothing
    else acted on it. SPEC S21.3's own stated rationale: this bare
    would-be transition is deliberately left visible, not suppressed, so
    R4 can test whether the DERIVED random fields (pmn_field's Layers
    1/2/bond) are what destroys it in this engine -- "the RFIM question in
    its cleanest falsifiable form," not something to pre-decide here.

    SHIM (SPEC §36 stage 2A): formula moved to solvers.materials.
    pmn_clean_alpha1 (an isolated copy of EPS0_SI, documented there --
    materials.py must stay a dependency LEAF under this module, so it
    cannot import EPS0_SI back from here). T0_K/C_cw_K are always passed
    through explicitly from THIS module's own T0_PMN_CW_K/C_CW_PMN_K
    defaults (unchanged, still used by bond_field_cor_alpha1 below), so
    behavior is identical regardless of materials.py's own defaults."""
    return materials.pmn_clean_alpha1(T, T0_K, C_cw_K)


# Higher-order terms (alpha11 etc.): SPEC S21.3 -- "from the PMN-PT x_PT->0
# trend where transcribable, otherwise left as a bracketed sensitivity
# range." Zhang et al. 2017 remains unreachable after the retry (see the
# "PMN-PT Landau backbone" comment block above) -- nothing IS transcribable, so
# this is entirely the "otherwise" branch. alpha111/alpha112/alpha123
# (6th-order shape detail) are left None/REQUIRED outright -- too
# speculative to bracket meaningfully with zero composition-trend data
# points in hand. alpha11 (the LEADING correction, which sets whether the
# quartic term stabilizes or further destabilizes the P=0 state -- directly
# relevant to R4's falsifiability question) gets an explicit, honestly-wide,
# SIGN-AGNOSTIC bracket instead of None, anchored ONLY to the ORDER OF
# MAGNITUDE already validated elsewhere in THIS project for chemically
# related Pb-perovskites (NOT a PMN-specific derivation -- explicitly
# flagged as such): literature_params.pto_haun1987 gives alpha11=-7.3e-2,
# bto_lcc2005_8th gives -2.10e-1, pzt_haun_x(x=0.48) gives +5.8e-2 (all
# deck units, T=298K) -- i.e. |alpha11| ~ O(0.1) deck units with the SIGN
# not even consistent across already-validated, closely-related systems in
# this same project. Bracket set to span and pad that observed range.
ALPHA11_PMN_BRACKET = (-0.3, 0.3)  # deck units, order-of-magnitude ONLY -- see note above, not PMN-derived
ALPHA11_PMN_NOTE = ("Not sourced from PMN-PT composition data (Zhang et al. 2017 unreachable). "
                     "Order-of-magnitude bracket anchored to this project's OWN already-validated "
                     "PT/BTO/PZT alpha11 values (-0.073/-0.210/+0.058 deck units at 298K) -- a "
                     "dimensional sanity range, not a PMN derivation. alpha111/alpha112/alpha123 "
                     "left unset (None/REQUIRED): too speculative to bracket with zero PMN-PT "
                     "composition-trend data points available.")
