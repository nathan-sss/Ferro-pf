"""WP4 §9.2: SI -> deck-unit converter (the crux of the parameter library).

Extracted from FERRET_ROOT/runs/rhombo_walls/tools/gld1d.py's own header
comment: "单位: Ferret 约定 (能量密度 1e9 J/m^3, 长度 nm, 极化 C/m^2)" -- deck
energy density = 1e9 J/m^3, deck length = nm (1e-9 m), P UNSCALED (C/m^2,
same number in SI and deck). This is also the normalization gld1d.py relies
on when it multiplies its own (deck-unit) sigma_w by 1000 to report mJ/m^2
directly, with no other prefactor: deck energy [1e9 J/m^3] x deck length
[nm=1e-9 m] = SI energy/area [J/m^2] exactly (1e9 * 1e-9 = 1), so the *1000
is a pure J/m^2 -> mJ/m^2 unit-name conversion, nothing else -- this is the
"exact normalization" SPEC directs to extract from that source.

Landau coefficient a_n (n = number of P-pairs, e.g. a1 multiplies P^2, a11
multiplies P^4, ..., a1111 multiplies P^8): f_bulk = sum a_n * P^(2n) is an
ENERGY DENSITY. Since F_SI[J/m^3] = F_deck[deck units] * 1e9 and P is
unscaled, every term's coefficient rescales by the SAME 1e-9 regardless of
n (P's power does not change how the ENERGY DENSITY prefactor rescales).
Elastic stiffness C_ij [Pa] is exactly an energy density too (stress =
energy/strain-volume), and 1 deck energy-density unit (1e9 J/m^3) equals
exactly 1 GPa (1e9 Pa) -- so deck C_ij is numerically "C_ij in GPa" via the
SAME 1e-9 factor from Pa. Electrostriction Q_ij [m^4/C^2, dimensionless
strain per (C/m^2)^2] and gradient-energy-normalized ratios carry NO length
or energy-density dimension that changes between SI and deck (strain is
dimensionless, P is unscaled) -- SPEC §9.1(a) explicitly keeps these
unconverted for bto_LCC2005_8th, consistent with this derivation.

Gate G6a (converter round-trip) is the sole authority test: applying
si_to_deck_landau to the SAME formula gld1d.py's own alphas(T) function
implements (scaled back up to "SI" by /SI_TO_DECK, since gld1d.py already
outputs deck-unit numbers) must reproduce bto_298K.toml/bto_118K.toml to
<=1e-8 relative -- see gates/g6a_converter_roundtrip.py.
"""
SI_TO_DECK_ENERGY = 1e-9  # deck energy-density unit = 1e9 J/m^3


def si_to_deck_landau(value_si):
    """Any Landau coefficient a_n (multiplies P^(2n) in an energy-density
    term): deck = SI * 1e-9, independent of n (P unscaled SI<->deck)."""
    return value_si * SI_TO_DECK_ENERGY


def si_to_deck_elastic(value_si_pa):
    """Elastic stiffness C_ij [Pa] -> deck units (numerically GPa, since
    1 deck energy-density unit = 1e9 J/m^3 = 1 GPa): deck = SI_Pa * 1e-9.
    Separate function from si_to_deck_landau (same numeric factor, but a
    different physical quantity -- kept distinct for readability/call-site
    clarity, not because the factor differs)."""
    return value_si_pa * SI_TO_DECK_ENERGY


def si_to_deck_gradient(value_si):
    """SPEC S34.2 (B2-A): SI -> deck conversion for a gradient-energy coefficient
    G_ij (f_grad ~ (1/2)*G*(dP/dx)^2, e.g. Pan et al. 2019's G110/G11/G12/G44/G44').
    FIRST USE of this conversion in the repo (every gradient sector so far was
    either a G110=1.0 renormalized reference, SPEC S9.1(a)'s bto_298K.toml-
    inherited set, or an unsourced dial, e.g. nnbm_wang2026.toml's G110 -- SPEC
    S34.2's own docstring: "their own values -- a SOURCED gradient sector,
    first time in the B/W lanes").

    Derivation (explicit factors, same discipline as si_to_deck_kT's docstring):
    with x_SI = x_deck * 1e-9 (deck length unit = nm, this project's standing
    convention, units.py's own header), d/dx_SI = 1e9 * d/dx_deck, so
    (dP/dx_SI)^2 = 1e18 * (dP/dx_deck)^2 (P itself unscaled, si_to_deck_landau's
    own premise). f_grad_SI = (1/2)*G_SI*(dP/dx_SI)^2 = (1/2)*G_SI*1e18*
    (dP/dx_deck)^2 [J/m^3]; converting this ENERGY DENSITY to deck via the same
    SI_TO_DECK_ENERGY=1e-9 factor si_to_deck_landau uses gives the deck-space
    energy density (1/2)*(G_SI*1e18*1e-9)*(dP/dx_deck)^2 = (1/2)*(G_SI*1e9)*
    (dP/dx_deck)^2 -- i.e. the coefficient the engine must be fed is
    G_deck = G_SI * 1e9, NOT SI_TO_DECK_ENERGY (1e-9) -- the INVERSE-looking
    direction is the same "extra length power" phenomenon si_to_deck_kT's own
    docstring already flags for a different quantity (there: energy-per-length
    vs energy-density differ by one length power; here: a gradient coefficient
    carries an extra length^2 in the DENOMINATOR relative to a bare Landau
    term, which SI->deck rescales by 1e9 = (1e-9)^-1 relative to
    SI_TO_DECK_ENERGY alone). Sanity check (SPEC S34.2 B2-A, reports/
    B2A_REPORT.md): Pan et al. 2019's G110=7.04e-11 SI -> 0.0704 deck, the
    same O(1e-2..1) magnitude family as every other deck's own G110 (1.0,
    -0.664, ...), not the ~9-orders-of-magnitude-off number a plain
    SI_TO_DECK_ENERGY application would give -- a numerical plausibility
    check, not a proof, but the wrong-direction factor would have failed it
    obviously."""
    return value_si * 1e9


def cubic_stiffness_to_compliance(c11, c12, c44):
    """Inverse of cubic_compliance_to_stiffness -- for cubic symmetry the
    SAME functional form self-inverts (standard result, e.g. Nye, "Physical
    Properties of Crystals"): s11=(c11+c12)/D, s12=-c12/D,
    D=(c11-c12)(c11+2*c12), s44=1/c44 (Voigt notation). WP10 SPEC §16 needs
    this for the ebar-with-C(r) update (Eq A16-style <sigma>=0 volume
    average requires the COMPLIANCE of the reference tensor C0). c11/c12/c44
    may be scalars or arrays (pure elementwise arithmetic, no branching).
    Round-trip verified numerically at import time by elastic.py's
    _verify_contract_C_cubic (not asserted from memory alone)."""
    denom = (c11 - c12) * (c11 + 2 * c12)
    s11 = (c11 + c12) / denom
    s12 = -c12 / denom
    s44 = 1.0 / c44
    return s11, s12, s44


K_BOLTZMANN_SI = 1.380649e-23  # J/K, exact (2019 SI redefinition)


def kT_langevin(T_kelvin, hx_nm, hy_nm, hz_nm, k_B=K_BOLTZMANN_SI):
    """THE Langevin FDT coefficient -- ONE convention for all three engines
    (SPEC V2-04.0 "CONVENTION OF RECORD", user ruling D16, 2026-09-16):

        kT = k_B * T_kelvin * 1e18 / (hx_nm * hy_nm * hz_nm)

    WHY IT IS A DENSITY, NOT AN ENERGY. The Langevin steppers sample
    exp(-F_total/k_B*T) with F_total = sum_cells f*V_cell exactly as
    solver.total_energy / bulk3d.total_energy_3d / the film's energy define
    it. Written in a stepper's own mode variable that free energy is
    F_total = pref * sum_k sum_b (1/2)*chi_b(k)*|m_b(k)|^2 with chi an energy
    DENSITY per P^2 -- so the coefficient the OU noise variance multiplies it
    by must be an energy density too: k_B*T per CELL VOLUME. The per-mode
    stationary law is then, in the UNNORMALISED rfft mode variable,

        <|m_k|^2> = N_cells * kT / chi_k

    -- identical in form to the law film3d_langevin.mode_amplitudes has always
    stated (its 1/sqrt(Nx*Ny) is the Parseval factor of the NORMALISED mode
    and nothing else). solver.step_langevin and bulk3d.step_langevin_3d no
    longer divide their k-space noise by sqrt(N_cells); the film never did.

    Dz IN 2D -- A PHYSICAL PARAMETER, NOT A BOOKKEEPING ONE. solver.py's
    F_total integrates over x and y only, so it is an energy per unit
    z-length and an out-of-plane thickness has to be named: hz_nm is the
    FLUCTUATION-SLAB THICKNESS Dz. With this kT and no 1/sqrt(N), the 2D
    stepper's stationary law is exp(-F_total*Dz/k_B*T) -- the Boltzmann
    weight of a slab of thickness Dz -- so Dz IS the fluctuation volume of a
    2D cell, hx*hy*Dz, and every 2D fluctuation variance scales as 1/Dz:
    <|m|^2> ∝ 1/Dz. A 2D thermal result therefore DEPENDS on Dz, which is
    exactly why it must be written out at every call site instead of being
    inherited silently. The legacy surrogate Dz = hx (WP9's
    formula_units_per_cell choice, inherited by si_to_deck_kT) is a CHOICE,
    not a fact -- and it ties the slab thickness to the mesh, so a 2D
    resolution study must hold Dz fixed by hand (K5 `kt_convention` Leg B and
    K3 `langevin_cutoff_grid_invariance` both do, and print the value).
    ("Dz cancels", REVIEW_v2_03c_kt.md derivation 7, was a statement about
    the OLD convention's T_equivalent RATIO -- the wrong kT and the right one
    both carried 1/Dz -- never about the physics.)

    NOISE CUTOFF. The optional noise_cutoff_nm of the three OU builders is
    the companion knob: it zeroes the noise (never the decay) on every mode
    with |k| >= 2*pi/l_c, so the retained modes stay exactly Boltzmann at T
    while the excluded ones relax deterministically. With the box and l_c
    fixed the retained mode set is grid-independent once h < l_c/2, which is
    what makes a thermal observable converge under refinement.

    RELATION TO THE TWO OLD CONVERTERS. This function returns exactly what
    film3d_langevin.kT_deck_film returned and still returns (it now delegates
    here). It is NOT units.si_to_deck_kT and NOT bulk3d.kT_deck_3d: those two
    are correct unit conversions of k_B*T into F_total units, and were the
    wrong quantity in the FDT slot (D16; REVIEW_v2_03c_kt.md findings
    KT1/KT2). K5 `kt_convention` is the gate that holds this convention to an
    absolute SI temperature, on all three engines."""
    return k_B * T_kelvin * 1e18 / (hx_nm * hy_nm * hz_nm)


def si_to_deck_kT(T_kelvin, hx_nm, k_B=K_BOLTZMANN_SI):
    """Unit conversion of k_B*T into F_total units -- NOT the Langevin FDT
    coefficient (SPEC V2-04.0, D16). Use units.kT_langevin for that; passing
    this number as a stepper's `kT` simulates an effective temperature
    T_kelvin*hx*hy under the D16 noise convention (hx, hy in nm: the cell
    area, = V_cell/Dz with this function's Dz = hx -- e.g. 0.16*T at
    0.4 nm cells; REVIEW_v2_03c_kt.md's KT1 factor N_cells/V_cell predates
    D16's per-cell-volume noise). The derivation below is correct AS A UNIT
    CONVERSION and is unchanged.

    WP12 (SPEC §18) kT-in-deck-units derivation -- a genuinely NEW
    conversion, not a reuse of si_to_deck_landau/_elastic (those convert an
    ENERGY DENSITY [J/m^3]; k_B*T is a plain ENERGY [J], a different
    dimension, so it needs its own factor).

    The engine is 2D: F_total (solver.total_energy) = integral of f_density
    over x,y ONLY (grid['Volume']=Lx*Ly is an AREA, z never integrated) --
    i.e. every energy this engine computes is really an energy PER UNIT
    LENGTH IN z (SI: J/m; deck: [deck energy density] x [deck length]^2,
    since Lx,Ly are in nm), not a true energy. For the Boltzmann weight
    exp(-F/kT) used by the FDT (SPEC §18) to be meaningful, kT must be
    expressed in the SAME "per unit z-length" units as F -- there is no
    avoiding picking a z-thickness Dz to spread the true k_B*T[J] over
    (kT_line = k_B*T/Dz, units J/m, matching F_total's units).

    Dz choice: this project ALREADY faced the identical problem -- "the 2D
    engine has no independent out-of-plane grid spacing" -- for WP9's
    formula_units_per_cell (pzt_field.py), which ruled Dz = the IN-PLANE
    GRID SPACING h (hz defaults to hx there; "a documented WP9 choice, not
    a literature number"). Reused verbatim here for consistency: Dz = hx_nm
    (the caller's grid spacing), NOT an arbitrary fixed length. This is a
    MODELING CHOICE (flagged, not silent) -- see WP12_REPORT.md.

    Derivation (explicit factors, not a single opaque constant):
      kT_SI            = k_B * T_kelvin                      [J]
      Dz_m             = hx_nm * 1e-9                         [m]
      kT_per_length_SI = kT_SI / Dz_m                         [J/m]
      kT_deck          = kT_per_length_SI * 1e9               [deck F units]
    (1 deck F-unit = [deck energy density]x[deck length]^2 = 1e9 J/m^3 x
    (1e-9 m)^2 = 1e-9 J/m -- so SI[J/m] -> deck is x1e9, the INVERSE-looking
    direction of SI_TO_DECK_ENERGY because energy-per-length and
    energy-density differ by an area, not because of an inconsistency.)

    Gate G6a-style round-trip authority does not apply here (no independent
    pre-existing deck kT constant to reproduce); G16a-e (gates/) tested that
    this NUMBER, used as the OU/FDT stepper's kT, reproduced the correct
    mode-variance statistics -- which it did, against the stepper's own law
    and not against an SI temperature (REVIEW_v2_03c_kt.md, KT4). The
    absolute anchor is K5 `kt_convention`, and it holds kT_langevin."""
    kT_si_J = k_B * T_kelvin
    dz_m = hx_nm * 1e-9
    kT_per_length_si = kT_si_J / dz_m
    return kT_per_length_si * 1e9


THETA_LEN_SCALE = 1e-9  # WP18/S2 (SPEC S28.5 R2): theta's own deck length unit = nm, SAME as the
# engine's existing grid-length convention (Lx,Ly,hx,hy already in nm) -- not a new arbitrary
# constant, just this already-established unit reused for a NEW length-dimensioned field.


def si_to_deck_theta(value_si_m):
    """SPEC S28.1/S28.5 R2: theta is a LINEAR OXYGEN DISPLACEMENT (meters, Chen's own Rabe App A
    text), NOT an angle or an already-unscaled quantity like P (P stays C/m^2 numerically
    identical in SI and deck, si_to_deck_landau's whole premise). theta needs its OWN
    length rescaling before it can sit next to P in the same coupled (P,theta) engine at a
    sane numeric magnitude (raw SI theta ~1e-10-1e-11 m would otherwise be ~10 orders of
    magnitude smaller than P's own O(0.1-1) deck range).

    Choice (flagged, not silent, per the si_to_deck_kT precedent for "a genuinely new
    conversion needs a documented choice"): deck_theta = SI_theta / THETA_LEN_SCALE, i.e.
    theta expressed as a pure number in the engine's OWN existing nm length unit -- reusing
    infrastructure already there for grid spacing, not inventing a new lattice-constant-style
    scale. deck_theta ~0.01-0.1 for real STO tilt displacements (~1-10 pm), a sane O(1)-ish
    range alongside P's own O(0.1-1)."""
    return value_si_m / THETA_LEN_SCALE


def si_to_deck_theta_angle_deg(value_deg):
    """WP18 Stage 2 (SPEC S28.7 R2): a SECOND, per-deck theta convention --
    Haun's Ferroelectrics 99 Part IV/V (reports/LIT_haun_tilt.md) define
    theta as an ANGLE IN DEGREES, not si_to_deck_theta's linear-displacement
    meters (that convention stays as-is for params/sto_afd_rabeA.toml; theta
    is per-deck, not global -- R2 explicit). Degrees is already a sane O(1-10)
    numeric range for real tilt angles (Table II/Table I: ~1-7 deg), so --
    unlike meters, which needed rescaling by 9 orders of magnitude -- degree-
    mode theta is kept UNSCALED, identical in SI and deck, the SAME
    "unscaled field" convention si_to_deck_landau's own docstring already
    uses for P (C/m^2, same number both sides). This function is the
    identity; it exists only for API symmetry with si_to_deck_theta and to
    give the convention a documented name at the call site."""
    return value_deg


# Degree-mode Landau-type coefficients (beta1, beta11, beta12, t11, t12, t44
# in a degree-convention deck) need NO separate function: since theta itself
# is unscaled (above) exactly like P, there is no extra length-power factor
# to fold in -- si_to_deck_landau's own plain x1e-9 energy-density rescaling
# already applies directly, the same call every P-sector coefficient uses.
# (Contrast si_to_deck_theta_landau below, meters-mode ONLY, which needs the
# THETA_LEN_SCALE**theta_power correction precisely because THAT mode's
# theta is NOT unscaled.)
def si_to_deck_theta_landau(value_si, theta_power):
    """Companion rescaling for any Landau-type coefficient multiplying theta^theta_power
    (theta_power = the TOTAL power of theta in that term, e.g. 2 for beta1*theta^2, 4 for
    beta11*theta^4, 2 for t11*P^2*theta^2 -- P itself contributes no extra factor, unscaled).

    Derivation: a term coeff_SI * P_SI^(2n) * theta_SI^m [energy density, N/m^2] with
    P_SI=P_deck (unscaled) and theta_SI=theta_deck*THETA_LEN_SCALE (si_to_deck_theta,
    inverted) becomes [coeff_SI * THETA_LEN_SCALE^m] * P_deck^(2n) * theta_deck^m in SI energy
    units; converting that bracket to deck energy units via si_to_deck_landau's OWN SI_TO_DECK_ENERGY
    factor gives coeff_deck = coeff_SI * THETA_LEN_SCALE^m * SI_TO_DECK_ENERGY. Verified by
    gates/g22c (FD force-energy consistency catches any wrong power here immediately -- a
    wrong theta_power exponent shows up as a large, obvious relative error, not a subtle one)."""
    return value_si * (THETA_LEN_SCALE ** theta_power) * SI_TO_DECK_ENERGY


def cubic_compliance_to_stiffness(s11, s12, s44):
    """Standard cubic-symmetry Voigt elastic compliance -> stiffness
    inversion (used for pto_Haun1987 and pzt_Haun_x, both given as
    compliances s_ij [m^2/N] in the source PDF): returns (c11,c12,c44) in
    the reciprocal unit [Pa] if s is in [1/Pa]=[m^2/N]. Standard result for
    cubic symmetry (3 independent constants): with s_ij in Voigt notation,
        c11 = (s11+s12) / [(s11-s12)(s11+2*s12)]
        c12 = -s12 / [(s11-s12)(s11+2*s12)]
        c44 = 1/s44
    """
    denom = (s11 - s12) * (s11 + 2 * s12)
    c11 = (s11 + s12) / denom
    c12 = -s12 / denom
    c44 = 1.0 / s44
    return c11, c12, c44


# ==========================================================================
# SPEC V2-02.1 (ruling D7, Lens 2 U1-U8): the PERMITTIVITY / E-FIELD mapping,
# derived from this module's own declared system (deck energy density 1e9
# J/m^3, deck length 1 nm, P unscaled) and the engine's own electrostatics
# (f_elec = -1/2 P.E_d - P.E_ext with div(kappa grad phi) = div P, SPEC 1.3):
#
#   E_unit = 1e9 J/m^3 / 1 C/m^2 = 1e9 V/m = 1 V/nm  =>  E_deck = E_SI x 1e-9
#   kappa  = [P]/[E] = 1 / 1e9 F/m                   =>  kappa_deck = kappa_SI x 1e9
#   kappa_SI = eps0 * eps_b                          =>  kappa_deck = 0.0088541878128 * eps_b
#
# CONSEQUENCE (the erratum of record): the legacy deck constant 0.08854187 is
# eps_b = 10 (9.9999991), NOT eps_b = 1 -- exactly what the Ferret decks this
# repo imported from say in words ("here we use e_b = 10", PRB 74, 104014
# (2006); $FERRET_ROOT/ferret/examples/domain_walls/BTO_wall_T298K.i:479-488),
# with the sentence dropped at the WP1 import. Per D7 no deck NUMBER changes;
# labels and helpers do. Verified to 2.2e-16 on solver/bulk3d and to O(hz^2)
# on film3d by gates/core/k5_materials.py item `permittivity_convention`,
# which is the authority for this block.
# ==========================================================================
EPSILON_0_SI = 8.8541878128e-12  # F/m (CODATA 2018 value)


def si_to_deck_field(value_v_per_m):
    """E [V/m] -> deck field; -E.P rescales like an energy density.

    E_deck = E_SI x 1e-9 (1 deck field unit = 1 V/nm = 1e9 V/m). This is the
    SAME plain x1e-9 factor si_to_deck_landau applies, and it is FORCED, not
    chosen: a1_deck = a1_SI x 1e-9 makes P = E/(2 a1) unit-invariant only if
    the field carries the same factor. Use THIS function (not
    si_to_deck_landau) whenever the quantity converted is an E-field."""
    return value_v_per_m * 1e-9


def deck_to_si_field(value_deck):
    """Deck field -> V/m (inverse of si_to_deck_field, x1e9)."""
    return value_deck * 1e9


def si_to_deck_permittivity(value_f_per_m):
    """Absolute permittivity [F/m] -> deck kappa via P^2/(2*kappa).

    kappa_deck = kappa_SI x 1e9, the companion of si_to_deck_field: Poisson
    div(kappa grad phi) = div P fixes [kappa] = [P]/[E] = 1 (C/m^2) / 1e9
    (V/m) = 1e-9 F/m (SPEC V2-02.1)."""
    return value_f_per_m * 1e9


def deck_to_si_permittivity(value_deck):
    """Deck kappa -> absolute permittivity [F/m]."""
    return value_deck * 1e-9


def relative_to_deck_permittivity(eps_r):
    """Relative background permittivity -> kappa (0.0088541878128*eps_r).

    Historical decks used 0.08854187*eps_r. That legacy convention is
    preserved in existing files but is NOT this SI energy-density mapping.
    Measured total low-frequency permittivity is not background eps_r.

    SPEC V2-02.1: the legacy base 0.08854187 IS this mapping evaluated at
    eps_b = 10 (0.088541878128 to 8e-8 relative) -- the Ferret original. So
    every deck reading `kappa_b * 0.08854187` runs at eps_b = 10*kappa_b, and
    every "eps_b = 1" label in Part I is off by exactly 10x. Numbers are
    frozen (D7); this helper is the ONE entry point for new SI-stated decks."""
    return si_to_deck_permittivity(EPSILON_0_SI * eps_r)
