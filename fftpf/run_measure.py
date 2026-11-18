"""RUN-ENTRY package E4 (SPEC § 16.5): what is read from a state, and the loop measures of a pe_loop stage.

``sample`` takes the volume-mean P, the polarization along the field axis and the energy from the Adapter;
``force_max`` is the force residual the relax stop rule uses.  Both are readings: they never change P and never
advance an RNG.

Loop measures (``measure.loop``), from the recorded (E, Pn) series of one pe_loop stage, E in kV/cm (signed along the
axis), Pn = mean P . axis in C/m^2.  Records are taken every ``record_every`` steps starting at the stage entry, and
``record_every`` divides the quarter cycle, so the field extremes and zero crossings are recorded points.

triangle (0 -> +Emax -> 0 -> -Emax -> 0), the LAST full cycle k = floor(cycles), with m records per quarter, points
j = 0 .. 4m (j = 0 and 4m: E = 0 rising, m: +Emax, 2m: E = 0 falling, 3m: -Emax):
    Pmax = Pn[m]                           Pn at +Emax
    Pr   = (Pn[2m] - Pn[4m]) / 2           half the difference of Pn at the two E = 0 crossings, falling minus rising
    Ec   = (E_rise - E_fall) / 2           E_rise: field where Pn crosses zero going up inside j = 0..m (first point with
                                           Pn < 0 <= Pn next), E_fall: where it crosses zero going down inside
                                           j = m..3m (Pn > 0 >= Pn next); both by linear interpolation between the two
                                           recorded points.  None when a branch has no crossing.
    W_store, W_rec, W_loss, eta            pf2d_energy_storage_loop.energy_integrals on the positive half of the cycle,
                                           j = 0 .. 2m (charge 0 -> +Emax, discharge +Emax -> 0), n1 = m.
unipolar (E: 0 -> Emax -> 0, one record per field point, 2 n1 + 1 points including the start):
    Pmax = Pn[n1], Pr = Pn[-1] (E = 0 after the discharge), W_store, W_rec, W_loss, eta from energy_integrals, n1.
energy_integrals works on E in deck field units, so the recorded kV/cm are converted with
units.si_to_deck_field(E * 1e5) (the conversion the stepping uses); its W are deck energy densities, written as
``W_*_deck``, and x 1e3 as ``W_*`` in MJ/m^3 (1 deck = 1e9 J/m^3).

Phase fractions (``measure.phase_fractions``): the classes of Runs/NBT6BT/two_well_20261104/readout_nbt6bt.py (``classify``),
with the same arithmetic.  P in C/m^2, P_ref the reference rhombohedral |P|, c111 = sum|P_i| / (sqrt(3) |P|),
c100 = max|P_i| / |P|, cos_a = cos(angle_deg) (15 deg by default):
    R      c111 >= cos_a and |P| >= R_factor * P_ref
    T      c100 > c111 and |P| >= P_min            (closer to <100> than to <111>)
    T15    c100 >= cos_a and |P| >= P_min          (inside T for angle_deg <= 25)
    other  |P| >= P_min and not R and not T
    low    |P| < P_min
R, T, other, low partition the cells.  Scheme "rto" (below) is the second, orthorhombic scheme.  Fractions are plain cell means for pf2d / pf3d; for film the in-plane mean of
each z layer is weighted by grid["qw"] / sum(qw) (the weights of Adapter.mean_P).

Scheme "rto" (``measure.phase_fractions.scheme = "rto"``; ``rt`` is the default and is exactly the scheme above): tetragonal /
orthorhombic / rhombohedral cones about <100> (6 directions), <110> (12) and <111> (8).  With a = |P_i| sorted descending
(a1 >= a2 >= a3) and |P| the length of P in C/m^2:
    low    |P| < P_min                                   (default 0.05 C/m^2)
    else   c100 = a1 / |P|,  c110 = (a1 + a2) / (sqrt(2) |P|),  c111 = (a1 + a2 + a3) / (sqrt(3) |P|)
           -- the cosines to the nearest member of each family.  The cell is T, O or R if the largest of the three belongs
           to <100>, <110> or <111> AND that cosine >= cos(angle_deg) (default 15 deg); otherwise "other".
    T, O, R, other, low partition the cells.  angle_deg must be <= 17.5 deg: the nearest members of different families are
    at least 35.26 deg apart (<110> to <111>; 45 deg <100> to <110>, 54.74 deg <100> to <111>), so cones of half-angle
    <= 17.63 deg are disjoint.  A cell that is inside one cone is therefore automatically the largest cosine of the three, and
    ties between families cannot occur inside a cone (equal cosines put the cell >= 17.63 deg from both, outside both):
    the code tests cos >= cos(angle_deg) per family and needs no argmax.  P_ref and R_factor are not used (ConfigError if given).
    Fractions are f_R, f_O, f_T, f_other, f_low; the film weighting is that of scheme "rt".

Dielectric spectrum (``ac`` / ``fdt`` stages, ``measure.spectrum``; section "dielectric spectrum" below for the definitions).

Saved states and domain maps (``measure.domain_map``): ``saved_states`` lists, from the file alone, the state files a run
writes (event names as run_protocol.schedule gives them); ``domain_map_targets`` resolves the ``at`` selectors.
"""
import numpy as np

from fftpf import units
from fftpf.run_input import ConfigError

if not hasattr(np, "trapezoid"):        # numpy < 2 (1.26 here): pf2d_energy_storage_loop.energy_integrals calls np.trapezoid,
    np.trapezoid = np.trapz             # which is np.trapz under its numpy 2 name -- the same function, so the same numbers


def sample(adapter, P, E, measure, axis=None):
    """One reading: dict(mean_P=[3 floats] if measure["mean_P"], Pn, E_total, f_total, terms).

    E_total is the engine's total energy (the number casekit.anneal records), f_total its
    density, ``terms`` the other entries of the energy totals when measure["energy_terms"].
    Pn = mean P . axis (NaN without an axis).
    """
    dens, tot = adapter.energy(P, E)
    out = dict(E_total=float(tot["total"]), f_total=float(dens["total"]))
    want = measure.get("mean_P", True)
    mean = adapter.mean_P(P) if (want or axis is not None) else None
    if want:
        out["mean_P"] = [float(x) for x in mean]
    if measure.get("phase_fractions"):
        out.update({f"f_{k}": v for k, v in phase_fractions(adapter, P, measure["phase_fractions"]).items()})
    out["Pn"] = float(mean @ np.asarray(axis)) if axis is not None else float("nan")
    if measure.get("energy_terms"):
        out["terms"] = {k: float(v) for k, v in tot.items() if k != "total"}
    return out


FRACTION_KEYS = ("R", "T", "T15", "other", "low")          # scheme "rt"
FRACTION_KEYS_RTO = ("R", "O", "T", "other", "low")        # scheme "rto"


def scheme_of(prm):
    """"rt" or "rto": the resolved phase_fractions table of scheme "rt" carries no ``scheme`` key."""
    return prm.get("scheme", "rt")


def fraction_keys(prm):
    return FRACTION_KEYS_RTO if scheme_of(prm) == "rto" else FRACTION_KEYS


def describe(prm):
    """One line for ``explain``: the scheme and its thresholds."""
    if scheme_of(prm) == "rto":
        return (f"phase fractions: scheme rto (T/O/R = cones of {prm['angle_deg']:g} deg about <100>/<110>/<111>, "
                f"low = |P| < {prm['P_min']:g} C/m^2, rest other)")
    return (f"phase fractions: scheme rt (R within {prm['angle_deg']:g} deg of <111> and |P| >= {prm['R_factor']:g} * "
            f"P_ref = {prm['R_factor'] * prm['P_ref']:g} C/m^2; T closer to <100> than <111>; low = |P| < {prm['P_min']:g} C/m^2)")


def classes_rto(P, prm):
    """Boolean masks R, O, T, other, low of the field P (3, ...) in C/m^2, scheme "rto" (module docstring)."""
    cos_a = float(np.cos(np.radians(prm["angle_deg"])))
    mag = np.sqrt((P * P).sum(0))
    safe = np.where(mag > 0., mag, 1.)
    a = np.sort(np.abs(P), axis=0)[::-1]                       # a1 >= a2 >= a3
    c100 = a[0] / safe
    c110 = (a[0] + a[1]) / (np.sqrt(2.) * safe)
    c111 = (a[0] + a[1] + a[2]) / (np.sqrt(3.) * safe)
    above = mag >= prm["P_min"]
    T, O, R = (c100 >= cos_a) & above, (c110 >= cos_a) & above, (c111 >= cos_a) & above
    return dict(R=R, O=O, T=T, other=above & ~(R | O | T), low=~above)


def classes(P, prm):
    """Boolean masks of the field P (3, ...) in C/m^2.  Scheme "rt": R, T, T15, other, low (readout_nbt6bt.classify, operation
    for operation); scheme "rto": R, O, T, other, low."""
    if scheme_of(prm) == "rto":
        return classes_rto(P, prm)
    cos_a = float(np.cos(np.radians(prm["angle_deg"])))
    mag = np.sqrt((P * P).sum(0))
    safe = np.where(mag > 0., mag, 1.)
    a = np.abs(P)
    c111, c100 = a.sum(0) / (np.sqrt(3.) * safe), a.max(0) / safe
    above = mag >= prm["P_min"]
    R = (c111 >= cos_a) & (mag >= prm["R_factor"] * prm["P_ref"])
    T = (c100 > c111) & above
    return dict(R=R, T=T, T15=(c100 >= cos_a) & above, other=above & ~R & ~T, low=~above)


def fractions(P, prm, qw=None):
    """Fractions per class (keys fraction_keys(prm)) of the field P (3, ...) in C/m^2; ``prm`` is the measure.phase_fractions table.

    qw: the film's z weights (P is then (3, Nz, Ny, Nx)); None -> plain mean over every cell."""
    masks = classes(np.asarray(P), prm)
    keys = fraction_keys(prm)
    if qw is None:
        return {k: float(masks[k].mean()) for k in keys}
    return {k: float((masks[k].mean(axis=(1, 2)) * qw).sum() / qw.sum()) for k in keys}


def phase_fractions(adapter, P, prm):
    """fractions() of a state of ``adapter`` (film: weighted by grid["qw"])."""
    return fractions(P, prm, adapter.grid["qw"] if adapter.engine == "film" else None)


def phase_index(P, prm):
    """Class index per cell.  Scheme "rt": 0 R, 1 T15, 2 T not T15, 3 other, 4 low (the precedence of fig_snap_20261106.py);
    scheme "rto": 0 R, 1 O, 2 T, 3 other, 4 low."""
    c = classes(np.asarray(P), prm)
    if scheme_of(prm) == "rto":
        return np.select([c["R"], c["O"], c["T"], c["other"]], [0, 1, 2, 3], 4)
    return np.select([c["R"], c["T15"], c["T"], c["other"]], [0, 1, 2, 3], 4)


def force_max(adapter, P, E):
    """max|L * dF/dP| on the grid -- casekit.anneal's residual (pf2d / pf3d) and the
    force_max_deck of cases/bto_film_biaxial.py (film: gradient force + explicit nonlinear force)."""
    from fftpf import film_engine, pf2d_engine, pf3d_engine
    a = adapter
    if a.engine == "pf2d":
        force = pf2d_engine.total_force(P, a.alpha, a.ops, E, elastic_params=a.el)[-1]
    elif a.engine == "pf3d":
        force = pf3d_engine.total_force_3d(P, a.alpha, a.ops, E, elastic_params=a.el)[-1]
    else:
        force = film_engine.grad_force_3d(P, a.grid, a.G) + film_engine.explicit_N(
            P, a.alpha, a.grid, a.pf, E, el=a.el)
    return float(np.max(np.abs(a.L * force)))


# --------------------------------------------------------------------------- loop measures

def _cross(E, Pn, lo, hi, rising):
    """Field of the first zero crossing of Pn between points lo and hi (linear interpolation), or None."""
    for j in range(lo, hi):
        a, b = Pn[j], Pn[j + 1]
        if (a < 0 <= b) if rising else (a > 0 >= b):
            return float(E[j] + (0.0 - a) * (E[j + 1] - E[j]) / (b - a))
    return None


def _integrals(E, Pn, n1):
    from fftpf.pf2d_energy_storage_loop import energy_integrals
    deck = energy_integrals(np.asarray([units.si_to_deck_field(e * 1e5) for e in E]), np.asarray(Pn), n1)
    out = {f"{k}_deck": v for k, v in deck.items() if k != "eta"}
    out.update(W_store=deck["W_store"] * 1e3, W_rec=deck["W_rec"] * 1e3, W_loss=deck["W_loss"] * 1e3, eta=deck["eta"])
    return out


def triangle_cycle(E, Pn, m):
    """All triangle measures from one cycle's 4m + 1 points (E in kV/cm, Pn in C/m^2); see the module docstring."""
    E, Pn = np.asarray(E, dtype=float), np.asarray(Pn, dtype=float)
    if len(E) != 4 * m + 1:
        raise ValueError(f"a triangle cycle has 4m+1 = {4 * m + 1} points, got {len(E)}")
    rise, fall = _cross(E, Pn, 0, m, True), _cross(E, Pn, m, 3 * m, False)
    out = dict(Pmax=float(Pn[m]), Pr=float((Pn[2 * m] - Pn[4 * m]) / 2),
               Ec=None if rise is None or fall is None else (rise - fall) / 2,
               E_rise=rise, E_fall=fall)
    out.update(_integrals(E[:2 * m + 1], Pn[:2 * m + 1], m))
    return out


def loop(samples, st, names, every):
    """The requested ``names`` for a pe_loop stage ``st`` from its recorded ``samples`` (dicts with step, E_kV_cm, Pn)."""
    wf = st["waveform"]
    if wf == "triangle":
        spc = st["steps_per_cycle"]
        m = (spc // 4) // every
        k = int(st.get("cycles", 1.0) + 1e-9)
        pts = samples[(k - 1) * 4 * m:k * 4 * m + 1]
        if len(pts) != 4 * m + 1 or pts[0]["step"] != (k - 1) * spc or pts[-1]["step"] != k * spc:
            raise ValueError("loop: the recorded series does not hold the last full cycle")
        full = triangle_cycle([r["E_kV_cm"] for r in pts], [r["Pn"] for r in pts], m)
        out = dict(waveform=wf, cycle=k, cycles_run=st.get("cycles", 1.0))
    else:                                           # unipolar
        n1 = st["n_points"]
        E, Pn = [r["E_kV_cm"] for r in samples], [r["Pn"] for r in samples]
        full = dict(Pmax=float(Pn[n1]), Pr=float(Pn[-1]), **_integrals(E, Pn, n1))
        out = dict(waveform=wf, n_points=n1)
    out.update({name: full[name] for name in names})
    if "Ec" in names and full["Ec"] is None:
        out["Ec_note"] = "no zero crossing of Pn on one branch of the cycle"
    if any(n.startswith("W_") or n == "eta" for n in names):
        out.update({k: v for k, v in full.items() if k.endswith("_deck")})
    return out


# --------------------------------------------------------------------------- saved states and domain maps

def stage_events(st):
    """([(event name, step in the stage)], total steps) of a cool / hold / pe_loop stage: the states it saves.

    The names are those of run_protocol.schedule: cool T<rung temperature :g> at each rung end; hold ``ramped`` (0 < ramp
    < n_steps) and ``end``; triangle c<k>_Emax, c<k>_E0down (E = 0 falling, positive remanent state), c<k>_E0up (E = 0
    rising, negative remanent state) at the quarter cycles; unipolar ``Emax`` and ``end``."""
    kind = st["kind"]
    if kind == "cool":
        ends, total = [], 0
        for T, n in st["rungs"]:
            total += n
            ends.append((f"T{T:g}", total))
        return ends, total
    if kind == "hold":
        total, ramp = st["n_steps"], st["ramp_steps"]
        return ([("ramped", ramp)] if 0 < ramp < total else []) + [("end", total)], total
    if st["waveform"] == "unipolar":
        n1, nd = st["n_points"], st["n_dwell"]
        return [("Emax", n1 * nd), ("end", 2 * n1 * nd)], 2 * n1 * nd
    sps = st["steps_per_cycle"] // 4
    total = int(round(4 * st.get("cycles", 1.0))) * sps
    names = {1: "Emax", 2: "E0down", 0: "E0up"}
    return [(f"c{(q * sps - 1) // (4 * sps) + 1}_{names[q % 4]}", q * sps)
            for q in range(1, total // sps + 1) if q % 4 in names], total


def saved_states(cfg):
    """Every state file a run of ``cfg`` writes, in stage order: dicts(i, kind, event, step, total, stem).

    relax / anneal: ``state_<i>_<kind>`` (event "end"); cool / hold / pe_loop: ``state_<i>_<kind>_<event>``."""
    out = []
    for i, st in enumerate(cfg["stage"]):
        if st["kind"] in ("relax", "anneal"):
            out.append(dict(i=i, kind=st["kind"], event="end", step=None, total=None, stem=f"state_{i}_{st['kind']}"))
        elif st["kind"] in ("cool", "hold", "pe_loop"):
            events, total = stage_events(st)
            out += [dict(i=i, kind=st["kind"], event=e, step=s, total=total, stem=f"state_{i}_{st['kind']}_{e}")
                    for e, s in events]
    return out


_EVENT_ALIAS = {"E0+": "E0down", "E0-": "E0up"}


def _select(sel, states, cfg):
    head, sep, event = sel.partition(":")
    if not sep or not head or not event:
        raise ConfigError(f"measure.domain_map.at: {sel!r} is not \"<stage kind or index>:<event>\"")
    stages = [i for i, st in enumerate(cfg["stage"]) if (str(i) == head if head.isdigit() else st["kind"] == head)]
    event = _EVENT_ALIAS.get(event, event)
    hit = []
    for i in stages:
        mine = [s for s in states if s["i"] == i]
        if event == "end":                                  # the state at the stage's last step
            pick = [s for s in mine if s["step"] is None or s["step"] == s["total"]][-1:]
        else:
            pick = [s for s in mine if s["event"] == event]
            if not pick:                                    # c<k>_<event>: the LAST cycle that has it
                pick = [s for s in mine if s["event"].endswith("_" + event)][-1:]
        hit += pick
    return hit


def domain_map_targets(cfg):
    """The saved states to draw (list of the dicts of saved_states), from measure.domain_map.at.

    at = "all": every saved state.  Otherwise each selector "<stage>:<event>": <stage> is a stage kind (every stage of
    that kind) or an index; <event> is "end" (the state at the stage's last step; a triangle loop that stops at -Emax
    has none), "E0+" / "E0-" (= E0down / E0up), "Emax", a rung name (T350) or any event name.  A triangle event given
    without its cycle (Emax, E0down, E0up, E0+, E0-) means the LAST cycle of the loop; ask for c1_E0down for another.
    A selector that matches no state is a ConfigError."""
    states = saved_states(cfg)
    at = cfg["measure"]["domain_map"]["at"]
    if at == "all":
        return states
    out = []
    for j, sel in enumerate(at):
        hit = _select(sel, states, cfg)
        if not hit:
            raise ConfigError(f"measure.domain_map.at[{j}]: {sel!r} matches no state this file saves "
                              f"(saved: {sorted({s['stem'][len('state_'):] for s in states})})")
        out += [s for s in hit if s not in out]
    return out


# --------------------------------------------------------------------------- dielectric spectrum (E3c)
#
# The observable of a spectrum stage is the RAW series  s(t) = sum over cells of (P . n),  n the unit axis (for n = (1,0,0)
# exactly P[0].sum(), as cases/wp13_debye_case.py records it).  The series is stored as it is; the per-volume quantities
# are made in the estimators, never by rescaling it.  With N = number of cells, V = cell volume, p = s / N the mean
# polarization along n, and kT the Langevin energy DENSITY of SPEC § 2 (units.kT_langevin = k_B T / V):
#
#   lock-in   chi_deck(w) = (complex amplitude of p) / E0_deck = lockin_chi(s, dt, w, N * E0_deck)
#             -- the lock-in divides by N * E0 because a uniform E0 couples to the sum s with N * E0 (wp13 ``E0_eff``).
#   FDT       chi_deck''(w) = w S_p(w) N / (2 kT),   S_p = S_s / N^2,  hence the estimators are called with kT_eff = N * kT:
#             welch_chi_pp(s, dt, N kT) gives w S_s / (4 N kT), and fl_chi(.., C_s, N kT) = C_s / (N kT) + ...;
#             static limit  chi_deck(0) = C_s(0) / (N kT) = Var(s) / (N kT) = N Var(p) / kT.
#             (Var(s) = N kT / stiffness is the stationary law of the unnormalised mode, pf2d_engine.step_langevin docstring.)
#   SI        chi = chi_deck / eps0_deck,  eps0_deck = units.relative_to_deck_permittivity(1.0) = 0.0088541878128
#             (P in C/m^2, E_deck = E_SI * 1e-9 V/m, so chi_deck = 1e9 * chi_SI * eps0_SI);
#             eps_real = eps_b + chi' / eps0_deck,  eps_imag = chi'' / eps0_deck,  tan_delta = eps_imag / eps_real.
# chi'' > 0 for w > 0 (the convention of dielectric_spectrum: chi0 / (1 - i w tau)).

N_BLOCKS = 4                                  # sub-series of one fdt trajectory used to find its resolved range


def eps0_deck():
    return units.relative_to_deck_permittivity(1.0)


def total_along(P, n):
    """The raw observable s = sum over cells of P . n (n a unit vector as a list of floats); exactly P[k].sum() for n = e_k."""
    s = 0.0
    for k in range(3):
        if n[k] != 0.0:
            s = s + n[k] * P[k].sum()
    return s


def lockin_deck(series, dt, omega, E0_deck, n_cells, n_skip=0):
    """(chi', chi'') per volume in deck units from the raw series of one tone.

    The first ``n_skip`` samples (the discarded periods) are dropped; the drive clock started at the first driven step, so
    the kept window starts n_skip * dt late and the lock-in result is turned back by exp(i w n_skip dt) (nothing is done,
    and the result is dielectric_spectrum.lockin_chi untouched, when n_skip = 0)."""
    from fftpf import dielectric_spectrum as ds
    cp, cpp = ds.lockin_chi(np.asarray(series)[n_skip:], dt, omega, n_cells * E0_deck)
    if n_skip:
        chi = complex(cp, cpp) * np.exp(1j * omega * n_skip * dt)
        cp, cpp = chi.real, chi.imag
    return cp, cpp


def fdt_deck(series, dt, kT, n_cells, nperseg, max_lag):
    """The FDT route of one equilibrium series: dict(omega, chi_pp (Welch), chi_p (fl_chi), C, C_blocks, tau, kT_eff,
    chi_static).  kT_eff = n_cells * kT (see the block comment); C_blocks: the correlation of N_BLOCKS equal sub-series."""
    from fftpf import dielectric_spectrum as ds
    kT_eff = n_cells * kT
    omega, chi_pp = ds.welch_chi_pp(series, dt, kT_eff, nperseg)
    C = ds.sample_correlation(series, max_lag)
    tau = np.arange(max_lag) * dt
    chi_p = ds.fl_chi(omega, tau, C, kT_eff).real
    blocks = np.array([ds.sample_correlation(x, max_lag) for x in np.array_split(np.asarray(series), N_BLOCKS)])
    return dict(omega=omega, chi_pp=chi_pp, chi_p=chi_p, C=C, C_blocks=blocks, tau=tau, kT_eff=kT_eff,
                chi_static=C[0] / kT_eff)


def _mean_se(a):
    """Mean over replicas (axis 0) and its standard error (None for one replica)."""
    a = np.asarray(a, dtype=float)
    return a.mean(axis=0), (a.std(axis=0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else None)


def spectrum_tables(recs, routes, seconds_per_deck_time=None):
    """Spectrum tables from the per-replica records (one dict per replica: route -> dict of arrays as the stages saved them).

    Returns (tables, info).  tables[route]: dict of equal-length arrays omega, [f_Hz], chi_real, chi_imag, eps_real,
    eps_imag, tan_delta (replica means) and, with more than one replica, the same names + "_se" (standard error of the
    mean, ddof = 1).  eps and tan_delta are formed PER replica, then averaged.  fdt: only the resolved range --
    dielectric_spectrum.measure_chi_p_floor + contiguous_resolved_range, the spread of chi' over the replicas, or over
    the N_BLOCKS sub-series of the one trajectory when there is a single replica."""
    from fftpf import dielectric_spectrum as ds
    R, e0, tables, info = len(recs), eps0_deck(), {}, {}
    for route in routes:
        rs = [r[route] for r in recs]
        omega, eps_b = np.asarray(rs[0]["omega"]), float(rs[0]["eps_b"])
        chi_p, chi_pp = (np.array([r[k] for r in rs]) for k in ("chi_p", "chi_pp"))
        keep, note = slice(None), {}
        if route == "fdt":
            C = np.array([r["C"] for r in rs]) if R > 1 else np.asarray(rs[0]["C_blocks"])
            _, _, ok = ds.measure_chi_p_floor(C, np.asarray(rs[0]["tau"]), omega, float(rs[0]["kT_eff"]))
            i_max, lo, hi = ds.contiguous_resolved_range(omega, ok)
            keep = slice(0, i_max + 1)
            note = dict(resolved_omega=None if i_max < 0 else [float(lo), float(hi)], n_resolved=i_max + 1,
                        resolved_from="replicas" if R > 1 else f"{N_BLOCKS} sub-series",
                        chi_static_deck=_stat([float(r["chi_static"]) for r in rs]))
        chi_p, chi_pp, omega = chi_p[:, keep], chi_pp[:, keep], omega[keep]
        er, ei = eps_b + chi_p / e0, chi_pp / e0
        cols = dict(chi_real=chi_p, chi_imag=chi_pp, eps_real=er, eps_imag=ei, tan_delta=ei / er)
        t = {"omega": omega}
        if seconds_per_deck_time is not None:
            t["f_Hz"] = omega / (2.0 * np.pi * seconds_per_deck_time)
        se = {}
        for k, a in cols.items():
            t[k], se[k + "_se"] = _mean_se(a)
        if R > 1:
            t.update(se)
        tables[route] = t
        info[route] = dict(n_points=int(len(omega)), eps_b=eps_b, **note)
    return tables, info


def _stat(values):
    m, se = _mean_se(np.array(values))
    return dict(mean=float(m), se=None if se is None else float(se), n=len(values))
