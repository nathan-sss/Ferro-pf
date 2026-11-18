"""RUN-ENTRY package E4 (SPEC § 16.5): domain maps, one PNG per map.

``domain_map(path, kind, P, grid, engine, title, prm=None, fractions=None)`` draws one map of the state P (C/m^2,
(3, Ny, Nx) for pf2d, (3, Nz, Ny, Nx) for pf3d and film; 3D: the mid-z plane, Nz // 2, named in the title):

  "direction"  polarization direction as signed RGB (the rule of Runs/NBT6BT/two_well_20261104/fig_snap_20261106.py):
               colour = 0.5 + 0.5 P / |P|, so [111]-type directions are the 8 corner colours and [100]-type the 6 face
               colours; |P| does not enter the colour (zero P is grey).  The mean and maximum |P| of the plane are
               stated under the map; a swatch key lists the 14 directions.
  "phase"      the class of each cell (run_measure.phase_index) in a fixed palette -- R dark, T light:
               R navy, T within 15 deg of <100> pale yellow, other T pale salmon, other mid grey, low near white;
               the legend gives each class with its box fraction (``fractions``, to 3 decimals).
               Scheme "rto" (prm["scheme"]): R navy, O vermillion (Okabe-Ito, a middle tone: blue / orange is the colour-blind-safe
               axis, and its lightness sits between R and T), T pale yellow, other mid grey, low near white.
               The direction key keeps its 14 swatches: the 12 <110> directions would make it unreadable at this width.

``spectrum(path, tables, quantities, title, calibrated)`` draws the dielectric spectrum of run_measure.spectrum_tables: one
panel per requested quantity (eps_real, eps_imag, tan_delta) against frequency on a log axis -- reduced frequency "(1 / deck
time)" unless calibrated (then f in Hz) -- lock-in as markers, FDT as a line, error bars / a band when the tables carry
standard errors.

Axes in nm, equal aspect, 200 dpi.  matplotlib is imported here, lazily, with the Agg backend.
"""
import numpy as np

from fftpf import run_measure

PALETTE = ("#1f3a68", "#f6dc7c", "#f0b49c", "#8c8c8c", "#f4f4f1")      # R, T15, T (not T15), other, low
PALETTE_RTO = ("#1f3a68", "#d55e00", "#f6dc7c", "#8c8c8c", "#f4f4f1")  # R, O, T, other, low
DPI = 200


def signed_rgb(p):
    """fig_snap_20261106.signed_rgb: 0.5 + 0.5 p / |p| along the first axis, moved to the last."""
    mag = np.sqrt((p * p).sum(0))
    return np.moveaxis(.5 + .5 * p / np.where(mag > 0, mag, 1), 0, -1)


def _plane(P):
    """(plane (3, Ny, Nx), z index or None, Nz or None)."""
    if P.ndim == 3:
        return P, None, None
    z = P.shape[1] // 2
    return P[:, z], z, P.shape[1]


def _draw_key(ax):
    """Swatches of the 8 R-like and 6 T-like directions, as fig_snap's draw_key."""
    from matplotlib.patches import Rectangle
    mark = {1: "1", 0: "0", -1: r"$\bar{1}$"}
    rv = [(1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1)]
    tv = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    x = 0.
    for vs, name in ((rv, "R-like"), (tv, "T-like")):
        ax.text(x, .5, name, ha="left", va="center", fontsize=7)
        x += 1.5
        for v in vs:
            c = .5 + .5 * np.array(v, float) / np.linalg.norm(v)
            ax.add_patch(Rectangle((x, .42), .78, .55, color=c, ec="#d9d8d3", lw=.5))
            ax.text(x + .39, .34, "[" + "".join(mark[q] for q in v) + "]", ha="center", va="top", fontsize=6)
            x += .86
        x += .5
    ax.set_xlim(0, x)
    ax.set_ylim(0, 1)
    ax.axis("off")


def domain_map(path, kind, P, grid, engine, title, prm=None, fractions=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    P = np.asarray(P, dtype=float)
    plane, z, Nz = _plane(P)
    if z is not None:
        title += f"\nmid-plane z layer {z} of {Nz}" if Nz > 1 else "\nthe only z layer (Nz = 1)"
    extent = [0, float(grid["Lx"]), 0, float(grid["Ly"])]
    fig = plt.figure(figsize=(5.4, 6.2), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, .09])
    ax, below = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    show = dict(origin="lower", extent=extent, interpolation="nearest")
    if kind == "direction":
        ax.imshow(signed_rgb(plane), **show)
        mag = np.sqrt((plane * plane).sum(0))
        below.set_title(f"colour = direction only; |P| in this plane: mean {mag.mean():.3f}, max {mag.max():.3f} C/m$^2$",
                        fontsize=7, pad=2)
        _draw_key(below)
    else:
        index = run_measure.phase_index(plane, prm)
        f = fractions
        low = f"low |P| (< {prm['P_min']:g} C/m$^2$)  {f['low']:.3f}"
        if run_measure.scheme_of(prm) == "rto":
            palette = PALETTE_RTO
            a = f"{prm['angle_deg']:g}$^\\circ$"
            names = (f"R, within {a} of <111>  {f['R']:.3f}", f"O, within {a} of <110>  {f['O']:.3f}",
                     f"T, within {a} of <100>  {f['T']:.3f}", f"other  {f['other']:.3f}", low)
        else:
            palette = PALETTE
            names = (f"R  {f['R']:.3f}", f"T within {prm['angle_deg']:g}$^\\circ$ of <100>  {f['T15']:.3f}",
                     f"T, wider  {f['T'] - f['T15']:.3f}", f"other  {f['other']:.3f}", low)
        ax.imshow(index, cmap=ListedColormap(palette), vmin=-.5, vmax=4.5, **show)
        below.axis("off")
        below.legend(handles=[Patch(facecolor=c, edgecolor="#bdbcb6", label=n) for c, n in zip(palette, names)],
                     loc="center", ncol=2, frameon=False, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)", fontsize=8)
    ax.set_ylabel("y (nm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


_Y_LABEL = {"eps_real": "$\\varepsilon'$", "eps_imag": "$\\varepsilon''$", "tan_delta": "tan $\\delta$"}


def spectrum(path, tables, quantities, title, calibrated=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_key = "f_Hz" if calibrated else "omega"
    fig, axes = plt.subplots(len(quantities), 1, sharex=True, figsize=(5.2, 2.3 * len(quantities) + 0.6), layout="constrained",
                             squeeze=False)
    colour = {"lockin": "#1f3a68", "fdt": "#c0504d"}
    label = {"lockin": "lock-in (ac)", "fdt": "fluctuation (fdt)"}
    for ax, q in zip(axes[:, 0], quantities):
        for route, t in tables.items():
            if not len(t[x_key]):
                continue
            x, y, e = t[x_key], t[q], t.get(q + "_se")
            if route == "lockin":
                ax.errorbar(x, y, yerr=e, fmt="o", ms=4, color=colour[route], capsize=2, lw=1, label=label[route])
            else:
                ax.plot(x, y, "-", lw=1, color=colour[route], label=label[route])
                if e is not None:
                    ax.fill_between(x, y - e, y + e, color=colour[route], alpha=.2, lw=0)
        ax.set_xscale("log")
        ax.set_ylabel(_Y_LABEL[q], fontsize=9)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(frameon=False, fontsize=7)
    axes[0, 0].set_title(title, fontsize=8)
    axes[-1, 0].set_xlabel("frequency (Hz)" if calibrated else "reduced frequency (1 / deck time)", fontsize=8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
