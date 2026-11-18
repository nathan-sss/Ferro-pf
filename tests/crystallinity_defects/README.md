# Portable feature verification

`python tests/run_crystallinity_defects.py` executes the 14 focused checks from
the accepted feature work, plus one mixed-spatial-mask restart check. These
helpers are a tracked, standalone mirror of the feature checks, not a copy or
replacement of the ignored local CORE gate infrastructure. No stored historical
baseline is regenerated or distributed here.

Coverage: crystallinity energy/force and limits; total static-field finite
differences including film surfaces/SPM; signed bias and frozen-wall controls;
aging energy derivatives, exact exponential and square-wave references; coupled
first-order convergence in all three geometries; disabled identity; restart
across drive switches and spatial g=0/tau=inf sites. Fourteen original check
names are retained; `defects_spatial_restart` adds 12 boundary-position checks.

`python tests/check_film_variational.py` separately verifies the prerequisite
film adjoint correction on an even grid, including surface perturbations,
scalar/profile/SPM permittivity, scalar/tensor gradients, reciprocity and a
closed-form physical electric field. Total energies exclude elasticity here.

The feature's zero/off identities use the repaired film implementation as their
reference. The preceding physical correction intentionally changes inconsistent
historical film numbers; these tests do not assert byte identity with that code.
