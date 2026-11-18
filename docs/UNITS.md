# Units used by FFT-ferro

The solvers use a common internal (“deck”) system. The TOML interface also accepts named SI fields. An unsuffixed Python coefficient generally expects deck units. Check the specific entry point in the [input schema](RUN_INPUT.md) or [parameter reference](PARAMETERS.md).

| Quantity | Convention or conversion to deck units |
|---|---|
| Position and cell spacing | nm. Periodic box lengths are cell count times spacing. Film thickness refers to the distance between its outer nodes. |
| Polarization | C/m². Components come first, followed by spatial axes in y,x or z,y,x order. |
| Free-energy density | One deck unit equals 10⁹ J/m³. A 3D density integrated over nm³ gives energy in units of 10⁻¹⁸ J. A 2D integral over nm² gives energy per out-of-plane length in units of 10⁻⁹ J/m. A wall energy integrated over one nm coordinate is in J/m². |
| Landau coefficients | Multiply SI energy-density coefficients by 10⁻⁹ for every power of unscaled polarization. |
| First-gradient coefficients | Multiply the SI coefficient of a squared spatial derivative by 10⁹. A normalized gradient ratio is dimensionless. |
| Stiffness, stress, strain | Stiffness and stress in GPa. Strain is a fraction internally. A TOML key ending in `_pct` takes percent. |
| Electrostriction | Physical Q in m⁴/C². The TOML `Q44_phys` uses engineering shear. The full bulk tensor input uses Q₁₂₁₂ = Q44_phys / 4. Leave this conversion to the run builder for TOML inputs. |
| Electric field | One deck unit equals 10⁹ V/m. Multiply a field specified in kV/cm by 10⁻⁴ to get deck units. Keys ending in `_kV_cm` take kV/cm directly. |
| Permittivity | For absolute κ in F/m multiply by 10⁹. For relative background εᵦ use κ = 0.0088541878128 εᵦ. |
| Temperature and noise | Temperature in K. For the Langevin step use `kT_langevin(T, hx, hy, hz)`, equal to kᴮT × 10¹⁸ / (hx hy hz) for spacings in nm. In a 2D run the last length is the physical fluctuation-slab thickness. |
| Time and frequency | The integrator advances in deck time. Angular frequency is in radians per deck time unless the interface specifies a physical time conversion. A conversion to seconds or Hz requires a calibrated time scale. |
| Octahedral tilt | Each material specifies its order parameter. The displacement convention converts metres to nm. The angle convention keeps degrees. Coefficients involving tilt depend on the same convention and the power of the tilt in the energy term. |

The [conversion functions](../fftpf/units.py) include the field and permittivity inverses, cubic compliance/stiffness inversion, the tilt-coefficient conversion, and a separate legacy thermal-energy conversion. For higher-gradient terms, normalization of the antiferroelectric field, film quadrature, and measurement-specific units, use the definitions in the [coefficient reference](PARAMETERS.md), [operators](OPERATORS.md), and [run manual](RUN_INPUT.md).
