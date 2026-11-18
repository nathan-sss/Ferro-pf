# FFT-ferro v1.0

FFT-ferro evolves a three-component polarization field using semi-implicit ferroelectric phase-field solvers. Choose a periodic plane (`pf2d`), a periodic bulk cell (`pf3d`), or a film with in-plane periodicity and a finite-difference thickness direction (`film`). The Python package is `fftpf`.

## Install and run

With Python 3.11 or newer, run these commands from a checkout of the release.

```sh
git clone --branch v1.0 https://github.com/nathan-sss/FFT-ferro.git
cd FFT-ferro
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m fftpf check docs/run_examples/01_anneal_pf2d.toml
python -m fftpf explain docs/run_examples/01_anneal_pf2d.toml
python -m fftpf run docs/run_examples/01_anneal_pf2d.toml
```

The last command prints the output directory. By default it is under `reports/runs/` in the calling directory. Set `FFT_RUN_ROOT` or the input's `output.dir` to change it. For a short run on each geometry see the [small test cases](tests/run_entry/examples). The [input manual](docs/RUN_INPUT.md) covers the TOML schema, checkpoints, driving protocols and output columns.

## Python interface

```python
import os
from fftpf import backend_config

os.environ.update(backend_config.environment(1))
from fftpf import run_input, run_protocol

config = run_input.load("docs/run_examples/05_film_relax.toml")
run_protocol.check(config)
result = run_protocol.run(config)
print(result["run_dir"])
```

For a custom calculation, import `fftpf.pf2d_engine`, `fftpf.pf3d_engine` or `fftpf.film_engine`. TOML `relax` advances deterministic TDGL steps. The Python `casekit.relax` and `relax_film` routines directly minimize discrete free energy. Only the time-stepping route provides a time trajectory. The [API index](docs/API_INDEX.md), [operator conventions](docs/OPERATORS.md), [minimization guide](docs/MINIMIZE.md) and [legacy import map](docs/MODULE_NAMES.md) document the lower-level interfaces.

## Units and materials

Start with the [unit and tensor conventions](docs/UNITS.md). The [parameter reference](docs/PARAMETERS.md) gives the meanings of the individual coefficients. The [material decks](params/) include source metadata and distinguish active, substrate and archived inputs. Example model coefficients must be evaluated for the material and conditions of the intended calculation.

## Verification

```sh
python -m pip install '.[test]'
python -m pytest -q tests
```

The default backend uses NumPy/SciPy. Install `.[performance]` for the optional Numba kernels. The portable tests include energy/force consistency, geometry and boundary checks, CLI input and restart checks. Some historical comparisons need research fixtures external to this release. Historical references in the migrated manuals identify the development records for those claims.

The solver and interfaces were extracted from FFT-Spectral commit `68210916f0be4f15467109286c34d9dcb99de75a`. FFT-ferro starts its public version series at v1.0.
