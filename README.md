# GBD-PCG

Cooperative, GPU-wide preconditioned conjugate gradient for symmetric positive definite
block-tridiagonal systems, header-only CUDA on top of [GLASS](https://github.com/A2R-Lab/GLASS)
block primitives. One CUDA block owns one block row; cooperative grid synchronization joins the
iterations. This differs from GLASS's single-block PCG and is intentional. All launched blocks
must be simultaneously resident on the current GPU.

GBD-PCG is the linear-system solver of [MPCGPU](https://github.com/A2R-Lab/MPCGPU), which pins
it as a submodule. Development happens here; MPCGPU's signed correctness receipt gates every pin
it takes.

## Build

Requires a C++17 CUDA toolkit and an NVIDIA GPU with cooperative-launch support to run the
examples (compiling needs no GPU).

```bash
git clone --recurse-submodules https://github.com/A2R-Lab/GBD-PCG.git
cd GBD-PCG
make -C examples test ARCH=sm_120 STATE_SIZE=14 KNOT_POINTS=32
```

`STATE_SIZE` and `KNOT_POINTS` are compile-time dimensions. The examples are valid SPD
known-solution systems in float and double; `test/run_gates.sh` runs the same gates.

### Supplying GLASS

GBD-PCG needs exactly one GLASS on the include path:

- **Standalone:** the `GLASS/` submodule in this repository. Its pin is what CI compiles against.
- **From a project that pins GLASS itself:** pass `GLASS_DIR=<path>` to `examples/Makefile`, and
  compile your own code with `-I<your GLASS> -I<GBD-PCG>/include`. The nested `GLASS/` is then
  unused and may be left uninitialized. MPCGPU builds this way with its top-level pin and
  checks that the two pins agree.

## API contract

Include `gpu_pcg.cuh` with `-I<GBD-PCG>/include -I<GLASS>` and C++17. Matrix strips are
column-major `[L | D | R]` per block row, totaling `3 * state_size * state_size * knot_points`
scalars. Boundary strips should be zero. Vectors contain `state_size * knot_points` scalars.

- Host `solvePCG(A,b,x,s,N,&config)`: copies inputs, initializes identity Pinv, solves
  synchronously, copies only x back, returns the actual iteration count. It supports
  `empty_pinv=true` only; a supplied host preconditioner or CSR input throws instead of running
  with uninitialized data.
- Device `solvePCGChecked(...)`: the caller owns matrix, Pinv, RHS, solution and scratch
  allocations. Returns `{iterations, iteration_limit}`. The compatibility device `solvePCG`
  returns iterations. Neither is asynchronous.
- Runtime dimensions must equal the compile-time `STATE_SIZE`/`KNOT_POINTS`. `pcg_block` must
  be one-dimensional and within device limits; launch occupancy is checked on the current
  device. Grid size is always N; legacy `pcg_grid` does not override the cooperative
  decomposition.
- Finite, nonnegative tolerances are required. Host inputs are finite-checked; device callers
  must supply valid finite SPD systems and preconditioners and non-overlapping buffers.
  Indefinite matrices are outside the contract; there is no structured breakdown status yet.
- Default stopping uses `|eta| < abs_tol + rel_tol*|eta_initial|` with `eta = rᵀ Pinv r`. This
  is **not** a relative Euclidean residual tolerance. An exact zero residual exits safely even at
  zero tolerances.
- Experimental `PCG_TRUE_EXIT_CHECK_PERIOD` and `PCG_RESIDUAL_REPLACE_PERIOD` alter stopping and
  the recurrence. They are not default performance claims; validate the true residual for your
  system.

## Examples and gates

| File | Purpose |
| --- | --- |
| `examples/test_api.cu` | Known-solution SPD solve and API regression gate (`host`, `device`, `zero`, `zero-exact`, `warm`, `cap`, `invalid` modes); float or double via `TEST_DOUBLE` |
| `examples/test_pcg_spd.cu` | Random SPD block-tridiagonal residual gate with identity preconditioner |
| `examples/test_bdmv.cu` | Cooperative block-tridiagonal matvec against a host multiply on a strips file |
| `examples/test_pcg_dumped.cu` | Runs the solver on Schur systems dumped by MPCGPU |
| `examples/pcg_solve*.cu` | Compatibility entry points that include `test_api.cu` |

MPCGPU's suite additionally gates small and odd dimensions, N=1, nontrivial preconditioning,
zero RHS, exact warm starts, iteration caps and invalid configurations, and signs the result.

## History and citation

GBD-PCG was published with MPCGPU (Adabag, Atal, Gerard, Plancher, ICRA 2024). Between August and
October 2026 it was developed inside the MPCGPU tree; that history is preserved here. The
current implementation differs from the published one: GLASS block primitives, a relative
tolerance, a converged-start guard and input validation. MPCGPU's
[speedup attribution](https://github.com/A2R-Lab/MPCGPU/blob/main/docs/speedup-attribution.md)
measures what those changes did and did not change.

MIT license (see `LICENSE`).
