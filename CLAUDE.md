# GBD-PCG orientation

Read README.md. GBD-PCG is a header-only, cooperative grid-wide PCG for SPD block-tridiagonal
systems. It is developed here and consumed by MPCGPU as a pinned submodule.

## GLASS

GBD-PCG builds on GLASS block primitives (`glass::block::`). Two ways to supply GLASS:

- Standalone: the `GLASS/` submodule in this repository (the pin CI compiles against).
- As a consumer: pass `GLASS_DIR=<path>` to `examples/Makefile`, or put your own GLASS on the
  include path (`-I<path>`) when compiling code that includes `gpu_pcg.cuh`. MPCGPU does this
  with its top-level GLASS pin; the nested `GLASS/` here is unused in that build.

Bump the `GLASS/` pin deliberately and note it in the commit; MPCGPU checks that its top-level
GLASS pin equals this repository's nested pin, so bump both together when changing either.

## Standing rules

- Preserve cooperative grid-wide iteration; defer block algebra to `glass::block::`. GLASS's
  single-block PCG is not a substitute for the grid-wide decomposition.
- The recurrence eta norm is not the true residual; SPD is a caller precondition.
- Small scratch layouts and N=1 halos have explicit regression gates (in MPCGPU's suite).
- Use `test_api.cu` and `test_pcg_spd.cu`; `pcg_solve*.cu` are compatibility entry points.
  `test_bdmv.cu` and `test_pcg_dumped.cu` consume matrices dumped by MPCGPU.
- Short single-line commits. Running the examples needs a GPU; CI only compiles. On a shared
  machine, correctness builds are fine; timing needs an explicitly assigned quiet window.
