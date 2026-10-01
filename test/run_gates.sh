#!/usr/bin/env bash
# GBD-PCG correctness gates.
# Inside MPCGPU (this repository checked out as its GBD-PCG submodule) run the individually
# attested pytest gates, which build against MPCGPU's top-level GLASS pin. Standalone, build
# and run the examples against GLASS_DIR (default: the GLASS submodule in this repository).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$here/../test/test_gates.py" ] && [ -f "$here/../GLASS/glass.cuh" ]; then
  cd "$here/.."
  exec "${PYTHON:-.venv/bin/python}" -m pytest test/test_gates.py -q -k gbd "$@"
fi
exec make -C "$here/examples" test "$@"
