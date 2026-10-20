# source before the others
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -z "${PY:-}" ]; then
  for c in python3.13 python3 python; do command -v $c >/dev/null 2>&1 && { PY=$c; break; }; done
fi
export PY
command -v "$PY" >/dev/null || { echo "no python found; set PY=/path/to/python"; return 1 2>/dev/null || exit 1; }
"$PY" -c "import numpy,scipy,jax,h5py,matplotlib,tqdm" 2>/dev/null || \
  echo "missing deps: pip install numpy scipy jax h5py matplotlib tqdm"
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CODE="$ROOT/code"
export OUT="$ROOT/out"
mkdir -p "$OUT"
