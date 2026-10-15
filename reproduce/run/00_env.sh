# source before the others
export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ -z "${PY:-}" ]; then
  for c in python3.13 python3 python; do command -v $c >/dev/null 2>&1 && { PY=$c; break; }; done
fi
export PY
command -v "$PY" >/dev/null || { echo "no python found; set PY=/path/to/python"; return 1 2>/dev/null || exit 1; }
"$PY" -c "import numpy,scipy,jax,h5py,matplotlib,tqdm,mpmath,timeout_decorator" 2>/dev/null || \
  echo "missing deps: pip install numpy scipy jax h5py matplotlib tqdm mpmath timeout-decorator"
"$PY" -c "import korr" 2>/dev/null || \
  echo "korr not found: pip install -e <path to korrfitter_v0>"
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CODE="$(cd "$ROOT/.." && pwd)"
export OUT="$ROOT/out"
mkdir -p "$OUT"
QS="0.0 0.02 0.05 0.1 0.2 0.35 0.5"

# extract <run dir> <meson|baryon> [ref <r^2>]
extract () {
  "$PY" "$CODE/gfmc_reduce.py" "$1" "$1.npz" "${NBINS:-100}" && \
  "$PY" "$CODE/gfmc_korr.py" "$1.npz" "$2" ${3:-} --out "$1.csv"
}
