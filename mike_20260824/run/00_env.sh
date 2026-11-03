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
# GFMC output is large -- one 5000-walker, 7-Q config is ~1-1.5 GB, and a whole campaign
# ran to 14 GB -- and $ROOT sits inside Dropbox, so writing here makes Dropbox sync every
# h5 file (observed: Dropbox.app pinned at 54% CPU while the campaign was running, on top
# of the machine already swapping). Keep the bulk out of any synced folder. The h5 files are
# regenerable, and gfmc_reduce.py compresses a run directory ~50x for analysis.
#   HQN_OUT=/path/to/scratch   overrides explicitly
#   ~/hqn_out                  used automatically when it exists
#   $ROOT/out                  fallback, so this stays portable
if [ -z "${HQN_OUT:-}" ] && [ -d "$HOME/hqn_out" ]; then HQN_OUT="$HOME/hqn_out"; fi
export OUT="${HQN_OUT:-$ROOT/out}"
mkdir -p "$OUT"
