import sys, numpy as onp, contextlib, io
N = int(sys.argv[1])
G = 'heavy_quark_nuclei_gfmc_boosted_FV_currents_MRS.py'
src = open(G).read(); cut = src.index("###\n# current operator building blocks")
def load(fast):
    sys.argv = (f"x --n_step 4 --n_walkers 2 --dtau_iMev 1 --count_skip 2 --nf 0 --Nc {max(N,3)} --N_coord {N} --mQ 1.0 --mu 1e-4 "
                f"--mufac 1 --Rstar 0.35 --spoila 0.7 --wavefunction compact --potential full --OLO LO --alpha_s 0.03 "
                f"--Q_dir 1 0 0 --outdir /tmp/xx/ --Q_mag 0.0 --no_color_vectors {'--fast_lap' if fast else ''}").split()
    ns = {'__name__': 'x'}
    with contextlib.redirect_stdout(io.StringIO()): exec(src[:cut], ns)
    return ns
A, B = load(False), load(True)
R = onp.random.default_rng(0).normal(size=(50, N, 3))*A['a0']
for f in ('grad_f_R', 'laplacian_f_R'):
    x, y = onp.array(A[f](R)), onp.array(B[f](R))
    print(N, f, x.shape, y.shape, 'max rel diff', float(onp.max(onp.abs(x-y))/onp.max(onp.abs(x))))
