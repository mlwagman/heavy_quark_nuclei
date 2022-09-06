import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import scipy as sp
import scipy.interpolate

from .util import parse_table, norm_3vec_sq


def t_n(xp, a, b, coeff, n):
    x = (2*xp-a-b)/(b-a)
    def body(prev_vals, _):
        tnm2, tnm1 = prev_vals
        tn = 2.0 * x * tnm1 - tnm2
        return (tnm1, tn), tn
    t0 = onp.ones(x.shape, dtype=x.dtype)
    t1 = x
    _, tns = jax.lax.scan(body, (t0, t1), onp.arange(len(coeff)-2))
    tns = np.concatenate((np.array([t0, t1]), tns), axis=0)
    return np.sum(coeff[(slice(None),) + (np.newaxis,)*(len(tns.shape)-1)] * tns, axis=0)

def u_n(xp, a, b, coeff, n):
    x = (2*xp-a-b)/(b-a)
    def body(prev_vals, _):
        unm2, unm1 = prev_vals
        un = 2.0 * x * unm1 - unm2
        return (unm1, un), un
    u0 = onp.ones(x.shape, dtype=x.dtype)
    u1 = 2*x
    _, uns = jax.lax.scan(body, (u0, u1), onp.arange(len(coeff)-2))
    uns = np.concatenate((np.array([u0, u1]), uns), axis=0)
    return np.sum(coeff[(slice(None),) + (np.newaxis,)*(len(uns.shape)-1)] * uns, axis=0)

# Chebyshev interp in r^2 space for WF (and deriv, deriv^2)
def load_nn_wavefunction_rsq(fname, ncheb=30):
    rs, f_r12 = parse_table(fname)
    f_r12 = f_r12[:,0]
    rsq = rs**2
    fspline = sp.interpolate.InterpolatedUnivariateSpline(rsq, f_r12)
    rsqi = onp.arange(0,rsq[-1],0.001)
    # interpolate based on fixed pts
    fcheb = onp.polynomial.chebyshev.Chebyshev.fit(rsqi,fspline(rsqi),ncheb)
    coeff = fcheb.coef
    nn = np.size(coeff)
    deriv_coeff = coeff[1:] * onp.arange(1,nn)
    deriv2_coeff_t = coeff * onp.arange(nn)*(onp.arange(nn)+1)
    deriv2_coeff_u = coeff * onp.arange(nn)
    r1 = rsqi[0]
    r2 = rsqi[-1]
    dx_drsq = 2 / (r2 - r1)
    f = lambda R: t_n(norm_3vec_sq(R),r1,r2,coeff,nn)
    df = lambda R: dx_drsq * u_n(norm_3vec_sq(R),r1,r2,deriv_coeff,nn-1)
    ddf = lambda R: dx_drsq**2 * (
        t_n(norm_3vec_sq(R),r1,r2,deriv2_coeff_t,nn) -
        u_n(norm_3vec_sq(R),r1,r2,deriv2_coeff_u,nn)
    ) / (((2*norm_3vec_sq(R) - r1 - r2) / (r2 - r1))**2 - 1)
    return (
        jax.jit(f),
        jax.jit(df),
        jax.jit(ddf)
    )

# Chebyshev interp in r^2 space for potentials
def make_interp_function_rsq(rsq, fr, rsqi=None, ncheb=30):
    fspline = sp.interpolate.InterpolatedUnivariateSpline(rsq, fr)
    if rsqi is None:
        rsqi = onp.arange(0,rsq[-1],0.001)
    # interpolate based on fixed pts
    fcheb = onp.polynomial.chebyshev.Chebyshev.fit(rsqi,fspline(rsqi),ncheb)
    coeff = fcheb.coef
    r1 = rsqi[0]
    r2 = rsqi[-1]
    nn = np.size(coeff)
    return jax.jit(lambda R: t_n(norm_3vec_sq(R),r1,r2,coeff,nn))
