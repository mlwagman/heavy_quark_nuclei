#!/usr/bin/env python3

from . import mnp

from .bootstrap import make_bs_tensor, make_nbs_tensor
from .eig_util import eig, mp_eig

from .filter import compute_ZCW, compute_block_ZCW

from .norm_util import norm_trick, normalize_overlaps
from .norm_util import block_norm_trick, normalize_block_overlaps

from .outlier import outlier_robust_avg, outlier_robust_avg_decorr

from .orr import make_block_rr_bounds_PrPl, make_block_rr_bounds_PrPl_offdiag

from .rgpof import diagonalize_rgpof, diagonalize_block_rgpof
from .rgpof import make_rgpof_overlaps, make_block_rgpof_overlaps
from .rgpof import make_block_rgpof_ritz_coeffs

from .util import stack_rpadded, rpad_to_shape
from .util import isreal, allreal
from .util import zip_items, unpack_ragged_dict
from .util import make_hankel, make_block_hankel
