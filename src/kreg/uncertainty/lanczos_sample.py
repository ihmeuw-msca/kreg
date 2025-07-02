from typing import Callable

import jax.numpy as jnp
from matfree.decomp import tridiag_sym
from matfree.funm import dense_funm_sym_eigh, funm_lanczos_sym


def get_PC_inv_rootH(
    PC_H_apply: Callable, rootpc_apply: Callable, max_order=50, reortho="full"
) -> Callable:
    """Gets function to apply rootpc @ (rootpc@H@rootpc)^(-1/2)
    in order to sample from N(0,H^-1)

    Parameters
    ----------
    PC_H_apply
        function that returns rootpc@H@rootpc@x
    rootpc_apply
        function that returns rootpc@x
    max_order
        maximum number of lanczos steps, by default 50

    Returns
    -------
    Callable
        function which applies rootpc @ (rootpc@H@rootpc)^(-1/2)

    """

    def inv_sqrt(x):
        return 1 / jnp.sqrt(x)

    alg_tridiag_sym = tridiag_sym(max_order, reortho=reortho)
    dense_funm = dense_funm_sym_eigh(inv_sqrt)
    matfun_vec = funm_lanczos_sym(dense_funm, alg_tridiag_sym)

    def lanczos_sample(x):
        return rootpc_apply(matfun_vec(PC_H_apply, x))

    return lanczos_sample
