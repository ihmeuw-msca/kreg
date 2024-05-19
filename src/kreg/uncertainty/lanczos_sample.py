from matfree.lanczos import funm_vector_product_spd
import jax.numpy as jnp

def get_PC_inv_rootH(
    PC_H_apply:callable,
    rootpc_apply:callable,
    max_order = 50,
)-> callable:
    """Gets function to apply rootpc @ (rootpc@H@rootpc)^(-1/2)
    in order to sample from N(0,H^-1)

    Parameters
    ----------
    PC_H_apply : callable
        function that returns rootpc@H@rootpc@x
    rootpc_apply : callable
        function that returns rootpc@x
    max_order : int, optional
        maximum number of lanczos steps, by default 50

    Returns
    -------
    Callable
        function which applies rootpc @ (rootpc@H@rootpc)^(-1/2)
    """
    
    inv_sqrt = lambda x:1/jnp.sqrt(x)
    matfun_vec = funm_vector_product_spd(inv_sqrt, max_order, PC_H_apply)
    def lanczos_sample(x):
        return rootpc_apply(matfun_vec(x))
    return lanczos_sample
    
