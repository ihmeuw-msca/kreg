import jax
import jax.numpy as jnp

def colnorm(M):
    return M/jnp.linalg.norm(M,axis=0,keepdims = True)

def diag_prod(A,B):
    return jnp.sum(A * B,axis=0)

def xdiag(A_apply,n,m,key = jax.random.PRNGKey(10)):
    """x diag estimator from 
    "XTrace: Making the Most of Every Sample in Stochastic Trace Estimation"
    by Epperly, Tropp, Webber

    Parameters
    ----------
    A_apply : Callable
        Linear operator, assumed symmetric
    m : int
        Number of test vectors
    """
    m = int(jnp.floor(m/2))
    rad = jax.random.rademacher(key,(n,m),'float64')
    Y = A_apply(rad)
    Q,R = jnp.linalg.qr(Y)
    Z = A_apply(Q)
    T = Z.T@rad
    S = colnorm(jnp.linalg.inv(R).T)
    
    d_QZ = diag_prod(Q.T,Z.T)
    d_QSSZ = diag_prod((Q@S).T,(Z@S).T)
    d_radQT = diag_prod(rad.T,(Q@T).T)
    d_radY =diag_prod(rad.T,Y.T)
    d_rad_QSST = diag_prod(rad.T,(Q@S@jnp.diag(diag_prod(S,T))).T)
    d = d_QZ + (- d_QSSZ + d_radY - d_radQT + d_rad_QSST)/m
    return d,[d_QZ,d_QSSZ,d_radY,d_radQT,d_rad_QSST]