import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kreg.kernel.kron_op import KroneckerOperator

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def mats() -> list[jnp.ndarray]:
    rng = np.random.default_rng(0)
    # non-symmetric factors so transpose handling is actually exercised
    return [jnp.asarray(rng.normal(size=(n, n))) for n in (2, 3, 4)]


@pytest.fixture
def dense(mats) -> jnp.ndarray:
    return functools.reduce(jnp.kron, mats)


def test_shape_and_len(mats):
    op = KroneckerOperator(mats)
    assert op.sizes == (2, 3, 4)
    assert op.shape == (24, 24)
    assert len(op) == 24


def test_to_array(mats, dense):
    assert jnp.allclose(KroneckerOperator(mats).to_array(), dense)


def test_matvec(mats, dense):
    op = KroneckerOperator(mats)
    x = jnp.asarray(np.random.default_rng(1).normal(size=24))
    assert jnp.allclose(op @ x, dense @ x)
    assert jnp.allclose(op.dot(x), dense @ x)


def test_matmat(mats, dense):
    op = KroneckerOperator(mats)
    x = jnp.asarray(np.random.default_rng(2).normal(size=(24, 5)))
    assert jnp.allclose(op @ x, dense @ x)


def test_rmatmul_and_transpose(mats, dense):
    op = KroneckerOperator(mats)
    x = jnp.asarray(np.random.default_rng(3).normal(size=24))
    assert jnp.allclose(x @ op, x @ dense)
    assert jnp.allclose(op.T @ x, dense.T @ x)
    assert jnp.allclose(x.T @ op @ x, x.T @ dense @ x)


def test_scalar_multiplication(mats, dense):
    op = KroneckerOperator(mats)
    x = jnp.asarray(np.random.default_rng(5).normal(size=24))
    assert jnp.allclose((2.0 * op) @ x, 2.0 * (dense @ x))
    assert jnp.allclose((op * 2.0) @ x, 2.0 * (dense @ x))
    assert jnp.allclose((2.0 * op).to_array(), 2.0 * dense)


def test_single_factor():
    mat = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    op = KroneckerOperator([mat])
    x = jnp.asarray([1.0, -1.0])
    assert jnp.allclose(op @ x, mat @ x)
    assert jnp.allclose(op.to_array(), mat)


def test_jit_and_grad(mats, dense):
    op = KroneckerOperator(mats)

    def quad(x):
        return 0.5 * x.T @ op @ x

    x = jnp.asarray(np.random.default_rng(4).normal(size=24))
    assert jnp.allclose(jax.jit(quad)(x), 0.5 * x @ dense @ x)
    assert jnp.allclose(jax.grad(quad)(x), 0.5 * (dense + dense.T) @ x)


def test_validation():
    with pytest.raises(ValueError):
        KroneckerOperator([])
    with pytest.raises(ValueError):
        KroneckerOperator([jnp.ones((2, 3))])
    op = KroneckerOperator([jnp.eye(2), jnp.eye(3)])
    with pytest.raises(ValueError):
        op @ jnp.ones(5)
    with pytest.raises(ValueError):
        op @ jnp.ones((6, 2, 2))
