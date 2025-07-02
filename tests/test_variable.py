from kreg.kernel import KernelComponent, KroneckerKernel
from kreg.kernel.factory import build_gaussianRBF_kfunc, vectorize_kfunc
from kreg.variable import Variable


def test_label():
    dummy_kfunc = vectorize_kfunc(build_gaussianRBF_kfunc(1.0))

    kernel = KroneckerKernel([KernelComponent("x", dummy_kfunc)])
    variable = Variable("intercept", kernel=kernel)
    assert variable.label == "intercept/x"

    kernel = KroneckerKernel([KernelComponent(["x", "y"], dummy_kfunc)])
    variable = Variable("intercept", kernel=kernel)
    assert variable.label == "intercept/x*y"

    kernel = KroneckerKernel(
        [
            KernelComponent(
                [{"name": "dummy", "coords": ("x", "y", "z")}], dummy_kfunc
            )
        ]
    )
    variable = Variable("intercept", kernel=kernel)
    assert variable.label == "intercept/(x,y,z)"

    kernel = KroneckerKernel(
        [
            KernelComponent(
                [{"name": "dummy", "interval": ("a", "b")}], dummy_kfunc
            )
        ]
    )
    variable = Variable("intercept", kernel=kernel)
    assert variable.label == "intercept/(a-b)"

    kernel = KroneckerKernel(
        [
            KernelComponent(
                [{"name": "dummy", "coords": ("x", "y", "z")}], dummy_kfunc
            ),
            KernelComponent(
                [{"name": "dummy", "interval": ("a", "b")}], dummy_kfunc
            ),
        ]
    )
    variable = Variable("intercept", kernel=kernel)
    assert variable.label == "intercept/(x,y,z)*(a-b)"
