"""Packaging invariants for the optional CuPy dependency.

CuPy wheels are built against a specific CUDA version, so CuPy cannot be a
normal dependency: a single pin installs a wheel that does not match the driver
of every user who is not on that exact CUDA version. Importing still succeeds,
and the first kernel launch then fails with ``cudaErrorInsufficientDriver``.

The wheels are ``gpu11`` / ``gpu12`` extras instead. These tests keep them that
way and keep the extra names aligned with the wheel they install.
"""
import pathlib

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10; tomli is in the dev extra for it
    tomllib = pytest.importorskip("tomli")

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


@pytest.fixture(scope="module")
def project():
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def test_cupy_is_not_a_normal_dependency(project):
    gpu_deps = [d for d in project["dependencies"] if d.startswith("cupy")]
    assert not gpu_deps, (
        "CuPy is tied to one CUDA version and must stay an extra, not a normal "
        "dependency; found %r" % (gpu_deps,))


@pytest.mark.parametrize("extra, wheel", [
    ("gpu11", "cupy-cuda11x"),
    ("gpu12", "cupy-cuda12x"),
])
def test_each_cuda_version_has_a_cupy_extra(project, extra, wheel):
    extras = project["optional-dependencies"]
    assert extra in extras, "missing the %s extra" % extra
    deps = extras[extra]
    assert any(d.startswith(wheel) for d in deps), (extra, deps)


def test_cuda_extras_do_not_share_a_wheel(project):
    extras = project["optional-dependencies"]
    wheels = {extra: extras[extra] for extra in ("gpu11", "gpu12")}
    assert wheels["gpu11"] != wheels["gpu12"], (
        "gpu11 and gpu12 must install different wheels: %r" % (wheels,))
