import pytest

from hud.types import CustomGym


def test_customgym_local_missing_dockerfile(tmp_path):
    """Expect validation error if local gym has no dockerfile and no Dockerfile in directory."""

    controller_dir = tmp_path / "controller"
    controller_dir.mkdir()

    with pytest.raises(ValueError):
        CustomGym(location="local", controller_source_dir=controller_dir)


def test_customgym_local_auto_reads_dockerfile(tmp_path):
    """CustomGym should read Dockerfile content automatically for local location."""

    controller_dir = tmp_path / "controller"
    controller_dir.mkdir()
    dockerfile_path = controller_dir / "Dockerfile"
    dockerfile_path.write_text("FROM scratch\n")

    cg = CustomGym(location="local", controller_source_dir=controller_dir)
    assert cg.dockerfile and "FROM scratch" in cg.dockerfile 