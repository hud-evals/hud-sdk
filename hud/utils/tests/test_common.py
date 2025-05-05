from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from hud.utils.common import directory_to_tar_bytes, get_gym_id

if TYPE_CHECKING:
    import pytest_mock


def test_directory_to_tar_bytes(tmpdir_factory: pytest.TempdirFactory):
    """Test that a directory can be converted to a tar bytes object."""
    temp_dir = tmpdir_factory.mktemp("test_dir")
    temp_dir_path = Path(temp_dir)

    (temp_dir_path / "test.txt").write_text("test content")

    nested_dir = temp_dir_path / "nested"
    nested_dir.mkdir(exist_ok=True)
    (nested_dir / "file.txt").write_text("nested content")

    tar_bytes = directory_to_tar_bytes(temp_dir_path)
    assert tar_bytes is not None
    assert len(tar_bytes) > 0

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = tar.getmembers()
        member_names = {m.name for m in members}

        assert "test.txt" in member_names
        assert "nested/file.txt" in member_names

        test_content = tar.extractfile("test.txt")
        assert test_content is not None
        assert test_content.read().decode() == "test content"

        nested_content = tar.extractfile("nested/file.txt")
        assert nested_content is not None
        assert nested_content.read().decode() == "nested content"


@pytest.mark.asyncio
async def test_get_gym_id(mocker: pytest_mock.MockerFixture):
    """Test that the gym ID can be retrieved."""
    mocker.patch("hud.utils.common.make_request", return_value={"id": "test_gym_id"})
    gym_id = await get_gym_id("test_gym")
    assert gym_id == "test_gym_id"
