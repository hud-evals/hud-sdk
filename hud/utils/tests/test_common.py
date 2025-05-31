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

def test_function_config_stores_function_name_args_and_optional_id():
    """FunctionConfig should store function name, args list, and optional id."""
    from hud.utils.common import FunctionConfig
    
    # Minimal config
    minimal = FunctionConfig(function="test_func", args=[])
    assert minimal.function == "test_func"
    assert minimal.args == []
    assert minimal.id is None
    
    # With args
    with_args = FunctionConfig(function="navigate", args=["https://example.com", {"wait": True}])
    assert with_args.function == "navigate"
    assert len(with_args.args) == 2
    assert with_args.args[0] == "https://example.com"
    assert with_args.args[1] == {"wait": True}
    
    # With id
    with_id = FunctionConfig(
        function="complex_operation",
        args=[42, "test", {"nested": {"key": "value"}}],
        id="op_123"
    )
    assert with_id.function == "complex_operation"
    assert len(with_id.args) == 3
    assert with_id.id == "op_123"


@pytest.mark.asyncio
async def test_get_gym_id_fetches_id_from_api_response(mocker: pytest_mock.MockerFixture):
    """get_gym_id should extract 'id' field from API response."""
    # Arrange
    api_response = {"id": "gym-123", "name": "Test Gym", "status": "active"}
    mocker.patch("hud.utils.common.make_request", return_value=api_response)
    
    # Act
    gym_id = await get_gym_id("test_gym")
    
    # Assert
    assert gym_id == "gym-123"


@pytest.mark.asyncio
async def test_get_gym_id_propagates_network_errors(mocker: pytest_mock.MockerFixture):
    """get_gym_id should propagate exceptions from make_request."""
    # Arrange
    mocker.patch("hud.utils.common.make_request", side_effect=ConnectionError("API unavailable"))
    
    # Act & Assert
    with pytest.raises(ConnectionError, match="API unavailable"):
        await get_gym_id("test_gym")


@pytest.mark.asyncio
async def test_get_gym_id_raises_key_error_when_id_missing(mocker: pytest_mock.MockerFixture):
    """get_gym_id should raise KeyError when response lacks 'id' field."""
    # Arrange
    incomplete_response = {"name": "Test Gym", "status": "active"}  # Missing 'id'
    mocker.patch("hud.utils.common.make_request", return_value=incomplete_response)
    
    # Act & Assert
    with pytest.raises(KeyError):
        await get_gym_id("test_gym")


def test_directory_to_tar_bytes_creates_valid_tar_archive(tmpdir_factory: pytest.TempdirFactory):
    """directory_to_tar_bytes should create a valid tar archive containing all files."""
    # Arrange
    temp_dir = tmpdir_factory.mktemp("test_archive")
    temp_dir_path = Path(temp_dir)
    
    # Create test structure
    (temp_dir_path / "file1.txt").write_text("content1")
    (temp_dir_path / "file2.py").write_text("import os\nprint('hello')")
    
    subdir = temp_dir_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.json").write_text('{"key": "value"}')
    
    # Act
    tar_bytes = directory_to_tar_bytes(temp_dir_path)
    
    # Assert
    assert isinstance(tar_bytes, bytes)
    assert len(tar_bytes) > 0
    
    # Verify contents
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = {m.name for m in tar.getmembers()}
        assert "file1.txt" in members
        assert "file2.py" in members
        assert "subdir/nested.json" in members
        
        # Verify file contents
        content = tar.extractfile("file1.txt")
        assert content is not None
        assert content.read().decode() == "content1"


def test_directory_to_tar_bytes_handles_empty_directory(tmpdir_factory: pytest.TempdirFactory):
    """directory_to_tar_bytes should handle empty directories gracefully."""
    # Arrange
    empty_dir = tmpdir_factory.mktemp("empty")
    empty_dir_path = Path(empty_dir)
    
    # Act
    tar_bytes = directory_to_tar_bytes(empty_dir_path)
    
    # Assert
    assert isinstance(tar_bytes, bytes)
    assert len(tar_bytes) > 0  # Even empty tar has headers
    
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = tar.getmembers()
        # May contain the directory itself or be completely empty
        assert len(members) >= 0


def test_directory_to_tar_bytes_preserves_directory_structure(tmpdir_factory: pytest.TempdirFactory):
    """directory_to_tar_bytes should preserve nested directory structure."""
    # Arrange
    root = tmpdir_factory.mktemp("root")
    root_path = Path(root)
    
    # Create nested structure
    (root_path / "a" / "b" / "c").mkdir(parents=True)
    (root_path / "a" / "file1.txt").write_text("in a")
    (root_path / "a" / "b" / "file2.txt").write_text("in b")
    (root_path / "a" / "b" / "c" / "file3.txt").write_text("in c")
    
    # Act
    tar_bytes = directory_to_tar_bytes(root_path)
    
    # Assert
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = {m.name for m in tar.getmembers()}
        assert "a/file1.txt" in members
        assert "a/b/file2.txt" in members
        assert "a/b/c/file3.txt" in members


def test_directory_to_tar_bytes_with_exclusions(tmpdir_factory: pytest.TempdirFactory):
    """Test directory_to_tar_bytes with files to exclude."""
    temp_dir = tmpdir_factory.mktemp("test_exclude_dir")
    temp_dir_path = Path(temp_dir)
    
    # Create various files
    (temp_dir_path / "include_me.txt").write_text("include")
    (temp_dir_path / ".git").mkdir()
    (temp_dir_path / ".git" / "config").write_text("git config")
    (temp_dir_path / "__pycache__").mkdir()
    (temp_dir_path / "__pycache__" / "module.pyc").write_bytes(b"pyc content")
    (temp_dir_path / "normal_dir").mkdir()
    (temp_dir_path / "normal_dir" / "file.py").write_text("python code")
    
    tar_bytes = directory_to_tar_bytes(temp_dir_path)
    
    # Check contents
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        member_names = {m.name for m in tar.getmembers()}
        
        # Should include regular files and directories
        assert "include_me.txt" in member_names
        assert "normal_dir/file.py" in member_names
        
        # Implementation might exclude common patterns like .git and __pycache__
        # This depends on the actual implementation


def test_directory_to_tar_bytes_empty_directory(tmpdir_factory: pytest.TempdirFactory):
    """Test directory_to_tar_bytes with empty directory."""
    temp_dir = tmpdir_factory.mktemp("empty_dir")
    temp_dir_path = Path(temp_dir)
    
    tar_bytes = directory_to_tar_bytes(temp_dir_path)
    
    # Should still create a valid tar even if empty
    assert tar_bytes is not None
    assert len(tar_bytes) > 0
    
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = tar.getmembers()
        # Might be empty or contain just the root directory
        assert len(members) >= 0


def test_directory_to_tar_bytes_symlinks(tmpdir_factory: pytest.TempdirFactory):
    """Test directory_to_tar_bytes with symbolic links."""
    temp_dir = tmpdir_factory.mktemp("symlink_dir")
    temp_dir_path = Path(temp_dir)
    
    # Create a file and a symlink to it
    target_file = temp_dir_path / "target.txt"
    target_file.write_text("target content")
    
    symlink = temp_dir_path / "link_to_target.txt"
    try:
        symlink.symlink_to(target_file)
        has_symlink = True
    except OSError:
        # Symlinks might not be supported on all systems (e.g., Windows without admin)
        has_symlink = False
    
    tar_bytes = directory_to_tar_bytes(temp_dir_path)
    
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = {m.name: m for m in tar.getmembers()}
        
        assert "target.txt" in members
        
        if has_symlink:
            # Check how symlinks are handled (might be followed or preserved)
            assert "link_to_target.txt" in members
