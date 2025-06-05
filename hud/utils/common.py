from __future__ import annotations

import io
import logging
import tarfile
import zipfile
from typing import TYPE_CHECKING, Any, TypedDict

from pathspec import PathSpec
from pydantic import BaseModel

from hud.server.requests import make_request
from hud.settings import settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger("hud.utils.common")


class FunctionConfig(BaseModel):
    function: str  # Format: "x.y.z"
    args: list[Any]  # Must be json serializable

    id: str | None = None  # Optional id for remote execution
    metadata: dict[str, Any] | None = None  # Optional metadata for telemetry

    def __len__(self) -> int:
        return len(self.args)

    def __getitem__(self, index: int) -> Any:
        return self.args[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.args)

    def __str__(self) -> str:
        return f"FC: {self.function}: {', '.join(str(arg) for arg in self.args)} ({self.metadata})"


# Type alias for the shorthand config, which just converts to function name and args
BasicType = str | int | float | bool | None
ShorthandConfig = tuple[BasicType | dict[str, Any] | list[BasicType] | list[dict[str, Any]], ...]

# Type alias for multiple config formats
FunctionConfigs = (
    ShorthandConfig
    | FunctionConfig
    | list[FunctionConfig]
    | list[ShorthandConfig]
    | dict[str, Any]
    | str
)


class Observation(BaseModel):
    """
    Observation from the environment.

    Attributes:
        screenshot: Base64 encoded PNG string of the screen
        text: Text observation, if available
    """

    screenshot: str | None = None  # base64 string png
    text: str | None = None

    def __str__(self) -> str:
        return f"""Observation(screenshot={
            f"{self.screenshot[:100]}..." if self.screenshot else "None"
        }, text={f"{self.text[:100]}..." if self.text else "None"})"""


class ExecuteResult(TypedDict):
    """
    Result of an execute command.

    Attributes:
        stdout: Standard output from the command
        stderr: Standard error from the command
        exit_code: Exit code of the command
    """

    stdout: bytes
    stderr: bytes
    exit_code: int


# ---------------------------------------------------------------------------
# Helper functions for handling ignore patterns
# ---------------------------------------------------------------------------


def _read_ignore_file(file_path: Path) -> list[str]:
    """Return patterns from *file_path* (ignoring blanks / comments)."""
    if not file_path.exists():
        return []

    patterns: list[str] = []
    for line in file_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _gather_ignore_patterns(root_dir: Path, filename: str) -> list[str]:
    """Collect *filename* patterns throughout *root_dir* respecting hierarchy.

    For a nested ignore file located at ``sub/dir/.gitignore`` containing the
    pattern ``foo/``, the returned pattern will be ``sub/dir/foo/`` so that it
    is evaluated relative to *root_dir* when passed to ``PathSpec``.
    """
    gathered: list[str] = []

    root_dir = root_dir.resolve()

    for ignore_file in root_dir.rglob(filename):
        prefix = ignore_file.parent.relative_to(root_dir).as_posix()
        base_prefix = "" if prefix == "." else prefix

        for pat in _read_ignore_file(ignore_file):
            negate = pat.startswith("!")
            pat_body = pat[1:] if negate else pat

            # Leading slash means relative to the directory the ignore file is
            # located in - remove it so we can prepend *prefix* below.
            if pat_body.startswith("/"):
                pat_body = pat_body.lstrip("/")

            full_pattern = f"{base_prefix}/{pat_body}" if base_prefix else pat_body
            if negate:
                full_pattern = f"!{full_pattern}"

            gathered.append(full_pattern)

    return gathered


def _compile_pathspec(
    directory: Path,
    *,
    respect_gitignore: bool,
    respect_dockerignore: bool,
    respect_hudignore: bool,
) -> PathSpec | None:
    """Compile a ``PathSpec`` from all relevant ignore files under *directory*.

    In addition to the standard ``.gitignore`` and ``.dockerignore`` files we now
    recognise a project-specific ``.hudignore`` file that shares the same pattern
    syntax. Each file can be toggled independently through the corresponding
    ``respect_*`` keyword argument.
    """
    patterns: list[str] = []

    if respect_gitignore:
        patterns.extend(_gather_ignore_patterns(directory, ".gitignore"))
    if respect_dockerignore:
        patterns.extend(_gather_ignore_patterns(directory, ".dockerignore"))
    if respect_hudignore:
        patterns.extend(_gather_ignore_patterns(directory, ".hudignore"))

    if not patterns:
        return None

    return PathSpec.from_lines("gitwildmatch", patterns)


def _iter_files(
    directory: Path,
    *,
    respect_gitignore: bool,
    respect_dockerignore: bool,
    respect_hudignore: bool,
) -> Iterator[tuple[Path, Path]]:
    """Yield ``(file_path, relative_path)`` while respecting ignore files."""
    spec = _compile_pathspec(
        directory,
        respect_gitignore=respect_gitignore,
        respect_dockerignore=respect_dockerignore,
        respect_hudignore=respect_hudignore,
    )

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(directory)
        rel_str = rel_path.as_posix()
        if spec and spec.match_file(rel_str):
            continue
        yield file_path, rel_path


def directory_to_tar_bytes(
    directory_path: Path,
    *,
    respect_gitignore: bool = False,
    respect_dockerignore: bool = False,
    respect_hudignore: bool = True,
) -> bytes:
    """
    Converts a directory to a tar archive and returns it as bytes.

    By default the archive respects ignore rules defined in ``.gitignore``,
    ``.dockerignore`` and ``.hudignore`` (each can be disabled via kwargs).
    """
    output = io.BytesIO()

    with tarfile.open(fileobj=output, mode="w") as tar:
        for file_path, rel_path in _iter_files(
            directory_path,
            respect_gitignore=respect_gitignore,
            respect_dockerignore=respect_dockerignore,
            respect_hudignore=respect_hudignore,
        ):
            logger.debug("Adding %s to tar archive", rel_path)
            tar.add(file_path, arcname=str(rel_path))

    output.seek(0)
    return output.getvalue()


def directory_to_zip_bytes(
    context_dir: Path,
    *,
    respect_gitignore: bool = False,
    respect_dockerignore: bool = False,
    respect_hudignore: bool = True,
) -> bytes:
    """Zip *context_dir* and return the zip archive as bytes, respecting ignore rules."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path, rel_path in _iter_files(
            context_dir,
            respect_gitignore=respect_gitignore,
            respect_dockerignore=respect_dockerignore,
            respect_hudignore=respect_hudignore,
        ):
            logger.debug("Adding %s to zip archive", rel_path)
            zipf.write(str(file_path), arcname=str(rel_path))
    return output.getvalue()


async def get_gym_id(gym_name_or_id: str) -> str:
    """
    Get the gym ID for a given gym name or ID.
    """
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/v1/gyms/{gym_name_or_id}",
        api_key=settings.api_key,
    )

    return data["id"]
