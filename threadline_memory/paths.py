"""Safe mapping from an arbitrary user id to a single JSON file name.

The user id is untrusted input. It must never be able to escape the users
directory (``../``, absolute paths, drive letters, NUL, reserved device names).
We map it to a deterministic, collision-resistant, filesystem-safe stem:
a slugified prefix plus a short hash of the exact original id.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from .errors import InvalidUserIdError

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_STEM = 48
# Windows reserved device names (case-insensitive), which are invalid as files.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def safe_user_filename(user_id: str) -> str:
    """Return a safe ``<slug>-<hash>.json`` file name for ``user_id``."""

    if type(user_id) is not str or not user_id.strip():
        raise InvalidUserIdError("user_id must be a non-empty string")

    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    slug = _SAFE_CHARS.sub("-", user_id.strip()).strip(".-_").lower()
    slug = slug[:_MAX_STEM] if slug else ""
    if not slug or slug in _RESERVED:
        slug = f"user-{slug}" if slug else "user"
    return f"{slug}-{digest}.json"


def user_file_path(users_dir: Path, user_id: str) -> Path:
    """Resolve the file for ``user_id`` and assert it stays inside ``users_dir``.

    This is defence in depth: even though :func:`safe_user_filename` produces a
    flat, safe name, we verify the resolved path is a direct child of the users
    directory and reject anything that would traverse out.
    """

    name = safe_user_filename(user_id)
    # The generated name must be a single, non-traversing path segment.
    if "/" in name or "\\" in name or PurePosixPath(name).name != name \
            or PureWindowsPath(name).name != name:
        raise InvalidUserIdError("user_id produced an unsafe file name")
    base = users_dir.resolve()
    candidate = (base / name).resolve()
    if candidate.parent != base:
        raise InvalidUserIdError("resolved user path escapes the users directory")
    return candidate
