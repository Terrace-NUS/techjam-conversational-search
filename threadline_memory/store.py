"""Lightweight JSON-file store: one file per user, atomic writes, auto-init.

There is no database. Each user's profile lives at
``<root>/users/<safe_user_id>.json``. Writes go through a temp file plus
``os.replace`` so a crash mid-write can never leave a half-written profile: the
previous file stays fully readable until the atomic rename completes.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import InvalidProfileError
from .paths import user_file_path
from .schema import PROFILE_SCHEMA, empty_profile


def _default_clock() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JsonProfileStore:
    """Per-user JSON persistence with safe file names and atomic replacement."""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._root = Path(root)
        self._users_dir = self._root / "users"
        self._clock = clock or _default_clock

    @property
    def root(self) -> Path:
        return self._root

    @property
    def users_dir(self) -> Path:
        return self._users_dir

    def now(self) -> str:
        return self._clock()

    def path_for(self, user_id: str) -> Path:
        self._users_dir.mkdir(parents=True, exist_ok=True)
        return user_file_path(self._users_dir, user_id)

    def exists(self, user_id: str) -> bool:
        self._users_dir.mkdir(parents=True, exist_ok=True)
        return user_file_path(self._users_dir, user_id).exists()

    def load(self, user_id: str) -> dict[str, Any] | None:
        """Return the stored profile, or ``None`` if the user has no file."""

        path = self.path_for(user_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InvalidProfileError(f"cannot read profile for {user_id}: {error}") from error
        if not isinstance(data, dict) or data.get("schema") != PROFILE_SCHEMA:
            raise InvalidProfileError(f"profile for {user_id} has an unexpected shape")
        return data

    def load_or_init(self, user_id: str) -> dict[str, Any]:
        """Return the stored profile, creating and persisting one if absent."""

        existing = self.load(user_id)
        if existing is not None:
            return existing
        fresh = empty_profile(user_id, created_at=self.now())
        self.save(user_id, fresh)
        return fresh

    def save(self, user_id: str, profile: dict[str, Any]) -> None:
        """Atomically write ``profile`` for ``user_id``.

        The document is serialized to a temp file in the same directory and then
        ``os.replace``d onto the target, which is atomic on the same filesystem.
        The old file is never truncated in place.
        """

        if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
            raise InvalidProfileError("refusing to save a non-profile document")
        path = self.path_for(user_id)
        payload = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=False)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def delete(self, user_id: str) -> bool:
        """Delete a user's file. Returns ``True`` if a file was removed."""

        path = self.path_for(user_id)
        if not path.exists():
            return False
        path.unlink()
        return True
