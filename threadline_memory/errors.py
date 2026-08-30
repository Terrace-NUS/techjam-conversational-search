"""Exception types raised by the memory module."""

from __future__ import annotations


class MemoryError(Exception):
    """Base class for all memory-module errors."""


class InvalidUserIdError(MemoryError):
    """A user id is empty, untrimmed, or otherwise unusable."""


class InvalidSessionIdError(MemoryError):
    """A session id is invalid or the session was not started for this user."""


class InvalidFieldPathError(MemoryError):
    """A profile field path is empty, untrimmed, or otherwise unusable."""


class InvalidProfileError(MemoryError):
    """A stored profile file is missing, unreadable, or has an unexpected shape."""


class InvalidPatchError(MemoryError):
    """A profile patch could not be validated. Reserved for strict callers.

    The default merge path never raises this — it drops invalid entries and
    records warnings instead, so a bad LLM response cannot corrupt the file.
    """
