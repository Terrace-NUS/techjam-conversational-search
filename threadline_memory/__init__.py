"""Public surface for Threadline long-term shopping memory (JSON-file backed)."""

from .adapters.shopping_copilot import (
    to_profile_prior,
    to_ranking_user_profile,
    to_reset_request,
)
from .errors import (
    InvalidFieldPathError,
    InvalidPatchError,
    InvalidProfileError,
    InvalidSessionIdError,
    InvalidUserIdError,
    MemoryError,
)
from .llm import (
    DeepSeekProfileUpdateClient,
    FakeProfileUpdateClient,
    ProfileUpdateLLM,
    RuleBasedProfileUpdateClient,
)
from .merge import delete_field, merge_patch
from .paths import safe_user_filename, user_file_path
from .schema import (
    PROFILE_SCHEMA,
    PROFILE_VERSION,
    RANKING_PROFILE_SCHEMA,
    RANKING_PROFILE_VERSION,
    empty_profile,
    normalize_base_profile,
)
from .service import MemoryResponse, MemoryService
from .store import JsonProfileStore

__all__ = [
    "PROFILE_SCHEMA",
    "PROFILE_VERSION",
    "RANKING_PROFILE_SCHEMA",
    "RANKING_PROFILE_VERSION",
    "DeepSeekProfileUpdateClient",
    "FakeProfileUpdateClient",
    "InvalidFieldPathError",
    "InvalidPatchError",
    "InvalidProfileError",
    "InvalidSessionIdError",
    "InvalidUserIdError",
    "JsonProfileStore",
    "MemoryError",
    "MemoryResponse",
    "MemoryService",
    "ProfileUpdateLLM",
    "RuleBasedProfileUpdateClient",
    "delete_field",
    "empty_profile",
    "merge_patch",
    "normalize_base_profile",
    "safe_user_filename",
    "to_profile_prior",
    "to_ranking_user_profile",
    "to_reset_request",
    "user_file_path",
]
