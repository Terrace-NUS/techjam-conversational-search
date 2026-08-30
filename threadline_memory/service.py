"""Host-facing facade for the JSON-backed long-term shopping memory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.shopping_copilot import to_profile_prior, to_ranking_user_profile
from .errors import InvalidFieldPathError, InvalidSessionIdError
from .llm import ProfileUpdateLLM, RuleBasedProfileUpdateClient
from .merge import delete_field, merge_patch
from .schema import normalize_base_profile
from .store import JsonProfileStore


@dataclass(frozen=True, slots=True)
class MemoryResponse:
    """Uniform result: the profile plus host-ready adapter views."""

    user_profile: dict[str, Any]
    profile_prior: dict[str, Any]
    ranking_user_profile: dict[str, Any]
    changed_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_profile": self.user_profile,
            "profile_prior": self.profile_prior,
            "ranking_user_profile": self.ranking_user_profile,
            "changed_fields": self.changed_fields,
            "warnings": self.warnings,
        }


class MemoryService:
    """Small facade keeping storage and the LLM boundary replaceable."""

    def __init__(
        self,
        store: JsonProfileStore,
        llm: ProfileUpdateLLM | None = None,
    ) -> None:
        self._store = store
        self._llm = llm or RuleBasedProfileUpdateClient()

    @classmethod
    def from_json_directory(
        cls,
        root: str | Path,
        *,
        llm: ProfileUpdateLLM | None = None,
        clock: Callable[[], str] | None = None,
    ) -> MemoryService:
        return cls(JsonProfileStore(root, clock=clock), llm=llm)

    # -- reads ---------------------------------------------------------------

    def get_profile(self, user_id: str) -> dict[str, Any]:
        return self._store.load_or_init(user_id)

    def _respond(
        self,
        profile: dict[str, Any],
        *,
        changed: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> MemoryResponse:
        return MemoryResponse(
            user_profile=profile,
            profile_prior=to_profile_prior(profile),
            ranking_user_profile=to_ranking_user_profile(profile),
            changed_fields=changed or [],
            warnings=warnings or [],
        )

    # -- session lifecycle ---------------------------------------------------

    def start_session(
        self,
        user_id: str,
        session_id: str,
        initial_profile: dict[str, Any] | None = None,
    ) -> MemoryResponse:
        """Open a session: create-or-load the file and merge any dataset prior."""

        _require_session_id(session_id)
        profile = self._store.load_or_init(user_id)
        changed: list[str] = []
        if initial_profile is not None:
            base = profile["base_profile"]
            merged = normalize_base_profile({**base, **initial_profile})
            # Union preference_tags so a returning user never loses prior tags.
            merged["preference_tags"] = list(
                dict.fromkeys([*base.get("preference_tags", []), *merged["preference_tags"]])
            )
            if merged != base:
                profile["base_profile"] = merged
                changed.append("base_profile")
        metadata = profile["metadata"]
        session_ids = metadata.setdefault("session_ids", [])
        if session_id not in session_ids:
            session_ids.append(session_id)
            metadata["session_count"] = int(metadata.get("session_count", 0)) + 1
            changed.append(f"+session:{session_id}")
        if changed:
            metadata["updated_at"] = self._store.now()
            self._store.save(user_id, profile)
        return self._respond(profile, changed=changed)

    def update_from_dialogue(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict[str, str]],
    ) -> MemoryResponse:
        """Extract a patch from the dialogue, merge it, and persist atomically.

        A failing or malformed LLM response never corrupts the file: the patch
        is validated and merged onto the loaded profile, and only a valid
        resulting document is written.
        """

        _require_session_id(session_id)
        profile = self._store.load_or_init(user_id)
        session_ids = profile["metadata"].get("session_ids", [])
        if session_id not in session_ids:
            raise InvalidSessionIdError(
                f"session_id {session_id!r} was not started for user {user_id!r}"
            )
        warnings: list[str] = []
        try:
            patch = self._llm.extract_profile_patch(profile, messages)
        except Exception as error:  # noqa: BLE001 - isolate the untrusted call
            warnings.append(f"llm error: {error}")
            patch = {}

        outcome = merge_patch(profile, patch)
        outcome.warnings = warnings + outcome.warnings
        updated = outcome.profile
        if outcome.changed_fields:
            updated["metadata"]["updated_at"] = self._store.now()
            self._store.save(user_id, updated)
        return self._respond(
            updated,
            changed=outcome.changed_fields,
            warnings=outcome.warnings,
        )

    # -- explicit user control ----------------------------------------------

    def delete_profile_field(self, user_id: str, field_path: str) -> MemoryResponse:
        """Delete a single field/entry addressed by ``field_path``."""

        if type(field_path) is not str or not field_path.strip() or field_path != field_path.strip():
            raise InvalidFieldPathError("field_path must be a non-blank trimmed string")
        profile = self._store.load_or_init(user_id)
        removed = delete_field(profile, field_path)
        changed: list[str] = []
        if removed:
            profile["metadata"]["updated_at"] = self._store.now()
            self._store.save(user_id, profile)
            changed.append(f"-{field_path}")
        warnings = [] if removed else [f"field not found: {field_path}"]
        return self._respond(profile, changed=changed, warnings=warnings)

    def delete_user(self, user_id: str) -> bool:
        """Delete the user's whole profile file. Returns True if one existed."""

        return self._store.delete(user_id)


def _require_session_id(value: object) -> None:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise InvalidSessionIdError("session_id must be a non-blank trimmed string")
