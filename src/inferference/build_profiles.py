"""Build profile names, and the old names that still resolve.

The first Vulkan profiles carried the name of the first cards of the rig
(`vulkan-mi25-...`). The flags never depended on the card, so the profiles now
have hardware-neutral names. A recorded build id or manifest can still carry an
old name. Readers map it to the new name here, so old records stay valid.

Keep `PROFILE_ALIASES` in step with the alias table in
`ci/build/build-llamacpp.sh`.
"""

from __future__ import annotations

__all__ = ["PROFILE_ALIASES", "canonical_build_id", "canonical_profile"]

PROFILE_ALIASES: dict[str, str] = {
    "vulkan-mi25": "vulkan",
    "vulkan-mi25-server": "vulkan-server",
    "vulkan-mi25-p2fork-server": "vulkan-p2fork-server",
}

# Longest old name first: `vulkan-mi25` is a prefix of the other two.
_BY_LENGTH = sorted(PROFILE_ALIASES, key=len, reverse=True)


def canonical_profile(profile: str) -> str:
    """Return the current name of a profile. A current name returns unchanged."""
    return PROFILE_ALIASES.get(profile, profile)


def canonical_build_id(build_id: str) -> str:
    """Return a build id with an old profile prefix replaced by the new one.

    A build id is `<profile>-<commit>[-residency-<hash>]`. Only a whole profile
    prefix is replaced: the old name must end the id or be followed by `-`.
    """
    for old in _BY_LENGTH:
        if build_id == old or build_id.startswith(old + "-"):
            return PROFILE_ALIASES[old] + build_id[len(old) :]
    return build_id
