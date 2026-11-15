"""The old `vulkan-mi25-...` profile names still resolve to the hardware-neutral names."""

import pytest

from inferference.build_profiles import PROFILE_ALIASES, canonical_build_id, canonical_profile


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("vulkan-mi25", "vulkan"),
        ("vulkan-mi25-server", "vulkan-server"),
        ("vulkan-mi25-p2fork-server", "vulkan-p2fork-server"),
    ],
)
def test_an_old_profile_maps_to_its_new_name(old: str, new: str) -> None:
    assert canonical_profile(old) == new
    assert canonical_profile(new) == new


def test_a_profile_with_no_alias_is_unchanged() -> None:
    assert canonical_profile("p2fork-server") == "p2fork-server"
    assert canonical_profile("vulkan-p2fork") == "vulkan-p2fork"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "vulkan-mi25-p2fork-server-ecc0cb0d3-residency-d916a61ea4d3",
            "vulkan-p2fork-server-ecc0cb0d3-residency-d916a61ea4d3",
        ),
        ("vulkan-mi25-server-ecc0cb0d3", "vulkan-server-ecc0cb0d3"),
        ("vulkan-mi25-ecc0cb0d3", "vulkan-ecc0cb0d3"),
        ("vulkan-mi25", "vulkan"),
    ],
)
def test_an_old_build_id_maps_to_the_new_prefix(old: str, new: str) -> None:
    assert canonical_build_id(old) == new
    assert canonical_build_id(new) == new


def test_a_build_id_with_only_a_look_alike_prefix_is_unchanged() -> None:
    assert canonical_build_id("vulkan-mi250-ecc0cb0d3") == "vulkan-mi250-ecc0cb0d3"
    assert canonical_build_id("fake-build-123") == "fake-build-123"


def test_the_alias_table_matches_the_build_script() -> None:
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "ci" / "build" / "build-llamacpp.sh").read_text()
    for old, new in PROFILE_ALIASES.items():
        assert f"{old})" in script
        assert f'new_profile="{new}"' in script
