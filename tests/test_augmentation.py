import numpy as np
from PIL import Image
import pytest

from cattle_id.augmentation import apply_augmentation, get_augmentation_specs


def test_all_profile_contains_original_plus_19_deterministic_variants():
    specs = get_augmentation_specs("all")

    assert len(specs) == 20
    assert specs[0].identifier == "original"
    assert len({spec.identifier for spec in specs}) == 20


def test_ablation_profiles_select_cutout_families_explicitly():
    no_cutout = get_augmentation_specs("no_cutout")
    cutout_only = get_augmentation_specs("cutout_only")

    assert {spec.family for spec in no_cutout} == {"original", "geometric", "photometric"}
    assert {spec.family for spec in cutout_only} == {"original", "cutout"}
    assert {spec.identifier for spec in cutout_only} == {
        "original",
        "cutout_center",
        "cutout_random",
    }


def test_augmentations_preserve_rgb_size_and_are_deterministic():
    image = Image.new("RGB", (32, 32), (120, 60, 30))

    for spec in get_augmentation_specs("all"):
        first = apply_augmentation(image, spec, seed=17)
        second = apply_augmentation(image, spec, seed=17)

        assert first.mode == "RGB"
        assert first.size == image.size
        assert np.array_equal(np.asarray(first), np.asarray(second))


def test_hardening_v2_rng_is_deterministic_per_source_and_differs_between_sources():
    image = Image.new("RGB", (32, 32), (100, 110, 120))
    noise = next(spec for spec in get_augmentation_specs("all") if spec.identifier == "gaussian_noise")

    first = apply_augmentation(
        image,
        noise,
        seed=1,
        source_id="cow_001.jpg",
        protocol_version="hardening_v2",
    )
    repeated = apply_augmentation(
        image,
        noise,
        seed=1,
        source_id="cow_001.jpg",
        protocol_version="hardening_v2",
    )
    second_source = apply_augmentation(
        image,
        noise,
        seed=1,
        source_id="cow_002.jpg",
        protocol_version="hardening_v2",
    )

    assert np.array_equal(np.asarray(first), np.asarray(repeated))
    assert not np.array_equal(np.asarray(first), np.asarray(second_source))


def test_hardening_v2_rng_rejects_missing_source_identifier():
    image = Image.new("RGB", (16, 16), (0, 0, 0))
    noise = next(spec for spec in get_augmentation_specs("all") if spec.identifier == "gaussian_noise")

    with pytest.raises(ValueError, match="source_id"):
        apply_augmentation(image, noise, seed=1, protocol_version="hardening_v2")


def test_cutout_places_zeroed_patch_without_changing_shape():
    image = Image.new("RGB", (32, 32), (120, 60, 30))
    cutout_spec = next(
        spec for spec in get_augmentation_specs("all") if spec.identifier == "cutout_center"
    )

    output = apply_augmentation(image, cutout_spec, seed=17)
    pixels = np.asarray(output)

    assert pixels.shape == (32, 32, 3)
    assert (pixels == 0).all(axis=2).sum() > 0
