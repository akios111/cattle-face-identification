from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class AugmentationSpec:
    identifier: str
    family: str
    params: dict[str, Any]


_ALL_SPECS: tuple[AugmentationSpec, ...] = (
    AugmentationSpec("original", "original", {}),
    AugmentationSpec("flip_horizontal", "geometric", {"op": "flip_horizontal"}),
    AugmentationSpec("flip_vertical", "geometric", {"op": "flip_vertical"}),
    AugmentationSpec("rotate_neg_15", "geometric", {"op": "rotate", "degrees": -15}),
    AugmentationSpec("rotate_pos_15", "geometric", {"op": "rotate", "degrees": 15}),
    AugmentationSpec("brightness_up", "photometric", {"op": "brightness", "factor": 1.25}),
    AugmentationSpec("brightness_down", "photometric", {"op": "brightness", "factor": 0.75}),
    AugmentationSpec("contrast_up", "photometric", {"op": "contrast", "factor": 1.25}),
    AugmentationSpec("contrast_down", "photometric", {"op": "contrast", "factor": 0.75}),
    AugmentationSpec("gaussian_noise", "photometric", {"op": "gaussian_noise", "sigma": 12.0}),
    AugmentationSpec("blur", "photometric", {"op": "blur", "radius": 1.4}),
    AugmentationSpec("sharpen", "photometric", {"op": "sharpen", "factor": 1.8}),
    AugmentationSpec("translate_left", "geometric", {"op": "translate", "x": -0.08, "y": 0.0}),
    AugmentationSpec("translate_right", "geometric", {"op": "translate", "x": 0.08, "y": 0.0}),
    AugmentationSpec("zoom_in", "geometric", {"op": "zoom", "factor": 1.12}),
    AugmentationSpec("zoom_out", "geometric", {"op": "zoom", "factor": 0.88}),
    AugmentationSpec("shear_x", "geometric", {"op": "shear", "x": 0.10, "y": 0.0}),
    AugmentationSpec("shear_y", "geometric", {"op": "shear", "x": 0.0, "y": 0.10}),
    AugmentationSpec("cutout_center", "cutout", {"op": "cutout", "size": 0.25, "mode": "center"}),
    AugmentationSpec("cutout_random", "cutout", {"op": "cutout", "size": 0.25, "mode": "random"}),
)


def get_augmentation_specs(profile: str = "all") -> list[AugmentationSpec]:
    profile = profile.lower()
    if profile == "all":
        return list(_ALL_SPECS)
    if profile == "none":
        return [spec for spec in _ALL_SPECS if spec.family == "original"]
    if profile == "geometric":
        return [spec for spec in _ALL_SPECS if spec.family in {"original", "geometric"}]
    if profile == "photometric":
        return [spec for spec in _ALL_SPECS if spec.family in {"original", "photometric"}]
    if profile == "no_cutout":
        return [spec for spec in _ALL_SPECS if spec.family != "cutout"]
    if profile == "cutout_only":
        return [spec for spec in _ALL_SPECS if spec.family in {"original", "cutout"}]
    if profile in {"all+cutout", "all_cutout"}:
        return list(_ALL_SPECS)
    raise ValueError(f"Unknown augmentation profile: {profile}")


def _rng_for(
    seed: int,
    identifier: str,
    *,
    source_id: str | None = None,
    protocol_version: str = "legacy",
) -> np.random.Generator:
    if protocol_version == "legacy":
        key = identifier
    elif protocol_version == "hardening_v2":
        if not source_id or not source_id.strip():
            raise ValueError("hardening_v2 augmentation requires a non-empty source_id")
        key = f"hardening_v2\x1f{source_id}\x1f{identifier}"
    else:
        raise ValueError(f"Unknown augmentation protocol version: {protocol_version}")
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)
    return np.random.default_rng(seed + offset)


def _translate(image: Image.Image, x_frac: float, y_frac: float) -> Image.Image:
    width, height = image.size
    x_shift = int(round(width * x_frac))
    y_shift = int(round(height * y_frac))
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, x_shift, 0, 1, y_shift),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def _zoom(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    if factor == 1:
        return image.copy()
    if factor > 1:
        crop_w = max(1, int(round(width / factor)))
        crop_h = max(1, int(round(height / factor)))
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize(image.size, Image.Resampling.BILINEAR)

    new_w = max(1, int(round(width * factor)))
    new_h = max(1, int(round(height * factor)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", image.size, (0, 0, 0))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def _shear(image: Image.Image, x: float, y: float) -> Image.Image:
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, x, 0, y, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=(0, 0, 0),
    )


def _gaussian_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    pixels = np.asarray(image).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=sigma, size=pixels.shape)
    noisy = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _cutout(image: Image.Image, size_frac: float, mode: str, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    cut_w = max(1, int(round(width * size_frac)))
    cut_h = max(1, int(round(height * size_frac)))
    if mode == "center":
        left = (width - cut_w) // 2
        top = (height - cut_h) // 2
    else:
        left = int(rng.integers(0, max(1, width - cut_w + 1)))
        top = int(rng.integers(0, max(1, height - cut_h + 1)))
    output = image.copy()
    patch = Image.new("RGB", (cut_w, cut_h), (0, 0, 0))
    output.paste(patch, (left, top))
    return output


def apply_augmentation(
    image: Image.Image,
    spec: AugmentationSpec,
    seed: int = 2026,
    *,
    source_id: str | None = None,
    protocol_version: str = "legacy",
) -> Image.Image:
    image = image.convert("RGB")
    op = spec.params.get("op", "original")
    rng = _rng_for(
        seed,
        spec.identifier,
        source_id=source_id,
        protocol_version=protocol_version,
    )

    if op == "original":
        return image.copy()
    if op == "flip_horizontal":
        return ImageOps.mirror(image)
    if op == "flip_vertical":
        return ImageOps.flip(image)
    if op == "rotate":
        return image.rotate(
            spec.params["degrees"],
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
    if op == "brightness":
        return ImageEnhance.Brightness(image).enhance(spec.params["factor"])
    if op == "contrast":
        return ImageEnhance.Contrast(image).enhance(spec.params["factor"])
    if op == "gaussian_noise":
        return _gaussian_noise(image, spec.params["sigma"], rng)
    if op == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=spec.params["radius"]))
    if op == "sharpen":
        return ImageEnhance.Sharpness(image).enhance(spec.params["factor"])
    if op == "translate":
        return _translate(image, spec.params["x"], spec.params["y"])
    if op == "zoom":
        return _zoom(image, spec.params["factor"])
    if op == "shear":
        return _shear(image, spec.params["x"], spec.params["y"])
    if op == "cutout":
        return _cutout(image, spec.params["size"], spec.params["mode"], rng)
    raise ValueError(f"Unsupported augmentation operation: {op}")


def resize_image(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
