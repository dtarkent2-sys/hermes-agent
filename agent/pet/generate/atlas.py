"""Deterministic spritesheet assembly — generated row strips → Hermes atlas.

Image-generation models are good at *drawing* a row of poses but bad at exact
grid geometry, so the model never owns the atlas layout: it produces one loose
horizontal strip per state, and these deterministic ops slice that strip into
clean, centered, transparent ``192x208`` cells and pack them into the sheet our
renderer reads.

The atlas is **Hermes-native**, not the petdex/Codex format. Our renderer
(:mod:`agent.pet.render`) keys frames as ``rows = states, cols = frames`` using
:data:`agent.pet.constants.STATE_ROWS`, so we emit exactly the six states the
engine drives — idle, wave, run, failed, review, jump — left-packed with
trailing transparent cells (which the renderer trims). Sheet is
``COLUMNS*192 x ROWS*208`` (1152x1248).

The frame-segmentation, fit-to-cell, and transparency-residue logic is adapted
from OpenAI's ``hatch-pet`` skill (openai/skills, Apache-2.0).
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path

from agent.pet.constants import FRAME_H, FRAME_W

logger = logging.getLogger(__name__)

CELL_WIDTH = FRAME_W
CELL_HEIGHT = FRAME_H

# (state, row index, frame count). Order/row indices MUST match
# ``STATE_ROWS`` so the renderer crops the right row for each driven state.
# Frame counts are the petdex-ish per-state lengths; the renderer trims any
# trailing blank columns, so rows shorter than ``COLUMNS`` just leave the tail
# transparent.
ROW_SPECS: list[tuple[str, int, int]] = [
    ("idle", 0, 6),
    ("wave", 1, 4),
    ("run", 2, 6),
    ("failed", 3, 6),
    ("review", 4, 6),
    ("jump", 5, 5),
]

ROWS = len(ROW_SPECS)
COLUMNS = max(count for _, _, count in ROW_SPECS)
ATLAS_WIDTH = COLUMNS * CELL_WIDTH
ATLAS_HEIGHT = ROWS * CELL_HEIGHT

FRAME_COUNTS: dict[str, int] = {state: count for state, _, count in ROW_SPECS}

# Alpha at/below which a pixel is "background" for component detection.
_ALPHA_FLOOR = 16
# Cell padding kept around a fitted sprite so poses never touch the edge.
_CELL_PAD = 10


# ───────────────────────── background removal ─────────────────────────


def _color_distance(r: int, g: int, b: int, key: tuple[int, int, int]) -> float:
    return math.sqrt((r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2)


def _has_transparency(image) -> bool:
    """True if the strip already carries a real alpha background."""
    extrema = image.getchannel("A").getextrema()
    # Min alpha 0 somewhere and a meaningful share of fully-transparent pixels.
    if extrema[0] > _ALPHA_FLOOR:
        return False
    hist = image.getchannel("A").histogram()
    transparent = sum(hist[: _ALPHA_FLOOR + 1])
    total = image.width * image.height
    return transparent > total * 0.05


def _dominant_corner_color(image) -> tuple[int, int, int]:
    """Sample the four corners and return the most common opaque color."""
    from collections import Counter

    w, h = image.width, image.height
    px = image.load()
    counter: Counter = Counter()
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        r, g, b, a = px[x, y]
        if a > _ALPHA_FLOOR:
            counter[(r, g, b)] += 1
    if not counter:
        return (0, 255, 0)
    return counter.most_common(1)[0][0]


def remove_background(image, *, chroma_key: tuple[int, int, int] | None = None, threshold: float = 110.0):
    """Return *image* (RGBA) with its flat background keyed out to transparent.

    If the strip already has a transparent background we leave it alone; else we
    key out *chroma_key* (or the dominant corner color when not given). This
    handles both providers that emit transparency natively and those that paint
    a solid backdrop.
    """
    rgba = image.convert("RGBA")
    if _has_transparency(rgba):
        return rgba

    key = chroma_key or _dominant_corner_color(rgba)
    px = rgba.load()
    for y in range(rgba.height):
        for x in range(rgba.width):
            r, g, b, a = px[x, y]
            if a > _ALPHA_FLOOR and _color_distance(r, g, b, key) <= threshold:
                px[x, y] = (0, 0, 0, 0)
    return rgba


# ───────────────────────── frame extraction ─────────────────────────


def _fit_to_cell(image):
    """Crop to content, scale to fit a padded cell, and center on transparent."""
    from PIL import Image

    target = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    bbox = image.getbbox()
    if bbox is None:
        return target

    sprite = image.crop(bbox)
    max_w = CELL_WIDTH - _CELL_PAD
    max_h = CELL_HEIGHT - _CELL_PAD
    scale = min(max_w / sprite.width, max_h / sprite.height, 1.0)
    if scale != 1.0:
        sprite = sprite.resize(
            (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))),
            Image.Resampling.LANCZOS,
        )
    left = (CELL_WIDTH - sprite.width) // 2
    top = (CELL_HEIGHT - sprite.height) // 2
    target.alpha_composite(sprite, (left, top))
    return target


def _connected_components(image) -> list[dict]:
    """Flood-fill the alpha mask into connected blobs (4-connectivity)."""
    alpha = image.getchannel("A")
    w, h = image.size
    data = alpha.tobytes()
    visited = bytearray(w * h)
    out: list[dict] = []

    for start, a in enumerate(data):
        if a <= _ALPHA_FLOOR or visited[start]:
            continue
        stack = [start]
        visited[start] = 1
        pixels: list[int] = []
        min_x = w
        min_y = h
        max_x = 0
        max_y = 0
        while stack:
            cur = stack.pop()
            pixels.append(cur)
            x = cur % w
            y = cur // w
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            for nb, ok in (
                (cur - 1, x > 0),
                (cur + 1, x + 1 < w),
                (cur - w, y > 0),
                (cur + w, y + 1 < h),
            ):
                if ok and not visited[nb] and data[nb] > _ALPHA_FLOOR:
                    visited[nb] = 1
                    stack.append(nb)
        out.append(
            {
                "pixels": pixels,
                "area": len(pixels),
                "bbox": (min_x, min_y, max_x + 1, max_y + 1),
                "center_x": (min_x + max_x + 1) / 2,
            }
        )
    return out


def _group_image(source, components: list[dict], padding: int = 4):
    from PIL import Image

    w, h = source.size
    min_x = max(0, min(c["bbox"][0] for c in components) - padding)
    min_y = max(0, min(c["bbox"][1] for c in components) - padding)
    max_x = min(w, max(c["bbox"][2] for c in components) + padding)
    max_y = min(h, max(c["bbox"][3] for c in components) + padding)

    out = Image.new("RGBA", (max_x - min_x, max_y - min_y), (0, 0, 0, 0))
    src_px = source.load()
    out_px = out.load()
    for c in components:
        for idx in c["pixels"]:
            x = idx % w
            y = idx // w
            out_px[x - min_x, y - min_y] = src_px[x, y]
    return out


def _component_frames(strip, frame_count: int) -> list | None:
    """Segment a strip into *frame_count* sprites by connected components.

    Picks the ``frame_count`` largest blobs as seeds (left→right), attaches
    smaller blobs to the nearest seed, and returns one fitted cell per group.
    Returns ``None`` when it can't find enough distinct sprites (caller falls
    back to equal slicing).
    """
    components = _connected_components(strip)
    if not components:
        return None

    largest = max(c["area"] for c in components)
    seed_threshold = max(120, largest * 0.20)
    seeds = [c for c in components if c["area"] >= seed_threshold]
    if len(seeds) < frame_count:
        seeds = sorted(components, key=lambda c: c["area"], reverse=True)[:frame_count]
    if len(seeds) < frame_count:
        return None

    seeds = sorted(
        sorted(seeds, key=lambda c: c["area"], reverse=True)[:frame_count],
        key=lambda c: c["center_x"],
    )
    seed_ids = {id(s) for s in seeds}
    groups: list[list[dict]] = [[s] for s in seeds]
    noise_threshold = max(12, largest * 0.002)
    for c in components:
        if id(c) in seed_ids or c["area"] < noise_threshold:
            continue
        nearest = min(range(len(seeds)), key=lambda i: abs(seeds[i]["center_x"] - c["center_x"]))
        groups[nearest].append(c)

    return [_fit_to_cell(_group_image(strip, g)) for g in groups]


def _slot_frames(strip, frame_count: int) -> list:
    """Fallback: slice the strip into *frame_count* equal columns."""
    slot = strip.width / frame_count
    frames = []
    for i in range(frame_count):
        left = round(i * slot)
        right = round((i + 1) * slot)
        frames.append(_fit_to_cell(strip.crop((left, 0, right, strip.height))))
    return frames


def extract_strip_frames(
    strip,
    frame_count: int,
    *,
    chroma_key: tuple[int, int, int] | None = None,
    method: str = "auto",
) -> list:
    """Turn one generated row strip into *frame_count* clean 192x208 cells.

    *strip* is a PIL image (or path). Background is keyed out, then frames are
    found by connected components (``auto``) with an equal-slot fallback.
    """
    from PIL import Image

    if isinstance(strip, (str, Path)):
        with Image.open(strip) as opened:
            strip = opened.convert("RGBA")
    else:
        strip = strip.convert("RGBA")

    strip = remove_background(strip, chroma_key=chroma_key)

    if method in ("auto", "components"):
        frames = _component_frames(strip, frame_count)
        if frames is not None:
            return frames
        if method == "components":
            raise ValueError(f"could not segment {frame_count} sprites from strip")
    return _slot_frames(strip, frame_count)


# ───────────────────────── atlas composition ─────────────────────────


def single_frame(image):
    """One fitted 192x208 cell from a standalone image (e.g. the base look).

    Used as an idle fallback so a pet always renders even if the idle row
    generation failed.
    """
    from PIL import Image

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGBA")
    return _fit_to_cell(remove_background(image))


def _clear_transparent_rgb(image):
    """Zero the RGB of fully-transparent pixels (no colored-halo residue)."""
    from PIL import Image

    rgba = image.convert("RGBA")
    data = bytearray(rgba.tobytes())
    for i in range(0, len(data), 4):
        if data[i + 3] == 0:
            data[i] = data[i + 1] = data[i + 2] = 0
    return Image.frombytes("RGBA", rgba.size, bytes(data))


def compose_atlas(frames_by_state: dict[str, list]):
    """Pack per-state frame lists into the Hermes atlas (RGBA, residue-cleared).

    Missing/short states leave their trailing cells transparent; extra frames
    beyond a state's spec are dropped.
    """
    from PIL import Image

    atlas = Image.new("RGBA", (ATLAS_WIDTH, ATLAS_HEIGHT), (0, 0, 0, 0))
    for state, row, count in ROW_SPECS:
        frames = frames_by_state.get(state) or []
        for col, frame in enumerate(frames[:count]):
            cell = frame.convert("RGBA")
            if cell.size != (CELL_WIDTH, CELL_HEIGHT):
                cell = _fit_to_cell(cell)
            atlas.alpha_composite(cell, (col * CELL_WIDTH, row * CELL_HEIGHT))
    return _clear_transparent_rgb(atlas)


def atlas_to_webp_bytes(atlas) -> bytes:
    """Encode an atlas image to lossless WebP bytes (the on-disk pet format)."""
    buf = io.BytesIO()
    atlas.save(buf, format="WEBP", lossless=True, quality=100, method=6, exact=True)
    return buf.getvalue()


def validate_atlas(atlas) -> dict:
    """Check geometry, per-cell occupancy, and transparency invariants.

    Returns ``{ok, width, height, errors, warnings, filled_states}``. Errors are
    blockers (wrong size, empty used cell, opaque/dirty transparency); warnings
    are soft (a whole state row blank — generation likely dropped a row).
    """
    from PIL import Image

    if isinstance(atlas, (str, Path)):
        with Image.open(atlas) as opened:
            atlas = opened.convert("RGBA")
    else:
        atlas = atlas.convert("RGBA")

    errors: list[str] = []
    warnings: list[str] = []

    if atlas.size != (ATLAS_WIDTH, ATLAS_HEIGHT):
        errors.append(f"expected {ATLAS_WIDTH}x{ATLAS_HEIGHT}, got {atlas.width}x{atlas.height}")
        return {"ok": False, "width": atlas.width, "height": atlas.height, "errors": errors, "warnings": warnings, "filled_states": []}

    filled_states: list[str] = []
    for state, row, count in ROW_SPECS:
        row_pixels = 0
        for col in range(count):
            left = col * CELL_WIDTH
            top = row * CELL_HEIGHT
            cell = atlas.crop((left, top, left + CELL_WIDTH, top + CELL_HEIGHT))
            nonblank = sum(cell.getchannel("A").histogram()[1:])
            row_pixels += nonblank
        if row_pixels > 0:
            filled_states.append(state)
        else:
            warnings.append(f"state '{state}' has no frames")

    if not filled_states:
        errors.append("atlas is empty — no state produced any frames")

    # Transparent pixels must carry zero RGB (no halo residue).
    data = atlas.tobytes()
    residue = 0
    for i in range(0, len(data), 4):
        if data[i + 3] == 0 and (data[i] or data[i + 1] or data[i + 2]):
            residue += 1
    if residue:
        errors.append(f"{residue} transparent pixels retain RGB residue")

    return {
        "ok": not errors,
        "width": atlas.width,
        "height": atlas.height,
        "errors": errors,
        "warnings": warnings,
        "filled_states": filled_states,
    }
