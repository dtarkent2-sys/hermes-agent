"""Pet generation orchestration — the base-draft → hatch flow.

Two steps, mirroring the UX across every surface:

1. :func:`generate_base_drafts` — a handful of prompt-only "what should this pet
   look like" variants. Cheap; the user picks one (or retries for a fresh set).
2. :func:`hatch_pet` — takes the chosen base and generates one grounded row
   strip per Hermes state, slices each into frames, composes the atlas, validates
   it, and writes the pet into the store.

Splitting it this way bounds cost (4 cheap base calls per round; the ~6 row
calls happen once, on the pet you actually keep) and gives each UI a natural
preview/loading point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.pet.generate import atlas, imagegen, prompts
from agent.pet.generate.imagegen import GenerationError, SpriteProvider

logger = logging.getLogger(__name__)

# (event, detail) — e.g. ("row", "idle"), ("compose", ""), ("save", "<slug>").
ProgressFn = Callable[[str, str], None]


@dataclass(frozen=True)
class HatchResult:
    """Outcome of a successful :func:`hatch_pet`."""

    slug: str
    display_name: str
    spritesheet: Path
    states: list[str]
    validation: dict


def _harden_transparency(path: Path) -> Path:
    """Key out any solid backdrop the provider painted; save as an RGBA PNG.

    ``background=transparent`` is requested on every call, but image models honor
    it inconsistently — some still paint a flat (often near-white) backdrop. We
    run the same chroma-key pass the row extractor uses so every base draft the
    user picks between (and the reference the rows are grounded on) is a clean
    cutout. Best-effort: a decode failure leaves the original untouched.
    """
    from PIL import Image

    try:
        with Image.open(path) as opened:
            keyed = atlas.remove_background(opened.convert("RGBA"))
        out = path.with_suffix(".png")
        keyed.save(out, format="PNG")
        return out
    except Exception as exc:  # noqa: BLE001 - cosmetic; fall back to the raw image
        logger.debug("base draft transparency hardening failed for %s: %s", path, exc)
        return path


def generate_base_drafts(
    concept: str,
    *,
    n: int = 4,
    style: str = "auto",
    provider: SpriteProvider | None = None,
) -> list[Path]:
    """Generate *n* candidate base looks for *concept*; returns image paths.

    Each draft is hardened to a transparent cutout (see :func:`_harden_transparency`).
    """
    prompt = prompts.build_base_prompt(concept, style=style)
    sprite = provider or imagegen.resolve_provider(require_references=False)
    raw = imagegen.generate(prompt, n=n, provider=sprite, prefix="pet_base")
    return [_harden_transparency(p) for p in raw]


def hatch_pet(
    *,
    base_image: str | Path,
    slug: str,
    display_name: str = "",
    description: str = "",
    concept: str = "",
    style: str = "auto",
    on_progress: ProgressFn | None = None,
    provider: SpriteProvider | None = None,
) -> HatchResult:
    """Turn an approved base image into a full, installed Hermes pet.

    Generates a grounded row strip per state, extracts frames, composes +
    validates the atlas, and registers it. The idle row falls back to the base
    look so the pet always renders. Raises :class:`GenerationError` on failure.
    """
    base = Path(base_image)
    if not base.is_file():
        raise GenerationError(f"base image not found: {base}")

    sprite = provider or imagegen.resolve_provider(require_references=True)
    progress = on_progress or (lambda *_: None)
    label = concept or display_name or slug

    frames_by_state: dict[str, list] = {}
    for state, _row, count in atlas.ROW_SPECS:
        progress("row", state)
        row_prompt = prompts.build_row_prompt(state, count, label, style=style)
        try:
            strips = imagegen.generate(
                row_prompt,
                n=1,
                reference_images=[base],
                provider=sprite,
                prefix=f"pet_row_{state}",
            )
            frames_by_state[state] = atlas.extract_strip_frames(strips[0], count, method="auto")
        except Exception as exc:  # noqa: BLE001 - a single row may fail; keep going
            logger.warning("pet row '%s' failed: %s", state, exc)

    # Idle is the resting state the renderer falls back to — guarantee it.
    if not frames_by_state.get("idle"):
        progress("row", "idle-fallback")
        frames_by_state["idle"] = [atlas.single_frame(base)]

    progress("compose", "")
    sheet = atlas.compose_atlas(frames_by_state)
    validation = atlas.validate_atlas(sheet)
    if not validation["ok"]:
        raise GenerationError("; ".join(validation["errors"]) or "atlas validation failed")

    from agent.pet import store

    progress("save", slug)
    pet = store.register_local_pet(
        sheet,
        slug=slug,
        display_name=display_name or slug,
        description=description,
    )
    return HatchResult(
        slug=pet.slug,
        display_name=pet.display_name,
        spritesheet=pet.spritesheet,
        states=validation["filled_states"],
        validation=validation,
    )
