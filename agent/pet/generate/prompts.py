"""Prompt builders for pet generation.

Two prompt shapes: a *base* prompt (prompt-only, produces the canonical look the
user picks between) and per-*state* *row* prompts (grounded on the chosen base,
produce one horizontal strip of N poses). Prompts stay concise and
sprite-production oriented; the identity lock and "one transparent row" framing
matter more than flowery description.

Hermes drives six states (see :data:`agent.pet.generate.atlas.ROW_SPECS`); these
mirror that set rather than the petdex/Codex nine.
"""

from __future__ import annotations

# What each Hermes state should depict (kept short — these go straight into the
# row prompt). Phrased to avoid the common sprite-gen failure modes (detached
# effects, motion lines, shadows).
STATE_ACTIONS: dict[str, str] = {
    "idle": "a calm idle loop: subtle breathing, a tiny blink or gentle bob, no big gestures",
    "wave": "a friendly greeting: raising a paw/hand/limb to wave, clear up-and-down gesture",
    "run": "focused active work: leaning in, concentrating, busy 'thinking/processing' energy (NOT foot-running)",
    "failed": "a sad or deflated reaction: slumped, dejected, small frown — readable but not noisy",
    "review": "careful inspection: a focused lean, head tilt, studying something intently",
    "jump": "a happy celebration jump: anticipation, lift off the ground, peak, and land",
}

_STYLE_HINTS: dict[str, str] = {
    "auto": "",
    "pixel": " Render in clean pixel-art style.",
    "plush": " Render as a soft plush toy.",
    "clay": " Render as a claymation / soft 3D clay figure.",
    "sticker": " Render as a glossy die-cut sticker.",
    "flat-vector": " Render in flat vector mascot style.",
    "3d-toy": " Render as a glossy 3D toy.",
    "painterly": " Render in a soft painterly style.",
}

_BACKGROUND = (
    "Center one full-body character on a fully transparent background. "
    "No text, no labels, no shadow, no ground line, no scenery, no frame, no border."
)


def style_hint(style: str | None) -> str:
    return _STYLE_HINTS.get((style or "auto").strip().lower(), "")


def build_base_prompt(concept: str, *, style: str | None = "auto") -> str:
    """The base look: a single, clean, centered full-body mascot."""
    concept = (concept or "a cute friendly mascot creature").strip()
    return (
        f"A cute, characterful mascot pet: {concept}. "
        "Compact, whole-body silhouette that reads clearly at small size, "
        "appealing face, simple consistent palette. "
        f"{_BACKGROUND}{style_hint(style)}"
    )


def build_row_prompt(state: str, frame_count: int, concept: str, *, style: str | None = "auto") -> str:
    """A row strip: *frame_count* poses of the SAME character, left→right.

    The attached base image is the identity source of truth; the prompt locks
    species, palette, face, and props to it.
    """
    action = STATE_ACTIONS.get(state, "a simple idle pose")
    concept = (concept or "the mascot").strip()
    return (
        f"Using the attached reference image as the exact same character "
        f"(same species, face, colors, markings, proportions, and props), "
        f"draw a single horizontal strip of {frame_count} animation frames showing {action}. "
        f"The {frame_count} poses must be evenly spaced left to right, each fully separated "
        "(not overlapping), same size and baseline, forming a smooth loop. "
        f"Keep the character identical across all frames. {_BACKGROUND}{style_hint(style)}"
    )
