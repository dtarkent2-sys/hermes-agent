"""Tests for pet generation: deterministic atlas ops, store register, orchestration.

No network/API calls — image generation is mocked with synthetic strips so the
whole pipeline (segmentation → compose → validate → register → adopt) is
exercised hermetically.
"""

from __future__ import annotations

import pytest

from agent.pet.generate import atlas

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


def _strip(n_blobs: int, *, transparent: bool = True, bg=(0, 255, 0, 255), size=(208, 208)) -> Image.Image:
    """A horizontal strip with *n_blobs* clearly-separated colored ellipses."""
    w = size[0] * n_blobs
    h = size[1]
    base = (0, 0, 0, 0) if transparent else bg
    img = Image.new("RGBA", (w, h), base)
    draw = ImageDraw.Draw(img)
    for i in range(n_blobs):
        cx = i * size[0] + size[0] // 2
        cy = h // 2
        r = size[0] // 3
        color = (40 + i * 30 % 200, 80, 200 - i * 20 % 180, 255)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    return img


# ───────────────────────── frame extraction ─────────────────────────


def test_extract_strip_frames_transparent_returns_centered_cells():
    frames = atlas.extract_strip_frames(_strip(6), 6)
    assert len(frames) == 6
    for frame in frames:
        assert frame.size == (atlas.CELL_WIDTH, atlas.CELL_HEIGHT)
        # Background corners must be transparent.
        assert frame.getpixel((0, 0))[3] == 0
        # Something is drawn.
        assert frame.getchannel("A").getextrema()[1] > 0


def test_extract_strip_frames_keys_out_solid_background():
    frames = atlas.extract_strip_frames(_strip(4, transparent=False), 4)
    assert len(frames) == 4
    # The green backdrop must be gone (corner transparent).
    assert frames[0].getpixel((0, 0))[3] == 0


def test_extract_strip_frames_slot_fallback_when_unsegmentable():
    # A single connected smear can't be split into 5 components → slot fallback.
    img = Image.new("RGBA", (200 * 5, 208), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((0, 80, 200 * 5 - 1, 120), fill=(200, 50, 50, 255))
    frames = atlas.extract_strip_frames(img, 5, method="auto")
    assert len(frames) == 5


def test_extract_components_method_raises_when_too_few():
    img = Image.new("RGBA", (400, 208), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((10, 10, 100, 100), fill=(255, 0, 0, 255))
    with pytest.raises(ValueError):
        atlas.extract_strip_frames(img, 6, method="components")


# ───────────────────────── atlas compose / validate ─────────────────────────


def _frames_for_all_states() -> dict[str, list]:
    out: dict[str, list] = {}
    for state, _row, count in atlas.ROW_SPECS:
        out[state] = atlas.extract_strip_frames(_strip(count), count)
    return out


def test_compose_atlas_geometry_and_validation():
    sheet = atlas.compose_atlas(_frames_for_all_states())
    assert sheet.size == (atlas.ATLAS_WIDTH, atlas.ATLAS_HEIGHT)
    result = atlas.validate_atlas(sheet)
    assert result["ok"], result["errors"]
    assert set(result["filled_states"]) == {s for s, _, _ in atlas.ROW_SPECS}


def test_compose_atlas_leaves_unused_tail_transparent():
    # wave has 4 frames; columns 4 and 5 of its row must be transparent.
    sheet = atlas.compose_atlas(_frames_for_all_states())
    wave_row = next(r for s, r, _ in atlas.ROW_SPECS if s == "wave")
    top = wave_row * atlas.CELL_HEIGHT
    for col in (4, 5):
        left = col * atlas.CELL_WIDTH
        cell = sheet.crop((left, top, left + atlas.CELL_WIDTH, top + atlas.CELL_HEIGHT))
        assert cell.getchannel("A").getextrema()[1] == 0


def test_validate_atlas_rejects_wrong_size():
    bad = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    result = atlas.validate_atlas(bad)
    assert not result["ok"]
    assert any("expected" in e for e in result["errors"])


def test_validate_atlas_rejects_rgb_residue():
    sheet = atlas.compose_atlas(_frames_for_all_states())
    # Poke a fully-transparent pixel with non-zero RGB.
    sheet.putpixel((0, 0), (120, 0, 0, 0))
    result = atlas.validate_atlas(sheet)
    assert not result["ok"]
    assert any("residue" in e for e in result["errors"])


def test_validate_atlas_warns_on_empty_state():
    frames = _frames_for_all_states()
    frames["jump"] = []
    sheet = atlas.compose_atlas(frames)
    result = atlas.validate_atlas(sheet)
    assert result["ok"]  # one empty row is a warning, not an error
    assert any("jump" in w for w in result["warnings"])


def test_single_frame_fits_cell():
    frame = atlas.single_frame(_strip(1))
    assert frame.size == (atlas.CELL_WIDTH, atlas.CELL_HEIGHT)
    assert frame.getchannel("A").getextrema()[1] > 0


# ───────────────────────── store register / adopt ─────────────────────────


def test_slugify_and_unique_slug():
    from agent.pet import store

    assert store.slugify("My Cool Pet!") == "my-cool-pet"
    assert store.slugify("   ") == "pet"
    first = store.unique_slug("Robo")
    (store.pets_dir() / first).mkdir(parents=True)
    assert store.unique_slug("Robo") == "robo-2"


def test_register_local_pet_appears_and_is_adoptable():
    from agent.pet import store

    sheet = atlas.compose_atlas(_frames_for_all_states())
    pet = store.register_local_pet(sheet, slug="Sparky", display_name="Sparky", description="zappy")
    assert pet.slug == "sparky"
    assert pet.exists
    assert any(p.slug == "sparky" for p in store.installed_pets())

    # install_pet returns the on-disk pet without ever hitting the manifest.
    adopted = store.install_pet("sparky")
    assert adopted.slug == "sparky"
    assert adopted.display_name == "Sparky"


def test_register_local_pet_accepts_bytes():
    from agent.pet import store

    sheet = atlas.compose_atlas(_frames_for_all_states())
    data = atlas.atlas_to_webp_bytes(sheet)
    pet = store.register_local_pet(data, slug="bytey")
    assert pet.exists


# ───────────────────────── orchestration (mocked imagegen) ─────────────────────────


def test_generate_base_drafts_returns_n(monkeypatch, tmp_path):
    from agent.pet.generate import imagegen, orchestrate

    calls = {"n": 0}

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet"):
        paths = []
        for i in range(n):
            calls["n"] += 1
            p = tmp_path / f"{prefix}_{calls['n']}.png"
            _strip(1).save(p)
            paths.append(p)
        return paths

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    drafts = orchestrate.generate_base_drafts("a fox", n=4)
    assert len(drafts) == 4


def test_generate_base_drafts_hardens_opaque_background(monkeypatch, tmp_path):
    """A provider that ignores background=transparent still yields a cutout."""
    from agent.pet.generate import imagegen, orchestrate

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet"):
        # Solid-green backdrop with a blob — i.e. the provider painted a backdrop.
        p = tmp_path / f"{prefix}_opaque.png"
        _strip(1, transparent=False, bg=(0, 255, 0, 255)).save(p)
        return [p]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    drafts = orchestrate.generate_base_drafts("a fox", n=1)
    assert len(drafts) == 1

    with Image.open(drafts[0]) as out:
        rgba = out.convert("RGBA")
    # The keyed backdrop is now transparent (corner pixel fully see-through).
    assert rgba.getpixel((0, 0))[3] == 0
    # The pet blob in the center is still opaque.
    assert rgba.getpixel((rgba.width // 2, rgba.height // 2))[3] > 0


def test_hatch_pet_end_to_end(monkeypatch, tmp_path):
    from agent.pet import store
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate

    base = tmp_path / "base.png"
    _strip(1).save(base)

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet"):
        # Return a synthetic row strip; frame count is inferable from the spec.
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        p = tmp_path / f"{prefix}.png"
        _strip(count).save(p)
        return [p]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    events: list[tuple[str, str]] = []
    result = orchestrate.hatch_pet(
        base_image=base,
        slug="mocky",
        display_name="Mocky",
        description="a test pet",
        concept="a fox",
        on_progress=lambda ev, detail: events.append((ev, detail)),
    )

    assert result.slug == "mocky"
    assert result.validation["ok"]
    assert set(result.states) == {s for s, _, _ in atlas_mod.ROW_SPECS}
    assert ("compose", "") in events
    # The pet is on disk and adoptable.
    assert store.load_pet("mocky").exists


def test_hatch_pet_idle_fallback_when_row_fails(monkeypatch, tmp_path):
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate
    from agent.pet.generate.imagegen import GenerationError

    base = tmp_path / "base.png"
    _strip(1).save(base)

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet"):
        if prefix == "pet_row_idle":
            raise GenerationError("boom")
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        p = tmp_path / f"{prefix}.png"
        _strip(count).save(p)
        return [p]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    result = orchestrate.hatch_pet(base_image=base, slug="fallbacky", concept="a fox")
    assert "idle" in result.states  # filled by the base-image fallback


def test_resolve_provider_errors_without_backend(monkeypatch):
    from agent.pet.generate import imagegen

    monkeypatch.setattr(imagegen, "_discover", lambda: None)
    monkeypatch.setattr("agent.image_gen_registry.get_active_provider", lambda: None)
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: None)

    with pytest.raises(imagegen.GenerationError):
        imagegen.resolve_provider(require_references=True)


def test_generate_retries_without_transparent_background(monkeypatch, tmp_path):
    """A model that rejects background=transparent still produces images."""
    from agent.pet.generate import imagegen

    saved = tmp_path / "img.png"
    _strip(1).save(saved)
    calls: list[dict] = []

    class FakeProvider:
        def generate(self, prompt, **kwargs):
            calls.append(kwargs)
            if kwargs.get("background") == "transparent":
                return {"success": False, "error": "Transparent background is not supported for this model."}
            return {"success": True, "image": str(saved)}

    sprite = imagegen.SpriteProvider(name="openai", provider=FakeProvider(), supports_references=False)

    out = imagegen.generate("a fox", n=2, provider=sprite)
    assert len(out) == 2
    # First variant probes transparent (rejected) then retries opaque; the second
    # variant skips the transparent probe entirely.
    backgrounds = [c.get("background") for c in calls]
    assert backgrounds == ["transparent", None, None]
