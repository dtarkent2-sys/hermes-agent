import { atom } from 'nanostores'

import { type PetInfo } from '@/store/pet'
import { type GatewayRequest, loadPetGallery } from '@/store/pet-gallery'

/**
 * Feature store for the "generate a pet" flow (Cmd-K → Pets → Generate).
 *
 * Three backend steps, mirrored as state here:
 *  - `pet.generate` produces N cheap base-look *drafts* keyed by a `token`.
 *  - `pet.hatch` turns the chosen draft into a full animated pet — installed but
 *    NOT active — and returns its renderer payload so we can preview all frames.
 *  - the user then *adopts* (`pet.select`) or *discards* (`pet.remove`) it.
 *
 * The store owns the draft set, the selected variant, the hatched preview, and
 * the busy/error status so the page is a thin view. Retry == regenerate (new
 * token). Kept separate from `pet-gallery` because its lifecycle (ephemeral
 * drafts + an unadopted preview) is unrelated to the long-lived gallery cache.
 */

export interface PetDraft {
  index: number
  /** Downscaled PNG data URI preview from the gateway. */
  dataUri: string
}

export type PetGenStatus =
  | 'idle'
  | 'generating'
  | 'ready'
  | 'hatching'
  | 'preview'
  | 'adopting'
  | 'error'
  | 'stale'

export const $petGenStatus = atom<PetGenStatus>('idle')
export const $petGenError = atom<string | null>(null)
export const $petGenToken = atom<string | null>(null)
export const $petGenDrafts = atom<PetDraft[]>([])
export const $petGenSelected = atom<number | null>(null)
/** The hatched-but-unadopted pet: its renderer payload, played in the preview. */
export const $petGenPreview = atom<PetInfo | null>(null)

function isMissingMethod(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /method not found|-32601|unknown method|no such method/i.test(message)
}

/** Clear all generation state (on close, or before a fresh run). */
export function resetPetGen(): void {
  $petGenStatus.set('idle')
  $petGenError.set(null)
  $petGenToken.set(null)
  $petGenDrafts.set([])
  $petGenSelected.set(null)
  $petGenPreview.set(null)
}

/**
 * Reset on palette close, deleting an unadopted preview pet first so a hatched-
 * but-never-adopted creature doesn't linger in the gallery. Fire-and-forget.
 */
export function cleanupPetGen(request: GatewayRequest): void {
  const preview = $petGenPreview.get()

  if ($petGenStatus.get() === 'preview' && preview?.slug) {
    void request('pet.remove', { slug: preview.slug }).catch(() => {})
  }

  resetPetGen()
}

interface GenerateOptions {
  prompt: string
  style?: string
  count?: number
}

/** Generate (or retry) a fresh set of base-look drafts for `prompt`. */
export async function generateDrafts(request: GatewayRequest, options: GenerateOptions): Promise<boolean> {
  const prompt = options.prompt.trim()

  if (!prompt) {
    return false
  }

  $petGenStatus.set('generating')
  $petGenError.set(null)
  $petGenDrafts.set([])
  $petGenSelected.set(null)

  try {
    const result = await request<{ ok: boolean; token: string; drafts: PetDraft[] }>('pet.generate', {
      prompt,
      style: options.style ?? 'auto',
      count: options.count ?? 4
    })

    if (!result?.ok || !result.drafts?.length) {
      throw new Error('generation produced no drafts')
    }

    $petGenToken.set(result.token)
    $petGenDrafts.set(result.drafts)
    $petGenSelected.set(result.drafts[0]?.index ?? 0)
    $petGenStatus.set('ready')

    return true
  } catch (e) {
    if (isMissingMethod(e)) {
      $petGenStatus.set('stale')
    } else {
      $petGenStatus.set('error')
      $petGenError.set(e instanceof Error ? e.message : 'Could not generate pet drafts.')
    }

    return false
  }
}

interface HatchOptions {
  name: string
  description?: string
  prompt?: string
  style?: string
}

/**
 * Hatch the selected draft into a full pet (installed but NOT yet active) and
 * load its renderer payload into the preview. Adoption is a separate, explicit
 * step (`adoptHatched`) so the user sees every frame play before committing.
 * Returns true when the preview is ready.
 */
export async function hatchSelected(request: GatewayRequest, options: HatchOptions): Promise<boolean> {
  const token = $petGenToken.get()
  const index = $petGenSelected.get()
  const name = options.name.trim()

  if (token === null || index === null || !name) {
    return false
  }

  $petGenStatus.set('hatching')
  $petGenError.set(null)

  try {
    const result = await request<{ ok: boolean; slug: string; displayName: string; pet?: PetInfo }>('pet.hatch', {
      token,
      index,
      name,
      description: options.description ?? '',
      prompt: options.prompt ?? name,
      style: options.style ?? 'auto'
    })

    if (!result?.ok || !result.pet?.spritesheetBase64) {
      throw new Error('hatch produced no preview')
    }

    $petGenPreview.set({ ...result.pet, enabled: true })
    $petGenStatus.set('preview')

    return true
  } catch (e) {
    $petGenStatus.set('error')
    $petGenError.set(e instanceof Error ? e.message : 'Could not hatch the pet.')

    return false
  }
}

export interface AdoptOutcome {
  ok: boolean
  slug?: string
  displayName?: string
}

/**
 * Adopt the previewed pet: activate it (`pet.select`), refresh the gallery + live
 * mascot, and clear generation state. No-op unless a preview exists.
 */
export async function adoptHatched(request: GatewayRequest): Promise<AdoptOutcome> {
  const preview = $petGenPreview.get()

  if (!preview?.slug) {
    return { ok: false }
  }

  $petGenStatus.set('adopting')
  $petGenError.set(null)

  try {
    const result = await request<{ ok: boolean; slug: string; displayName: string }>('pet.select', {
      slug: preview.slug
    })

    if (!result?.ok) {
      throw new Error('adopt failed')
    }

    await loadPetGallery(request, { force: true })
    resetPetGen()

    return { ok: true, slug: result.slug, displayName: result.displayName }
  } catch (e) {
    $petGenStatus.set('preview')
    $petGenError.set(e instanceof Error ? e.message : 'Could not adopt the pet.')

    return { ok: false }
  }
}

/**
 * Throw away the previewed pet (`pet.remove`) and return to the draft picker so
 * the user can choose another base or regenerate. Best-effort on the delete.
 */
export async function discardHatched(request: GatewayRequest): Promise<void> {
  const preview = $petGenPreview.get()

  if (preview?.slug) {
    await request('pet.remove', { slug: preview.slug }).catch(() => {})
  }

  $petGenPreview.set(null)
  $petGenError.set(null)
  $petGenStatus.set($petGenDrafts.get().length > 0 ? 'ready' : 'idle')
}
