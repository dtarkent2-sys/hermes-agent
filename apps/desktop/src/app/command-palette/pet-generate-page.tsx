/**
 * Cmd-K → Pets → "Generate" page — describe a pet, pick a draft, hatch it.
 *
 * A thin view over the `pet-generate` store. The palette search box doubles as
 * the concept prompt; this page renders the variant grid, the selection, the
 * retry/hatch actions, and the loading states. The store owns the two-step
 * `pet.generate` → `pet.hatch` flow.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { PetSprite } from '@/components/pet/pet-sprite'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Egg, Loader2, PawPrint, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { closeCommandPalette } from '@/store/command-palette'
import { type PetInfo, type PetState } from '@/store/pet'
import {
  $petGenDrafts,
  $petGenError,
  $petGenPreview,
  $petGenSelected,
  $petGenStatus,
  adoptHatched,
  discardHatched,
  generateDrafts,
  hatchSelected
} from '@/store/pet-generate'

const VARIANT_COUNT = 4

// Fixed render scale for the preview so it's a predictable size regardless of
// the user's configured `display.pet.scale`.
const PREVIEW_SCALE = 0.7

// States the preview cycles through so the user sees every animation row.
const PREVIEW_STATES: PetState[] = ['idle', 'wave', 'run', 'review', 'jump', 'failed']
const PREVIEW_STATE_MS = 1500

interface PetGeneratePageProps {
  search: string
}

export function PetGeneratePage({ search }: PetGeneratePageProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet
  const { requestGateway } = useGatewayRequest()

  const status = useStore($petGenStatus)
  const error = useStore($petGenError)
  const drafts = useStore($petGenDrafts)
  const selected = useStore($petGenSelected)
  const preview = useStore($petGenPreview)
  const [name, setName] = useState('')

  const prompt = search.trim()
  const busy = status === 'generating' || status === 'hatching'

  const generate = () => {
    if (prompt) {
      void generateDrafts(requestGateway, { prompt })
    }
  }

  const hatch = () => {
    void hatchSelected(requestGateway, { name: name.trim() || prompt, prompt })
  }

  const adopt = () => {
    void adoptHatched(requestGateway).then(out => {
      if (out.ok) {
        triggerHaptic('crisp')
        closeCommandPalette()
      }
    })
  }

  if (status === 'stale') {
    return <Status text={copy.staleBackend} tone="error" />
  }

  // Hatching is slow (several grounded image generations) — own the whole pane.
  if (status === 'hatching') {
    return <Status icon={<Loader2 className="size-4 animate-spin" />} text={copy.hatching} />
  }

  // Preview: play every animation row before the user commits.
  if ((status === 'preview' || status === 'adopting') && preview) {
    return (
      <HatchPreview
        adopting={status === 'adopting'}
        error={error}
        onAdopt={adopt}
        onDiscard={() => void discardHatched(requestGateway)}
        pet={preview}
      />
    )
  }

  const hasDrafts = drafts.length > 0
  const generating = status === 'generating'
  const cells = generating ? Array.from({ length: VARIANT_COUNT }, (_, i) => ({ index: i, dataUri: '' })) : drafts

  return (
    <div className="flex flex-col gap-2 p-2">
      {error && <p className="px-1 text-[0.6875rem] text-(--ui-red)">{error}</p>}

      {!hasDrafts && !generating && (
        <p className="px-1 py-1 text-xs text-muted-foreground">{prompt ? copy.readyHint : copy.promptHint}</p>
      )}

      {(hasDrafts || generating) && (
        <div className="grid grid-cols-2 gap-2">
          {cells.map((draft, i) => {
            const isSelected = !generating && selected === draft.index

            return (
              <button
                className={cn(
                  'relative flex aspect-square items-center justify-center overflow-hidden rounded-lg border bg-(--ui-bg-quinary) transition-colors',
                  isSelected
                    ? 'border-(--ui-accent) ring-2 ring-(--ui-accent)/40'
                    : 'border-(--ui-stroke-tertiary) hover:border-foreground/40'
                )}
                disabled={generating || busy}
                key={generating ? i : draft.index}
                onClick={() => $petGenSelected.set(draft.index)}
                onMouseDown={event => event.preventDefault()}
                type="button"
              >
                {generating ? (
                  <Loader2 className="size-5 animate-spin text-muted-foreground" />
                ) : (
                  <img alt="" className="size-full object-contain" draggable={false} src={draft.dataUri} />
                )}
                {isSelected && (
                  <span className="absolute right-1 top-1 rounded-full bg-(--ui-accent) p-0.5 text-(--ui-base)">
                    <Check className="size-3" />
                  </span>
                )}
              </button>
            )
          })}
        </div>
      )}

      {hasDrafts ? (
        <div className="flex flex-col gap-2">
          <input
            className="w-full rounded-md border border-(--ui-stroke-tertiary) bg-transparent px-2 py-1.5 text-xs outline-none placeholder:text-muted-foreground focus:border-foreground/40"
            onChange={event => setName(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                hatch()
              }
            }}
            placeholder={copy.namePlaceholder}
            value={name}
          />
          <div className="flex gap-2">
            <button
              className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-border px-2 py-1.5 text-xs font-medium transition-colors hover:bg-(--chrome-action-hover) disabled:opacity-50"
              disabled={busy || !prompt}
              onClick={generate}
              onMouseDown={event => event.preventDefault()}
              type="button"
            >
              <RefreshCw className="size-3.5" />
              {copy.retry}
            </button>
            <button
              className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-primary px-2 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
              disabled={busy || selected === null}
              onClick={hatch}
              onMouseDown={event => event.preventDefault()}
              type="button"
            >
              <PawPrint className="size-3.5" />
              {copy.hatch}
            </button>
          </div>
        </div>
      ) : (
        <button
          className="flex items-center justify-center gap-1.5 rounded-md bg-primary px-2 py-2 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
          disabled={busy || !prompt}
          onClick={generate}
          onMouseDown={event => event.preventDefault()}
          type="button"
        >
          {generating ? <Loader2 className="size-3.5 animate-spin" /> : <Egg className="size-3.5" />}
          {generating ? copy.generating : copy.generate}
        </button>
      )}
    </div>
  )
}

interface HatchPreviewProps {
  pet: PetInfo
  adopting: boolean
  error: string | null
  onAdopt: () => void
  onDiscard: () => void
}

function HatchPreview({ pet, adopting, error, onAdopt, onDiscard }: HatchPreviewProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet
  const [stateIndex, setStateIndex] = useState(0)

  // Cycle through the animation rows so the preview showcases all frames.
  useEffect(() => {
    const id = setInterval(() => {
      setStateIndex(i => (i + 1) % PREVIEW_STATES.length)
    }, PREVIEW_STATE_MS)

    return () => clearInterval(id)
  }, [])

  const previewInfo: PetInfo = { ...pet, scale: PREVIEW_SCALE }

  return (
    <div className="flex flex-col items-center gap-2 p-2">
      <div className="flex min-h-[9rem] w-full items-center justify-center rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary) py-2">
        <PetSprite info={previewInfo} stateOverride={PREVIEW_STATES[stateIndex]} />
      </div>

      {pet.displayName && <p className="text-xs font-medium text-foreground">{pet.displayName}</p>}

      {error && <p className="px-1 text-[0.6875rem] text-(--ui-red)">{error}</p>}

      <div className="flex w-full gap-2">
        <button
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md border border-border px-2 py-1.5 text-xs font-medium transition-colors hover:bg-(--chrome-action-hover) disabled:opacity-50"
          disabled={adopting}
          onClick={onDiscard}
          onMouseDown={event => event.preventDefault()}
          type="button"
        >
          <RefreshCw className="size-3.5" />
          {copy.startOver}
        </button>
        <button
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-primary px-2 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-50"
          disabled={adopting}
          onClick={onAdopt}
          onMouseDown={event => event.preventDefault()}
          type="button"
        >
          {adopting ? <Loader2 className="size-3.5 animate-spin" /> : <PawPrint className="size-3.5" />}
          {copy.adopt}
        </button>
      </div>
    </div>
  )
}

function Status({ icon, text, tone }: { icon?: React.ReactNode; text: string; tone?: 'error' }) {
  return (
    <div
      className={cn(
        'flex items-center justify-center gap-2 px-2 py-6 text-xs',
        tone === 'error' ? 'text-(--ui-red)' : 'text-muted-foreground'
      )}
    >
      {icon}
      {text}
    </div>
  )
}
