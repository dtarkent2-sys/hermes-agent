/**
 * Composer — the input row (spec v4 §2). A native <textarea> captured by ref;
 * Enter submits, the input clears imperatively, and a live slash-completion
 * dropdown renders ABOVE it as you type `/…` (spec §1 completions).
 *
 * Newlines: Shift+Enter inserts one on kitty-protocol terminals (ghostty/
 * kitty/wezterm — legacy input can't distinguish it from Enter, both arrive as
 * the CR byte, so plain Enter stays submit everywhere); Alt+Enter is the
 * universal fallback (ESC-prefixed CR in legacy terminals), Ctrl+J too (the
 * legacy linefeed byte keeps the stock `newline` binding). Height: the box
 * auto-grows to COMPOSER_MAX_ROWS (Ink parity 8; HERMES_TUI_COMPOSER_ROWS
 * overrides) then scrolls INTERNALLY — the native edit buffer keeps the cursor
 * in view, Up/Down navigate lines in a multi-line buffer (history recall is a
 * single-line affair), Home/End are per-line (Ctrl+Home/End jump the buffer),
 * and a muted `line 12/40` indicator appears only while the buffer is taller
 * than the cap.
 *
 * Gotchas (§8 #3): `flexShrink:0` so it never collapses onto its rule; clear via
 * `.clear()` (NOT key-remount); a `submitting` re-entrancy guard.
 *
 * Completions: `onContentChange` reports the text → `onType` (entry boundary)
 * queries `complete.slash` and fills `completions()`. The textarea owns key input
 * (so live-refine-by-typing works); menu keys are routed by the pure
 * `routeMenuKey` table (Epic 8): Tab accepts / Esc dismisses any open menu, and
 * for the SLASH menu (first char `/`) Up/Down move a themed highlight (wrapping)
 * and Enter accepts it — `key.preventDefault()` keeps the consumed key out of
 * the textarea (cursor moves / its `submit` binding). Path/@-mention menus keep
 * Tab-only accept so arrows/Enter retain history/cursor/submit meanings.
 * `onSubmit`/`onType` are plain callbacks wired by the entry — no Effect here.
 *
 * Skill highlighting + one-edit autocorrect (Epic 6): standalone `/name` tokens
 * whose name exactly matches a valid command/skill name get a native textarea
 * highlight (editBuffer.addHighlightByCharRange + a SyntaxStyle — the same
 * range-styling seam the extmarks demo uses, WITHOUT ExtmarksController's
 * cursor monkey-patching, so the token stays normally editable). The catalog of
 * valid names is LEARNED from the slash-completion batches the gateway already
 * sends (module-level, survives composer remounts) — the completion flow is the
 * source of truth, nothing is hardcoded. When the message is EXACTLY a bare
 * lead token one edit away from one valid name (`/comit`) and the gateway menu
 * is empty, a synthetic "did you mean" row rides the SAME dropdown (same
 * routeMenuKey routing/accept path; Esc dismisses it until the text changes).
 * Anti-jank: a `/` mid-prose never completes or autocorrects — exact tokens get
 * highlight-only, everything else gets nothing (see logic/skillMatch.ts).
 *
 * Always-active input (item 2): the textarea focuses on mount, on click
 * (onMouseDown), and reclaims focus on the next PRINTABLE keystroke if focus ever
 * drifted off (e.g. the transcript scrollbox grabbed it on a mouse-scroll). Nav
 * keys are left alone so keyboard transcript-scroll still works (opencode keeps
 * the prompt focused via a reactive effect; here a keystroke net is enough since
 * the composer remounts+refocuses whenever an overlay closes).
 */
import { SyntaxStyle, type PasteEvent, type TextareaRenderable } from '@opentui/core'
import { useKeyboard } from '@opentui/solid'
import { createEffect, createMemo, createSignal, For, on, onCleanup, onMount, Show } from 'solid-js'

import { MENU_MAX, routeMenuKey } from '../logic/completionMenu.ts'
import { envComposerRows } from '../logic/env.ts'
import { createDoublePress } from '../logic/promptHistory.ts'
import { analyzeSlash, learnableNames, nativeCharOffset } from '../logic/skillMatch.ts'
import type { CompletionItem } from '../logic/store.ts'
import type { PromptHistory } from '../logic/history.ts'
import { type PasteStore, shouldPlaceholder } from '../logic/pastes.ts'
import { useDimensions } from './dimensions.tsx'
import { useTheme } from './theme.tsx'

const GUTTER = 2

/** Valid command/skill names learned from gateway slash-completion batches
 *  (Epic 6). Module-level so the catalog survives composer remounts (overlays
 *  REPLACE the composer); it only ever holds names the gateway itself offered. */
const LEARNED_NAMES = new Set<string>()

/** Test hook: reset the learned catalog between cases. */
export function resetLearnedNames(): void {
  LEARNED_NAMES.clear()
}

/** Keys that must NOT steal focus back to the composer (scroll/edit/nav). */
const NAV_KEYS = new Set([
  'return',
  'linefeed',
  'tab',
  'escape',
  'backspace',
  'delete',
  'insert',
  'up',
  'down',
  'left',
  'right',
  'home',
  'end',
  'pageup',
  'pagedown',
  'clear',
  'menu'
])

/** A printable, unmodified key press (recoverable into the textarea). */
function isPrintableKey(k: {
  name: string
  ctrl: boolean
  meta: boolean
  option: boolean
  super?: boolean
  sequence: string
  eventType?: string
}): boolean {
  return (
    k.eventType !== 'release' &&
    !k.ctrl &&
    !k.meta &&
    !k.option &&
    !k.super &&
    !NAV_KEYS.has(k.name) &&
    typeof k.sequence === 'string' &&
    k.sequence.length >= 1 &&
    (k.sequence.codePointAt(0) ?? 0) >= 0x20
  )
}

export function Composer(props: {
  onSubmit: (text: string) => void
  onType?: ((text: string) => void) | undefined
  completions?: (() => CompletionItem[]) | undefined
  completionFrom?: (() => number) | undefined
  onDismiss?: (() => void) | undefined
  history?: PromptHistory | undefined
  onImagePaste?: (() => void) | undefined
  pasteStore?: PasteStore | undefined
  /** Down on an EMPTY focused composer with no dropdown open (Epic 2.7 tray
   *  handoff): return true to consume the key (the tray took focus). */
  onFocusDown?: (() => boolean) | undefined
  /** Hands the parent a "focus the textarea" callback (Esc from the tray). */
  registerFocus?: ((focus: () => void) => void) | undefined
  /** Esc+Esc (≤800ms) on an EMPTY composer with no dropdown open (Epic 5): the
   *  parent opens the session prompt-history viewer (or does nothing when the
   *  session has no prompts yet — never an empty modal). */
  onDoubleEsc?: (() => void) | undefined
}) {
  const theme = useTheme()
  const dims = useDimensions()
  // Auto-expand the input as you type, but cap the VISIBLE height at
  // COMPOSER_MAX_ROWS (Ink parity: 8 lines; HERMES_TUI_COMPOSER_ROWS overrides) —
  // beyond that the native edit buffer scrolls internally and keeps the cursor
  // in view. Small screens still bound it to ~a third of the rows.
  const capRows = envComposerRows(process.env.HERMES_TUI_COMPOSER_ROWS)
  const maxHeight = () => Math.min(capRows, Math.max(3, Math.floor(dims().height / 3)))
  let ta: TextareaRenderable | undefined
  let submitting = false
  const completions = () => props.completions?.() ?? []
  /** The gateway's dropdown rows (capped; selection wraps within them). */
  const storeItems = () => completions().slice(0, MENU_MAX)

  // ── skill highlighting + one-edit autocorrect (Epic 6) ────────────────
  // The composer text as a signal (onContentChange keeps it current) so the
  // token analysis is reactive; `namesRev` bumps when the learned catalog grows
  // (completion batches arrive async, after the text changed).
  const [bufText, setBufText] = createSignal('')
  const [namesRev, setNamesRev] = createSignal(0)
  // Cursor line / total lines (item 3): drives the quiet `line 12/40` indicator
  // shown only while the buffer is taller than the visible cap (internal scroll).
  const [cursorLine, setCursorLine] = createSignal(0)
  const [lineTotal, setLineTotal] = createSignal(1)
  // Esc on the suggestion row parks it for THIS exact text; any edit re-arms.
  const [dismissedFor, setDismissedFor] = createSignal<string | undefined>(undefined)
  // Learn names from slash-completion batches (bare `/…` lead token only —
  // after a space the gateway completes ARGS, not names; see learnableNames).
  createEffect(
    on(
      () => props.completions?.(),
      items => {
        let grew = false
        for (const name of learnableNames(bufText(), items ?? [])) {
          if (!LEARNED_NAMES.has(name)) {
            LEARNED_NAMES.add(name)
            grew = true
          }
        }
        if (grew) setNamesRev(r => r + 1)
      }
    )
  )
  const analysis = createMemo(() => {
    namesRev() // re-analyze when the catalog grows
    return analyzeSlash(bufText(), LEARNED_NAMES)
  })
  /** The one-edit autocorrect, gated anti-jank: only while the gateway menu is
   *  EMPTY (a live prefix menu always wins) and not Esc-dismissed for this text. */
  const suggested = () => {
    const s = analysis().suggestion
    return s && dismissedFor() !== bufText() ? s : undefined
  }
  /** The visible dropdown rows: the gateway menu, else the synthetic suggestion. */
  const menuItems = (): CompletionItem[] => {
    const items = storeItems()
    if (items.length > 0) return items
    const s = suggested()
    return s ? [{ display: `/${s.name}`, meta: 'did you mean? (Tab/Enter to accept)', text: s.name }] : []
  }

  // Native highlight plumbing: one SyntaxStyle per mount holding the token
  // style; ranges are recomputed from `analysis()` on every change (clear+add —
  // the same recompute model ExtmarksController uses). Best-effort: a native
  // styling failure must never take the composer down.
  let syntax: SyntaxStyle | undefined
  let tokenStyleId = 0
  onMount(() => {
    try {
      const style = SyntaxStyle.create()
      tokenStyleId = style.registerStyle('slash-token', { bold: true, fg: theme().color.accent })
      if (ta) ta.syntaxStyle = style
      syntax = style
    } catch {
      syntax = undefined
    }
  })
  onCleanup(() => {
    try {
      if (ta && !ta.isDestroyed) ta.syntaxStyle = null
      syntax?.destroy()
    } catch {
      /* teardown is best-effort */
    }
    syntax = undefined
  })
  createEffect(() => {
    const a = analysis()
    if (!ta || !syntax || ta.isDestroyed) return
    try {
      const text = bufText()
      ta.editBuffer.clearAllHighlights()
      for (const t of a.highlights) {
        ta.editBuffer.addHighlightByCharRange({
          end: nativeCharOffset(text, t.end),
          start: nativeCharOffset(text, t.start),
          styleId: tokenStyleId
        })
      }
      ta.requestRender()
    } catch {
      /* highlight is cosmetic — never crash on a native hiccup */
    }
  })
  // Highlighted dropdown row (Epic 8). New candidates (every refine keystroke
  // swaps the array) reset it to the top match.
  const [selected, setSelected] = createSignal(0)
  createEffect(
    on(
      () => props.completions?.(),
      () => setSelected(0)
    )
  )
  // Whether the composer text starts with `/` (slash menu vs path menu) — a
  // signal so the dropdown hint stays reactive; the key handler re-checks
  // `ta.plainText` directly.

  /** Replace the textarea content and park the cursor at the end (history recall). */
  const setBuffer = (text: string) => {
    if (!ta) return
    ta.setText(text)
    ta.cursorOffset = text.length
  }

  /** Splice the n-th candidate into the buffer (Tab/Enter accept). Only the token
   *  being completed is replaced — `completionFrom` is the gateway's
   *  `replace_from` / token start — then the trailing space lets arg-completion
   *  continue (setText fires onContentChange → onType re-queries). */
  const acceptCompletion = (index: number) => {
    const item = menuItems()[index] ?? menuItems()[0]
    if (!item || !ta) return
    // A synthetic suggestion row (gateway menu empty) replaces from just past
    // the `/` (its own `from`); gateway rows keep the store's replace_from.
    const synthetic = storeItems().length === 0
    const from = synthetic ? (suggested()?.from ?? 1) : (props.completionFrom?.() ?? 0)
    const before = ta.plainText.slice(0, Math.min(Math.max(0, from), ta.plainText.length))
    setBuffer(before + item.text + ' ')
    props.onDismiss?.()
  }

  // Esc+Esc → session prompt history (Epic 5; free-code's double-press model).
  // ONLY an Esc that nothing else consumed counts: the dropdown-dismiss branch
  // returns before press() is reached (so a dismissing Esc never arms), and any
  // other key resets the window (intervening keys disarm).
  const doubleEsc = createDoublePress()

  const submit = () => {
    if (submitting || !ta) return
    // Expand any `[Pasted text #N]` placeholders back to their full content before
    // sending (item: pasted-text). No-op when nothing was placeheld.
    const text = (props.pasteStore?.expand(ta.plainText) ?? ta.plainText).trim()
    if (!text) return
    submitting = true
    props.onSubmit(text)
    props.history?.push(text)
    ta.clear()
    props.pasteStore?.clear()
    props.onDismiss?.()
    submitting = false
  }

  /** Refresh the line-indicator signals from the textarea (best-effort). */
  const syncCursorLine = () => {
    if (!ta || ta.isDestroyed) return
    setCursorLine(ta.logicalCursor.row)
    setLineTotal(ta.lineCount)
  }

  useKeyboard(key => {
    // line-indicator upkeep: plain cursor movement emits no native event, so
    // sync AFTER this dispatch (the global handler runs before the textarea
    // processes the key; the microtask runs after both). Equal-value signal
    // sets are no-ops, so this is cheap on every keystroke.
    queueMicrotask(syncCursorLine)
    // 0) double-Esc bookkeeping: any non-Esc press is an intervening key and
    // disarms the pending Esc (free-code resets on every other input).
    if (key.eventType !== 'release' && key.name !== 'escape') doubleEsc.reset()
    // 1) completion-menu keys while the dropdown is open (Epic 8): Tab accept /
    // Esc dismiss for ANY menu (the pre-existing semantics — Esc stays exactly
    // "dismiss if open, else fall through"), plus Up/Down/Enter for the SLASH
    // menu only (routeMenuKey's precedence table). preventDefault keeps a
    // consumed arrow/Enter from also reaching the textarea (cursor move / its
    // `submit` keybinding); Tab/Esc stay un-prevented as before.
    const menu = menuItems()
    if (menu.length > 0) {
      const action = routeMenuKey(key.name, key.ctrl || key.meta || key.option, {
        count: menu.length,
        selected: selected(),
        slashMenu: (ta?.plainText ?? '').startsWith('/')
      })
      if (action.kind === 'move') {
        setSelected(action.selected)
        key.preventDefault()
        return
      }
      if (action.kind === 'accept') {
        acceptCompletion(action.index)
        if (key.name === 'return') key.preventDefault()
        return
      }
      if (action.kind === 'dismiss') {
        // also park the synthetic suggestion for this exact text (Esc must not
        // re-open it on the next analysis pass); any edit re-arms it.
        setDismissedFor(ta?.plainText ?? '')
        props.onDismiss?.()
        // a CONSUMED Esc never counts toward the Esc+Esc viewer (Epic 5).
        doubleEsc.reset()
        return
      }
    }
    // 1.5) Esc+Esc on an EMPTY, FOCUSED composer (no dropdown — the dismiss
    // branch returned above) opens the session prompt-history viewer (Epic 5).
    // With text in the buffer the Esc is just an intervening key (disarms);
    // unfocused (e.g. the agents tray owns the keys) it never counts.
    if (key.name === 'escape' && key.eventType !== 'release' && !key.ctrl && !key.meta && !key.option) {
      if (ta?.focused === true && ta.plainText === '' && !key.defaultPrevented) {
        if (doubleEsc.press()) props.onDoubleEsc?.()
      } else {
        doubleEsc.reset()
      }
      return
    }
    // 2) background-agents tray handoff (Epic 2.7): Down on an EMPTY focused
    // composer with NO dropdown open offers focus to the tray. The parent decides
    // eligibility (≥1 running agent; overlays/prompts replace the composer
    // entirely, so they can't get here) and returns true when it took focus —
    // preventDefault keeps the consumed Down out of the textarea AND out of the
    // tray's own selection handler (it skips defaultPrevented keys). Otherwise
    // Down keeps every existing meaning (menu nav above, history below).
    if (
      key.name === 'down' &&
      !key.ctrl &&
      !key.meta &&
      !key.option &&
      menu.length === 0 &&
      ta?.focused === true &&
      ta.plainText === '' &&
      props.onFocusDown?.() === true
    ) {
      key.preventDefault()
      return
    }
    // 3) prompt history (item 6): Up/Down recall older/newer prompts — but ONLY
    // while the buffer is a SINGLE line (big-text navigation, item 3): in a
    // multi-line buffer the arrows always fall through to the textarea's own
    // cursor movement (the internal viewport follows the cursor), so a tall
    // paste is never clobbered by a history recall. Modified arrows (Shift
    // select-up, Ctrl/Alt word-ish moves) belong to the textarea too. Gated on
    // the textarea being FOCUSED: while focus is elsewhere (the agents tray,
    // the transcript scrollbox) arrows must not recall history into the buffer.
    if (ta?.focused && props.history) {
      const unmodified = !key.ctrl && !key.meta && !key.option && !key.shift
      if (unmodified && ta.lineCount <= 1) {
        if (key.name === 'up') {
          const entry = props.history.prev(ta.plainText)
          if (entry !== null) setBuffer(entry)
          return
        }
        if (key.name === 'down') {
          const entry = props.history.next()
          if (entry !== null) setBuffer(entry)
          return
        }
      }
      // any edit resets the recall cursor so the next Up starts from the bottom
      if (key.name === 'backspace' || key.name === 'delete' || isPrintableKey(key)) {
        props.history.reset()
      }
    }
    // 4) always-active input (item 2): a printable key while the textarea lost
    // focus reclaims it. The renderer runs this GLOBAL handler BEFORE routing the
    // key to the focused renderable, so after focus() the SAME keystroke is still
    // delivered to the (now-focused) textarea — do NOT insert it here too, or the
    // first letter doubles. Nav/scroll keys are untouched.
    if (ta && !ta.focused && isPrintableKey(key)) {
      ta.focus()
    }
  })

  onMount(() => {
    ta?.focus()
    props.registerFocus?.(() => ta?.focus())
  })

  return (
    <box style={{ flexDirection: 'column', flexShrink: 0 }}>
      <Show when={menuItems().length > 0}>
        <box
          style={{
            backgroundColor: theme().color.completionBg,
            flexDirection: 'column',
            paddingLeft: 1,
            paddingRight: 1
          }}
        >
          {/* the completion dropdown is transient input chrome (menu rows + the
              key-hint) — not transcript content — so it's excluded from mouse
              selection (item 4). The highlighted row tracks `selected()` (Epic 8)
              with the THEMED completionCurrentBg — Up/Down move it on the slash
              menu; on path menus it stays on the top match (Tab's target). */}
          <For each={menuItems()}>
            {(c, i) => (
              <box
                style={{
                  backgroundColor: i() === selected() ? theme().color.completionCurrentBg : theme().color.completionBg
                }}
              >
                <text selectable={false} fg={i() === selected() ? theme().color.accent : theme().color.text}>
                  {c.display || c.text}
                  {c.meta ? `  ${c.meta}` : ''}
                </text>
              </box>
            )}
          </For>
          <text selectable={false} fg={theme().color.muted}>
            {'↑/↓ select · Enter/Tab accept · Esc dismiss'}
          </text>
        </box>
      </Show>
      {/* prompt glyph + textarea — the glyph (item 3) marks the input line so the
          composer is distinguished by structure (glyph + the status-bar rule above),
          not a background tint. PRIMARY BOLD: the idle view's one bright action
          (design pass — gold sits on the newest answer and on the ❯ waiting for
          the next command, nowhere else). */}
      <box style={{ flexDirection: 'row', flexShrink: 0 }}>
        <box style={{ flexShrink: 0, width: GUTTER }}>
          <text selectable={false}>
            <span style={{ fg: theme().color.primary }}>
              <b>{theme().brand.prompt}</b>
            </span>
          </text>
        </box>
        <textarea
          ref={el => (ta = el)}
          minHeight={1}
          maxHeight={maxHeight()}
          style={{ flexGrow: 1, minWidth: 0 }}
          placeholder={theme().brand.welcome}
          placeholderColor={theme().color.muted}
          textColor={theme().color.text}
          cursorColor={theme().color.accent}
          keyBindings={[
            // Enter submits. LEGACY input can't tell Shift+Enter from Enter (both
            // are the CR byte), so plain return must stay submit everywhere.
            { action: 'submit', name: 'return' },
            { action: 'submit', name: 'kpenter' },
            // kitty-protocol terminals (ghostty/kitty/wezterm) report Shift on
            // return as a distinct event → newline, the modern chat-input feel.
            { action: 'newline', name: 'return', shift: true },
            { action: 'newline', name: 'kpenter', shift: true },
            // Alt+Enter: the UNIVERSAL newline fallback — legacy terminals send
            // ESC-prefixed CR for it, so it works without kitty. (Overrides the
            // stock meta+return submit, which plain Enter already covers. Ctrl+J
            // — the legacy linefeed byte — also stays newline via the defaults.)
            { action: 'newline', meta: true, name: 'return' },
            { action: 'newline', meta: true, name: 'kpenter' },
            // Home/End move PER LINE in a multi-line buffer (stock binds them to
            // the buffer ends); Ctrl+Home/End keep the whole-buffer jumps.
            { action: 'line-home', name: 'home' },
            { action: 'line-end', name: 'end' },
            { action: 'select-line-home', name: 'home', shift: true },
            { action: 'select-line-end', name: 'end', shift: true },
            { action: 'buffer-home', ctrl: true, name: 'home' },
            { action: 'buffer-end', ctrl: true, name: 'end' }
          ]}
          onMouseDown={() => ta?.focus()}
          onSubmit={submit}
          onPaste={(e: PasteEvent) => {
            const text = new TextDecoder().decode(e.bytes)
            // An empty bracketed paste = an image-only clipboard (item 1) — read + attach it.
            if (text.trim() === '') {
              e.preventDefault()
              props.onImagePaste?.()
              return
            }
            // A large paste becomes a compact `[Pasted text #N +M lines]` chip instead
            // of flooding the input; the real text is expanded back on submit.
            if (props.pasteStore && shouldPlaceholder(text)) {
              e.preventDefault()
              ta?.insertText(props.pasteStore.add(text))
              return
            }
            // small pastes fall through to the textarea's native insert
          }}
          onCursorChange={() => syncCursorLine()}
          onContentChange={() => {
            const text = ta?.plainText ?? ''
            setBufText(text) // drives the token analysis (highlight + suggestion)
            syncCursorLine()
            props.onType?.(text)
          }}
        />
      </box>
      {/* quiet line indicator (item 3): only while the buffer is TALLER than the
          visible cap (the textarea is scrolling internally), so the position in a
          big paste isn't a mystery. Muted + right-aligned — input chrome, not
          transcript content, hence selectable={false}. */}
      <Show when={lineTotal() > maxHeight()}>
        <box style={{ flexDirection: 'row', flexShrink: 0, justifyContent: 'flex-end' }}>
          <text selectable={false} fg={theme().color.muted}>
            {`line ${Math.min(cursorLine() + 1, lineTotal())}/${lineTotal()}`}
          </text>
        </box>
      </Show>
    </box>
  )
}
