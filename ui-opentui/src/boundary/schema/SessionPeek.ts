/**
 * SessionPeek decoder — decode-at-boundary (house rule) for the `session.peek`
 * RPC result (tui_gateway/server.py `@method("session.peek")`, shipped with the
 * resume-picker gateway half, commit 529d8084b). The response powers the
 * picker's Space preview:
 *
 *   { session: {id, title, source, model, cwd, started_at, ended_at,
 *               end_reason, message_count, last_active, cost_usd},
 *     head: [{id, role, content(≤2000), truncated, timestamp}, …],
 *     tail: [same — never overlaps head],
 *     total_messages: int }
 *
 * Wire nullability per the server: `model`/`cwd`/`ended_at`/`end_reason`/
 * `cost_usd` are `None` when unknown; message `id`/`timestamp` come straight
 * off DB rows (left loose). Decoded with `Schema.decodeUnknownOption` — a
 * malformed payload yields `Option.none` and the preview pane shows its
 * honest "preview unavailable" line instead of crashing the overlay.
 */
import { Schema } from 'effect'

const Str = Schema.String
const Num = Schema.Number
const opt = Schema.optionalKey

const PeekMessageSchema = Schema.Struct({
  role: opt(Str),
  content: opt(Str),
  truncated: opt(Schema.Boolean),
  timestamp: opt(Schema.NullOr(Schema.Unknown))
})

export const SessionPeekSchema = Schema.Struct({
  session: opt(
    Schema.Struct({
      id: opt(Str),
      title: opt(Schema.NullOr(Str)),
      source: opt(Schema.NullOr(Str)),
      model: opt(Schema.NullOr(Str)),
      cwd: opt(Schema.NullOr(Str)),
      started_at: opt(Schema.NullOr(Num)),
      ended_at: opt(Schema.NullOr(Num)),
      end_reason: opt(Schema.NullOr(Str)),
      message_count: opt(Schema.NullOr(Num)),
      last_active: opt(Schema.NullOr(Num)),
      cost_usd: opt(Schema.NullOr(Num))
    })
  ),
  head: opt(Schema.Array(PeekMessageSchema)),
  tail: opt(Schema.Array(PeekMessageSchema)),
  total_messages: opt(Num)
})
export type SessionPeekDecoded = typeof SessionPeekSchema.Type

/** Decode a loose session.peek result → `Option<SessionPeekDecoded>`. */
export const decodeSessionPeek = Schema.decodeUnknownOption(SessionPeekSchema)
