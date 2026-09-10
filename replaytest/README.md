# replaytest

A fixture replay harness for provider stream tests. It plays a recorded
upstream response through a real fantasy provider and compares the resulting
`fantasy.StreamPart` sequence against a saved golden file, so regressions in
stream handling become visible diffs.

## Fixture layout

Fixtures live under `providertests/testdata/shapes/<shape>/<case>/`. The
directory names describe what the upstream chunks look like (`toolcall`,
`reasoning_then_text`, `usage_in_trailing_chunk`, ...), never which provider
produced them. Provider-specific protocols are namespaced explicitly, e.g.
`anthropic/toolcall_stream/basic`, where the protocol itself is the shape.

Each fixture directory contains:

| file              | required | purpose                                                      |
| ----------------- | -------- | ------------------------------------------------------------ |
| `meta.json`       | yes      | `{"provider": "openaicompat"}` or `{"provider": "anthropic"}` |
| `request.json`    | yes      | the upstream request the provider sends, for documentation   |
| `response.sse`    | one of   | the raw streaming response body                              |
| `response.json`   | one of   | the raw non-streaming response body                          |
| `parts.golden.json` | yes    | golden for the provider's StreamPart sequence                |
| `step.golden.json`  | no     | golden for a one-step agent run (see RunAgentStep)           |

`response.sse` is split into events on blank-line boundaries; each event is
kept verbatim (including its `data:`/`event:` prefixes) and written to the
replay server one event at a time with a flush after each. Events are never
merged or split. CRLF line endings (as produced by git checkouts on Windows)
are normalized to LF before splitting. A fixture event consisting of exactly
the line
`<connection closed>` makes the server flush what it has written and close
the connection, simulating an upstream that dies mid-stream.

Fixtures use placeholder model names (for example `test-model`) and contain
no real request IDs, response-ID prefixes, system fingerprints, endpoints, or
org IDs. They document current behavior, including known issues; when a
golden looks wrong, leave it and fix the behavior in a dedicated card.

## Golden format

`parts.golden.json` holds a JSON array of `PartRecord` values, one per
`StreamPart` in emission order. `step.golden.json` holds a single
`StepRecord`. Only the fields meaningful for the part type are set; the rest
are omitted:

- `delta` carries streamed text, reasoning, and tool-input fragments, and
  the full text of text and reasoning content parts
- `name` and `input` carry tool call names and arguments; for tool-result
  content parts, `input` carries the text result
- `reason` carries the finish reason on finish parts
- `usage` carries token usage, with zero fields omitted so goldens show
  exactly which numbers the provider reported
- `metadata` carries provider metadata (and source fields for source parts)
- `warnings` and `error` carry call warnings and provider errors; error text
  has the replay server's listener port normalized to `127.0.0.1:0`

Golden comparisons normalize CRLF line endings to LF on both sides, so
goldens stay byte-stable regardless of how git checked the files out.

## Running and updating goldens

```sh
go test ./providertests -run TestFixtureShapes
go test ./providertests -run TestFixtureShapes -update
```

Under `-update` the golden files are rewritten with the observed output and
the tests pass. Any other package can use the same flag via
`replaytest.AssertGolden`.

**Rule: never regenerate an existing golden to make a change pass.** If your
change alters an existing golden, stop and report the diff in your PR,
unchanged, unless the ticket explicitly says the goldens will change — then
list every changed line and why. A missing golden for a fixture you are
adding is fine to create with `-update`; review it before committing.

## API

```go
fixture, err := replaytest.Load(dir)
server := replaytest.Serve(t, fixture)                    // or Serve(t, fixture, opts...)
requests := server.Requests()                             // every request body received
records := replaytest.Collect(stream)                     // StreamParts -> PartRecords
replaytest.AssertGolden(t, path, records)                 // compare or -update
record := replaytest.RunAgentStep(t, model, fixture, tools...) // one agent step
```

- `Load(dir)` reads `request.json` and exactly one of `response.sse` or
  `response.json`.
- `Serve` answers the first request with the fixture's response. Second and
  later requests get a canned minimal finish response shaped by the request
  (streaming or not, messages endpoint or chat-completions endpoint), unless
  `WithSubsequentSSE(events...)` or `WithSubsequentJSON(body)` supplies one.
  The canned responses are keyed by what the request looks like, never by
  provider name.
- `RunAgentStep` runs `Agent.Stream` for one step against a fresh server
  built from the fixture, with the given stub tools (wrapped so dispatches
  are recorded) and retries disabled. It returns the step content as
  `PartRecord`s, the finish reason, which tools were dispatched, and the raw
  body of the next request the agent sent (from `Server.Requests()[1]`), or
  empty when the step made no further request. If the step errors, the
  record's `error` field carries it. The agent prompt is fixed (`hi`) so
  step goldens are stable.

## Determinism

Providers call `fantasy.NewID` (default `uuid.NewString`) for generated IDs.
`RunAgentStep` replaces it with a counter (`id-1`, `id-2`, ...) for the
duration of the call and restores it afterwards. Fixtures themselves are
authored clean and scrub nothing at runtime.
