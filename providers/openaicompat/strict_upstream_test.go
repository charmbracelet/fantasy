package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

// This file pins the DeepSeek/Kimi replay contract against a strict upstream
// double. Unlike serveSSE, which records bodies for after-the-fact
// assertions, serveStrictDeepSeek enforces the two documented rejection rules
// as hard 400s, the way the real strict servers do:
//
//  1. An assistant message carrying neither content nor tool calls is
//     rejected ("Invalid assistant message: content or tool_calls must be
//     set"). Sending one bricks every later request, which is exactly what
//     charmbracelet/crush#3794 and #3788 reported.
//  2. When the request carries tools, an assistant tool-call turn without a
//     reasoning_content field is rejected ("The reasoning_content in the
//     thinking mode must be passed back to the API"), per DeepSeek's
//     thinking-mode guide; dropping it is what made models loop in
//     charmbracelet/crush#2696.
//
// A client that satisfies both rules can drive multi-step tool loops and
// replay canceled-mid-thinking history without poisoning the session.

// strictMessage mirrors one inbound messages[] entry with raw fields so the
// validator can distinguish absent/null/empty from a real value.
type strictMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	ToolCalls        json.RawMessage `json:"tool_calls"`
	ReasoningContent json.RawMessage `json:"reasoning_content"`
}

type strictRequest struct {
	Stream   bool            `json:"stream"`
	Tools    json.RawMessage `json:"tools"`
	Messages []strictMessage `json:"messages"`
}

// hasContent reports whether the message carries user-visible content: a
// non-empty string or a non-empty array of content parts.
func (m strictMessage) hasContent() bool {
	raw := strings.TrimSpace(string(m.Content))
	if raw == "" || raw == "null" {
		return false
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s != ""
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		return len(parts) > 0
	}
	return true
}

func (m strictMessage) hasToolCalls() bool {
	raw := strings.TrimSpace(string(m.ToolCalls))
	if raw == "" || raw == "null" {
		return false
	}
	var calls []json.RawMessage
	if err := json.Unmarshal(m.ToolCalls, &calls); err != nil {
		return false
	}
	return len(calls) > 0
}

// hasReasoningField reports whether reasoning_content is present and not
// null. DeepSeek 400s a tool-call turn when the key is missing; a null means
// "no reasoning in this message" and is not accepted as replay.
func (m strictMessage) hasReasoningField() bool {
	raw := strings.TrimSpace(string(m.ReasoningContent))
	return raw != "" && raw != "null"
}

func strictError(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusBadRequest)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    "invalid_request_error",
			"code":    "invalid_request_error",
		},
	})
}

// serveStrictDeepSeek returns an httptest server that validates every inbound
// request against the two contract rules above (answering 400 on violation)
// and otherwise behaves like serveSSE: streaming POSTs get the given SSE
// payloads in order, non-streaming POSTs get a canned completion. Every
// request body is recorded.
func serveStrictDeepSeek(t *testing.T, payloads ...string) (*httptest.Server, *[][]byte) {
	t.Helper()
	var bodies [][]byte
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		bodies = append(bodies, body)

		var req strictRequest
		if err := json.Unmarshal(body, &req); err != nil {
			strictError(w, "bad request: openai_error")
			return
		}

		toolsPresent := len(strings.TrimSpace(string(req.Tools))) > 0 &&
			string(strings.TrimSpace(string(req.Tools))) != "null"
		for _, m := range req.Messages {
			if m.Role != "assistant" {
				continue
			}
			if !m.hasContent() && !m.hasToolCalls() {
				strictError(w, "Invalid assistant message: content or tool_calls must be set")
				return
			}
			if toolsPresent && m.hasToolCalls() && !m.hasReasoningField() {
				strictError(w, "The reasoning_content in the thinking mode must be passed back to the API")
				return
			}
		}

		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprint(w, `{"id":"y","created":1,"model":"deepseek-v4-flash","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"The date is 2026-08-28."},"finish_reason":"stop"}],"usage":{"prompt_tokens":200,"completion_tokens":10,"total_tokens":210}}`)
			return
		}

		payload := payloads[min(calls, len(payloads)-1)]
		calls++

		w.Header().Set("Content-Type", "text/event-stream")
		rc := http.NewResponseController(w)
		for event := range strings.Lines(payload) {
			if strings.TrimSpace(event) == "" {
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", strings.TrimSpace(event))
			_ = rc.Flush()
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		_ = rc.Flush()
	}))
	t.Cleanup(srv.Close)
	return srv, &bodies
}

// A multi-step tool loop against the strict upstream: thinking + tool call on
// step one, plain answer on step two. The server hard-validates both rules on
// every request, so the loop completing at all proves the reasoning replay
// contract held (charmbracelet/crush#2696) and no empty assistant message was
// ever emitted (charmbracelet/crush#3794).
func TestStrictUpstream_ToolCallLoop(t *testing.T) {
	srv, bodies := serveStrictDeepSeek(t, deepseekThinkingToolCallSSE, cannedStopReplySSE)
	provider, err := New(WithBaseURL(srv.URL), WithAPIKey("x"))
	require.NoError(t, err)
	lm, err := provider.LanguageModel(context.Background(), "deepseek-v4-flash")
	require.NoError(t, err)

	var invocations []string
	getDate := fantasy.NewAgentTool(
		"get_date",
		"Get the current date.",
		func(_ context.Context, _ struct{}, _ fantasy.ToolCall) (fantasy.ToolResponse, error) {
			invocations = append(invocations, "get_date")
			return fantasy.NewTextResponse("2026-08-28"), nil
		},
	)

	agent := fantasy.NewAgent(lm, fantasy.WithTools(getDate))
	result, err := agent.Stream(context.Background(), fantasy.AgentStreamCall{Prompt: "what's the date?"})
	require.NoError(t, err, "the strict upstream rejected a request; the loop must run clean")
	require.NotNil(t, result)
	require.Equal(t, []string{"get_date"}, invocations)

	// Two steps: the tool-call turn and the final answer. The second request
	// is the replay: the tool-call assistant message must carry the
	// concatenated reasoning byte-for-byte.
	require.Len(t, *bodies, 2)
	var second strictRequest
	require.NoError(t, json.Unmarshal((*bodies)[1], &second))
	var assistant *strictMessage
	for i := range second.Messages {
		if second.Messages[i].Role == "assistant" {
			assistant = &second.Messages[i]
		}
	}
	require.NotNil(t, assistant, "no assistant message in second request: %s", (*bodies)[1])
	require.JSONEq(t, `"The user wants the date. I'll call get_date."`, string(assistant.ReasoningContent),
		"reasoning_content must round-trip byte-for-byte")
}

// A turn canceled while the model was still thinking lands in history as an
// assistant message whose only part is reasoning (charmbracelet/crush#3794,
// #3788). Replayed on the next turn it must be dropped, not sent: the strict
// upstream 400s the empty shape, and once it is in history every later
// request fails the same way. A completed tool-call turn in the same history
// must keep its reasoning_content.
func TestStrictUpstream_CanceledThinkingTurnReplayed(t *testing.T) {
	srv, bodies := serveStrictDeepSeek(t, cannedStopReplySSE)
	provider, err := New(WithBaseURL(srv.URL), WithAPIKey("x"))
	require.NoError(t, err)
	lm, err := provider.LanguageModel(context.Background(), "deepseek-v4-flash")
	require.NoError(t, err)

	prompt := fantasy.Prompt{
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ReasoningPart{Text: "Thinking out loud, then the user hit escape…"},
		}},
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "are you there?"}}},
	}

	parts := streamParts(t, lm, prompt)
	require.Zero(t, countType(parts, fantasy.StreamPartTypeError),
		"stream errored; the strict upstream 400s the reasoning-only replay shape")
	require.NotZero(t, countType(parts, fantasy.StreamPartTypeFinish),
		"stream must finish; the strict upstream 400s the reasoning-only replay shape")

	require.Len(t, *bodies, 1)
	var req strictRequest
	require.NoError(t, json.Unmarshal((*bodies)[0], &req))
	for i, m := range req.Messages {
		if m.Role != "assistant" {
			continue
		}
		require.True(t, m.hasContent() || m.hasToolCalls(),
			"assistant message %d went out with neither content nor tool calls: %s", i, (*bodies)[0])
	}
}
