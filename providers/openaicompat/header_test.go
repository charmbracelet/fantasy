package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/require"
)

// prismHeaderFunc copies the Hyper Prism router headers and trailers into
// the provider metadata extra fields, mirroring how Crush consumes them.
func prismHeaderFunc(header http.Header, metadata *openai.ProviderMetadata) {
	for key, field := range map[string]string{
		"X-Prism-Model-Id":            "x-prism-model-id",
		"X-Prism-Model-Name":          "x-prism-model-name",
		"X-Prism-Hypercredit-Savings": "x-prism-hypercredit-savings",
	} {
		if v := header.Get(key); v != "" {
			if metadata.ExtraFields == nil {
				metadata.ExtraFields = make(map[string]json.RawMessage)
			}
			metadata.ExtraFields[field] = json.RawMessage(strconv.Quote(v))
		}
	}
}

// servePrism answers both streaming and non-streaming chat completion
// requests with canned payloads carrying the Prism router headers.
func servePrism(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Prism-Model-Id", "prism-42")
		w.Header().Set("X-Prism-Model-Name", "GPT-5.2 Codex Max")
		var reqShape struct {
			Stream bool `json:"stream"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &reqShape)
		if reqShape.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"z","created":1,"model":"prism","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func prompt() fantasy.Prompt {
	return fantasy.Prompt{
		{
			Role:    fantasy.MessageRoleUser,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hello"}},
		},
	}
}

func prismProvider(t *testing.T, srv *httptest.Server) fantasy.Provider {
	t.Helper()
	provider, err := New(
		WithBaseURL(srv.URL),
		WithAPIKey("test"),
		WithLanguageModelOptions(openai.WithLanguageModelHeaderFunc(prismHeaderFunc)),
	)
	require.NoError(t, err)
	return provider
}

func requirePrismFields(t *testing.T, metadata fantasy.ProviderMetadata) {
	t.Helper()
	data, ok := metadata[openai.Name]
	require.True(t, ok, "expected openai provider metadata")
	providerMetadata, ok := data.(*openai.ProviderMetadata)
	require.True(t, ok, "expected *openai.ProviderMetadata")
	require.NotNil(t, providerMetadata.ExtraFields)

	var modelID string
	ok = providerMetadata.ExtraField("x-prism-model-id", &modelID)
	require.True(t, ok)
	require.Equal(t, "prism-42", modelID)

	var modelName string
	ok = providerMetadata.ExtraField("x-prism-model-name", &modelName)
	require.True(t, ok)
	require.Equal(t, "GPT-5.2 Codex Max", modelName)
}

func TestResponseHeaders_Generate(t *testing.T) {
	t.Parallel()
	srv := servePrism(t)
	provider := prismProvider(t, srv)
	lm, err := provider.LanguageModel(context.Background(), "prism")
	require.NoError(t, err)

	response, err := lm.Generate(context.Background(), fantasy.Call{Prompt: prompt()})
	require.NoError(t, err)
	requirePrismFields(t, response.ProviderMetadata)
}

func TestResponseHeaders_Stream(t *testing.T) {
	t.Parallel()
	srv := servePrism(t)
	provider := prismProvider(t, srv)
	lm, err := provider.LanguageModel(context.Background(), "prism")
	require.NoError(t, err)

	stream, err := lm.Stream(context.Background(), fantasy.Call{Prompt: prompt()})
	require.NoError(t, err)
	var finish *fantasy.StreamPart
	for part := range stream {
		if part.Type == fantasy.StreamPartTypeFinish {
			finish = &part
		}
	}
	require.NotNil(t, finish, "expected a finish part")
	requirePrismFields(t, finish.ProviderMetadata)
}

func TestResponseHeaders_StreamWithoutUsage(t *testing.T) {
	t.Parallel()
	// A stream whose chunks carry no usage leaves the stream metadata nil
	// after the loop; the header func must still surface the headers.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Prism-Model-Id", "prism-42")
		w.Header().Set("X-Prism-Model-Name", "GPT-5.2 Codex Max")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)
	provider := prismProvider(t, srv)
	lm, err := provider.LanguageModel(context.Background(), "prism")
	require.NoError(t, err)

	stream, err := lm.Stream(context.Background(), fantasy.Call{Prompt: prompt()})
	require.NoError(t, err)
	var finish *fantasy.StreamPart
	for part := range stream {
		if part.Type == fantasy.StreamPartTypeFinish {
			finish = &part
		}
	}
	require.NotNil(t, finish)
	require.Zero(t, finish.Usage.TotalTokens)
	requirePrismFields(t, finish.ProviderMetadata)
}

// serveResponses answers Responses API requests (streaming and
// non-streaming) with canned payloads carrying the Prism router headers.
func serveResponses(t *testing.T) *httptest.Server {
	t.Helper()
	const responseBody = `{"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"gpt-5","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hi","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"output_tokens_details":{},"total_tokens":2}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Prism-Model-Id", "prism-42")
		w.Header().Set("X-Prism-Model-Name", "GPT-5.2 Codex Max")
		var reqShape struct {
			Stream bool `json:"stream"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &reqShape)
		if reqShape.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			rc := http.NewResponseController(w)
			events := []string{
				`{"type":"response.created","response":` + responseBody + `}`,
				`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}`,
				`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hi"}`,
				`{"type":"response.completed","response":` + responseBody + `}`,
			}
			for _, event := range events {
				fmt.Fprintf(w, "event: %s\ndata: %s\n\n", strings.SplitN(event, ":", 2)[0][2:], event)
				_ = rc.Flush()
			}
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(responseBody))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func prismResponsesProvider(t *testing.T, srv *httptest.Server) fantasy.Provider {
	t.Helper()
	provider, err := New(
		WithBaseURL(srv.URL),
		WithAPIKey("test"),
		WithUseResponsesAPI(),
		WithLanguageModelOptions(openai.WithLanguageModelHeaderFunc(prismHeaderFunc)),
	)
	require.NoError(t, err)
	return provider
}

func requirePrismResponsesFields(t *testing.T, metadata fantasy.ProviderMetadata) {
	t.Helper()
	data, ok := metadata[openai.Name]
	require.True(t, ok, "expected openai provider metadata")
	responsesMetadata, ok := data.(*openai.ResponsesProviderMetadata)
	require.True(t, ok, "expected *openai.ResponsesProviderMetadata, got %T", data)
	require.NotNil(t, responsesMetadata.ExtraFields)

	var modelID string
	ok = responsesMetadata.ExtraField("x-prism-model-id", &modelID)
	require.True(t, ok)
	require.Equal(t, "prism-42", modelID)

	var modelName string
	ok = responsesMetadata.ExtraField("x-prism-model-name", &modelName)
	require.True(t, ok)
	require.Equal(t, "GPT-5.2 Codex Max", modelName)
}

func TestResponseHeaders_ResponsesGenerate(t *testing.T) {
	t.Parallel()
	srv := serveResponses(t)
	provider := prismResponsesProvider(t, srv)
	lm, err := provider.LanguageModel(context.Background(), "gpt-5")
	require.NoError(t, err)

	response, err := lm.Generate(context.Background(), fantasy.Call{Prompt: prompt()})
	require.NoError(t, err)
	requirePrismResponsesFields(t, response.ProviderMetadata)
}

func TestResponseHeaders_ResponsesStream(t *testing.T) {
	t.Parallel()
	srv := serveResponses(t)
	provider := prismResponsesProvider(t, srv)
	lm, err := provider.LanguageModel(context.Background(), "gpt-5")
	require.NoError(t, err)

	stream, err := lm.Stream(context.Background(), fantasy.Call{Prompt: prompt()})
	require.NoError(t, err)
	var finish *fantasy.StreamPart
	for part := range stream {
		if part.Type == fantasy.StreamPartTypeFinish {
			finish = &part
		}
	}
	require.NotNil(t, finish, "expected a finish part")
	requirePrismResponsesFields(t, finish.ProviderMetadata)
}

// servePrismTrailers answers chat completion requests carrying the Prism
// savings as declared HTTP trailers, which net/http populates on the
// client only after the response body is fully consumed.
func servePrismTrailers(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var reqShape struct {
			Stream bool `json:"stream"`
		}
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &reqShape)
		w.Header().Set("Trailer", "X-Prism-Hypercredit-Savings")
		if reqShape.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			rc := http.NewResponseController(w)
			_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n"))
			_ = rc.Flush()
			_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
			_ = rc.Flush()
			_, _ = w.Write([]byte("data: {\"id\":\"z\",\"created\":1,\"model\":\"prism\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n"))
			_ = rc.Flush()
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			_ = rc.Flush()
			// The savings trailer is written only after usage is final, which
			// on a real server happens after the DONE sentinel has been sent:
			// pricing and stats run in between. Give the client time to observe
			// DONE first, so the test reproduces a trailer that arrives after
			// the stream's terminal event.
			time.Sleep(100 * time.Millisecond)
			w.Header().Set("X-Prism-Hypercredit-Savings", "1.5")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"z","created":1,"model":"prism","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
		w.Header().Set("X-Prism-Hypercredit-Savings", "1.5")
	}))
	t.Cleanup(srv.Close)
	return srv
}

func requirePrismTrailerField(t *testing.T, metadata fantasy.ProviderMetadata) {
	t.Helper()
	data, ok := metadata[openai.Name]
	require.True(t, ok, "expected openai provider metadata")
	providerMetadata, ok := data.(*openai.ProviderMetadata)
	require.True(t, ok, "expected *openai.ProviderMetadata")

	var savings string
	ok = providerMetadata.ExtraField("x-prism-hypercredit-savings", &savings)
	require.True(t, ok)
	require.Equal(t, "1.5", savings)
}

func TestResponseTrailers_Generate(t *testing.T) {
	t.Parallel()
	srv := servePrismTrailers(t)
	provider := prismProvider(t, srv)
	lm, err := provider.LanguageModel(context.Background(), "prism")
	require.NoError(t, err)

	response, err := lm.Generate(context.Background(), fantasy.Call{Prompt: prompt()})
	require.NoError(t, err)
	requirePrismTrailerField(t, response.ProviderMetadata)
}

func TestResponseTrailers_Stream(t *testing.T) {
	t.Parallel()
	srv := servePrismTrailers(t)
	provider := prismProvider(t, srv)
	lm, err := provider.LanguageModel(context.Background(), "prism")
	require.NoError(t, err)

	stream, err := lm.Stream(context.Background(), fantasy.Call{Prompt: prompt()})
	require.NoError(t, err)
	var finish *fantasy.StreamPart
	for part := range stream {
		if part.Type == fantasy.StreamPartTypeFinish {
			finish = &part
		}
	}
	require.NotNil(t, finish, "expected a finish part")
	requirePrismTrailerField(t, finish.ProviderMetadata)
}
