package bedrock

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type capturedRequest struct {
	host    string
	path    string
	headers http.Header
	body    map[string]any
}

// hostRecorder keeps the host a request was addressed to before the test
// transport redirects it to the local server.
type hostRecorder struct{ next http.RoundTripper }

func (h hostRecorder) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set("X-Test-Original-Host", req.URL.Host)
	return h.next.RoundTrip(req)
}

func testClient(server *httptest.Server) *http.Client {
	return &http.Client{Transport: hostRecorder{next: redirectTransport(server.URL)}}
}

func newOpenAIServer(t *testing.T) (*httptest.Server, *[]capturedRequest) {
	t.Helper()
	var captured []capturedRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		captured = append(captured, capturedRequest{host: r.Header.Get("X-Test-Original-Host"), path: r.URL.Path, headers: r.Header.Clone(), body: body})

		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/chat/completions") {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": body["model"],
				"choices": []any{map[string]any{
					"index": 0, "finish_reason": "stop",
					"message": map[string]any{"role": "assistant", "content": "Hi"},
				}},
				"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "resp_1", "object": "response", "created_at": 0, "status": "completed", "model": body["model"],
			"output": []any{map[string]any{
				"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
				"content": []any{map[string]any{"type": "output_text", "text": "Hi", "annotations": []any{}}},
			}},
			"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
		})
	}))
	t.Cleanup(server.Close)
	return server, &captured
}

func generate(t *testing.T, p fantasy.Provider, modelID string, call fantasy.Call) {
	t.Helper()
	model, err := p.LanguageModel(t.Context(), modelID)
	require.NoError(t, err)
	assert.Equal(t, Name, model.Provider())
	call.Prompt = fantasy.Prompt{
		{Role: fantasy.MessageRoleSystem, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Be brief"}}},
		{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Hi"}}},
	}
	_, err = model.Generate(t.Context(), call)
	require.NoError(t, err)
}

func TestOpenAIModelsUseResponsesAPI(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-must-not-leak")
	t.Setenv("OPENAI_PROJECT_ID", "proj_must_not_leak")
	server, captured := newOpenAIServer(t)

	p, err := New(
		WithAPIKey("bedrock-key"),
		WithRegion("us-west-2"),
		WithHTTPClient(testClient(server)),
	)
	require.NoError(t, err)

	effort := openai.ReasoningEffortHigh
	generate(t, p, "us.openai.gpt-6-sol", fantasy.Call{
		ProviderOptions: fantasy.ProviderOptions{
			openai.Name: &openai.ResponsesProviderOptions{ReasoningEffort: &effort},
		},
	})

	require.Len(t, *captured, 1)
	req := (*captured)[0]
	assert.Equal(t, "bedrock-runtime.us-west-2.amazonaws.com", req.host)
	assert.Equal(t, "/openai/v1/responses", req.path)
	assert.Equal(t, "Bearer bedrock-key", req.headers.Get("Authorization"))
	assert.Empty(t, req.headers.Get("OpenAI-Project"))
	assert.Equal(t, "us.openai.gpt-6-sol", req.body["model"])
	assert.Equal(t, map[string]any{"effort": "high"}, req.body["reasoning"])
	input := req.body["input"].([]any)
	assert.Equal(t, "developer", input[0].(map[string]any)["role"])
}

func TestGPTOSSUsesChatCompletions(t *testing.T) {
	server, captured := newOpenAIServer(t)

	p, err := New(
		WithAPIKey("bedrock-key"),
		WithHTTPClient(testClient(server)),
	)
	require.NoError(t, err)

	generate(t, p, "openai.gpt-oss-120b-1:0", fantasy.Call{})

	require.Len(t, *captured, 1)
	assert.Equal(t, "bedrock-runtime.us-east-1.amazonaws.com", (*captured)[0].host)
	assert.Equal(t, "/openai/v1/chat/completions", (*captured)[0].path)
}

func TestOpenAIModelsSignWithAWSCredentials(t *testing.T) {
	t.Setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_CONFIG_FILE", "/dev/null")
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", "/dev/null")
	t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
	t.Setenv("OPENAI_API_KEY", "sk-must-not-leak")
	server, captured := newOpenAIServer(t)

	p, err := New(
		WithRegion("eu-west-1"),
		WithHTTPClient(testClient(server)),
	)
	require.NoError(t, err)

	generate(t, p, "global.openai.gpt-6-luna", fantasy.Call{})

	require.Len(t, *captured, 1)
	auth := (*captured)[0].headers.Get("Authorization")
	assert.True(t, strings.HasPrefix(auth, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"), auth)
	assert.Contains(t, auth, "/eu-west-1/bedrock/aws4_request")
	assert.Equal(t, "bedrock-runtime.eu-west-1.amazonaws.com", (*captured)[0].host)
}

func TestOpenAIModelsHonorBaseURL(t *testing.T) {
	server, captured := newOpenAIServer(t)

	p, err := New(
		WithAPIKey("bedrock-key"),
		WithRegion("us-east-1"),
		WithBaseURL("https://bedrock-runtime.us-east-1.amazonaws.com/"),
		WithHTTPClient(testClient(server)),
	)
	require.NoError(t, err)

	generate(t, p, "us.openai.gpt-6-astra", fantasy.Call{})

	require.Len(t, *captured, 1)
	assert.Equal(t, "/openai/v1/responses", (*captured)[0].path)
}

func TestAnthropicModelsStillUseAnthropicSDK(t *testing.T) {
	t.Parallel()
	var paths []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(mockAnthropicResponse())
	}))
	defer server.Close()

	p, err := New(
		WithAPIKey("k"),
		WithSkipAuth(true),
		WithHTTPClient(&http.Client{Transport: redirectTransport(server.URL)}),
	)
	require.NoError(t, err)

	generate(t, p, "us.anthropic.claude-sonnet-4-20250514-v1:0", fantasy.Call{})

	require.Len(t, paths, 1)
	assert.Contains(t, paths[0], "/model/us.anthropic.claude-sonnet-4-20250514-v1:0/invoke")
}
