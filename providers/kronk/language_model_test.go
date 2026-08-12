package kronk

import (
	"context"
	"errors"
	"io"
	"reflect"
	"testing"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

type fakeChatModel struct {
	chatResponse   model.ChatResponse
	chatErr        error
	streamResponse []model.ChatResponse
	streamErr      error
	document       model.D
	modelInfo      model.ModelInfo
}

func (f *fakeChatModel) Chat(_ context.Context, d model.D) (model.ChatResponse, error) {
	f.document = d
	return f.chatResponse, f.chatErr
}

func (f *fakeChatModel) ChatStreaming(_ context.Context, d model.D) (<-chan model.ChatResponse, error) {
	f.document = d
	if f.streamErr != nil {
		return nil, f.streamErr
	}

	ch := make(chan model.ChatResponse, len(f.streamResponse))
	for _, response := range f.streamResponse {
		ch <- response
	}
	close(ch)

	return ch, nil
}

func (f *fakeChatModel) ModelInfo() model.ModelInfo {
	return f.modelInfo
}

func TestPrepareDocumentToolChoice(t *testing.T) {
	lm := languageModel{
		provider:        Name,
		prepareCallFunc: DefaultPrepareCallFunc,
		toPromptFunc:    DefaultToPrompt,
	}
	tool := fantasy.FunctionTool{Name: "weather", InputSchema: map[string]any{"type": "object"}}

	tests := []struct {
		name       string
		toolChoice fantasy.ToolChoice
		want       any
	}{
		{name: "none", toolChoice: fantasy.ToolChoiceNone, want: "none"},
		{name: "auto", toolChoice: fantasy.ToolChoiceAuto, want: "auto"},
		{name: "required", toolChoice: fantasy.ToolChoiceRequired, want: "required"},
		{
			name:       "specific",
			toolChoice: fantasy.SpecificToolChoice("weather"),
			want: model.D{
				"type": "function",
				"function": model.D{
					"name": "weather",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d, _, err := lm.prepareDocument(fantasy.Call{
				Tools:      []fantasy.Tool{tool},
				ToolChoice: &tt.toolChoice,
			})
			if err != nil {
				t.Fatalf("prepareDocument: %v", err)
			}
			if got := d["tool_choice"]; !reflect.DeepEqual(got, tt.want) {
				t.Errorf("tool_choice: got %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestGenerateResponse(t *testing.T) {
	finishReason := model.FinishReasonTool
	client := fakeChatModel{
		modelInfo: model.ModelInfo{
			ID:            "loaded-model",
			Desc:          "test model",
			Type:          model.ModelTypeDense,
			HasProjection: true,
			Size:          1024,
			VRAMTotal:     768,
			SlotMemory:    256,
		},
		chatResponse: model.ChatResponse{
			Model:             "response-model",
			SystemFingerprint: "fp_kronk",
			Choices: []model.Choice{
				{
					FinishReasonPtr: &finishReason,
					Message: &model.ResponseMessage{
						Content: "calling weather",
						ToolCalls: []model.ResponseToolCall{
							{
								ID: "call-1",
								Function: model.ResponseToolCallFunction{
									Name:      "weather",
									Arguments: model.ToolCallArguments{"city": "Paris"},
								},
							},
						},
					},
				},
			},
			Usage: &model.Usage{
				PromptTokens:        12,
				PromptTokensDetails: model.PromptTokensDetails{CachedTokens: 3},
				CompletionTokens:    4,
				TotalTokens:         16,
				TokensPerSecond:     24.5,
				TimeToFirstTokenMS:  32.5,
				DraftTokens:         8,
				DraftAcceptedTokens: 6,
				DraftAcceptanceRate: 0.75,
				DraftCoverage:       0.5,
				DraftDisableReason:  "imc-hit",
			},
		},
	}
	lm := newLanguageModel("model", Name, &client)

	response, err := lm.Generate(t.Context(), fantasy.Call{})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got, want := response.Content.Text(), "calling weather"; got != want {
		t.Errorf("Content.Text: got %q, want %q", got, want)
	}
	toolCalls := response.Content.ToolCalls()
	if len(toolCalls) != 1 {
		t.Fatalf("Content.ToolCalls: got %d, want 1", len(toolCalls))
	}
	if got, want := toolCalls[0].Input, `{"city":"Paris"}`; got != want {
		t.Errorf("ToolCall.Input: got %q, want %q", got, want)
	}
	if got, want := response.Usage.TotalTokens, int64(16); got != want {
		t.Errorf("Usage.TotalTokens: got %d, want %d", got, want)
	}
	if got, want := response.Usage.CacheReadTokens, int64(3); got != want {
		t.Errorf("Usage.CacheReadTokens: got %d, want %d", got, want)
	}
	if got, want := response.FinishReason, fantasy.FinishReasonToolCalls; got != want {
		t.Errorf("FinishReason: got %q, want %q", got, want)
	}
	metadata, ok := response.ProviderMetadata[Name].(*ProviderMetadata)
	if !ok {
		t.Fatalf("ProviderMetadata: got %T, want *ProviderMetadata", response.ProviderMetadata[Name])
	}
	if got, want := metadata.Model, "response-model"; got != want {
		t.Errorf("ProviderMetadata.Model: got %q, want %q", got, want)
	}
	if got, want := metadata.SystemFingerprint, "fp_kronk"; got != want {
		t.Errorf("ProviderMetadata.SystemFingerprint: got %q, want %q", got, want)
	}
	if !metadata.HasProjection {
		t.Error("ProviderMetadata.HasProjection: got false, want true")
	}
	if got, want := metadata.TimeToFirstTokenMS, 32.5; got != want {
		t.Errorf("ProviderMetadata.TimeToFirstTokenMS: got %v, want %v", got, want)
	}
	if got, want := metadata.DraftAcceptedTokens, int64(6); got != want {
		t.Errorf("ProviderMetadata.DraftAcceptedTokens: got %d, want %d", got, want)
	}
}

func TestStreamResponse(t *testing.T) {
	toolFinish := model.FinishReasonTool
	client := fakeChatModel{
		modelInfo: model.ModelInfo{ID: "loaded-model", Type: model.ModelTypeDense},
		streamResponse: []model.ChatResponse{
			{Model: "response-model", SystemFingerprint: "fp_kronk", Choices: []model.Choice{{Delta: &model.ResponseMessage{Reasoning: "thinking"}}}},
			{Choices: []model.Choice{{Delta: &model.ResponseMessage{ToolCallDeltas: []model.ResponseToolCallDelta{
				{ID: "call-1", Index: 0, Function: model.ResponseToolCallDeltaFunction{Name: "weather"}},
			}}}}},
			{Choices: []model.Choice{{Delta: &model.ResponseMessage{ToolCallDeltas: []model.ResponseToolCallDelta{
				{Index: 0, Function: model.ResponseToolCallDeltaFunction{Arguments: `{"city":"Paris"}`}},
			}}}}},
			{Choices: []model.Choice{{Delta: &model.ResponseMessage{}, FinishReasonPtr: &toolFinish}}},
			{Choices: []model.Choice{}, Usage: &model.Usage{
				PromptTokens:        12,
				CompletionTokens:    4,
				TotalTokens:         16,
				TokensPerSecond:     20,
				TimeToFirstTokenMS:  30,
				DraftTokens:         6,
				DraftAcceptedTokens: 5,
			}},
		},
	}
	lm := newLanguageModel("model", Name, &client)

	stream, err := lm.Stream(t.Context(), fantasy.Call{})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var parts []fantasy.StreamPart
	for part := range stream {
		parts = append(parts, part)
	}

	wantTypes := []fantasy.StreamPartType{
		fantasy.StreamPartTypeReasoningStart,
		fantasy.StreamPartTypeReasoningDelta,
		fantasy.StreamPartTypeReasoningEnd,
		fantasy.StreamPartTypeToolInputStart,
		fantasy.StreamPartTypeToolInputDelta,
		fantasy.StreamPartTypeToolInputEnd,
		fantasy.StreamPartTypeToolCall,
		fantasy.StreamPartTypeFinish,
	}
	gotTypes := make([]fantasy.StreamPartType, len(parts))
	for i := range parts {
		gotTypes[i] = parts[i].Type
	}
	if !reflect.DeepEqual(gotTypes, wantTypes) {
		t.Fatalf("part types: got %v, want %v", gotTypes, wantTypes)
	}
	if got, want := parts[6].ToolCallInput, `{"city":"Paris"}`; got != want {
		t.Errorf("ToolCallInput: got %q, want %q", got, want)
	}
	if got, want := parts[7].Usage.TotalTokens, int64(16); got != want {
		t.Errorf("Usage.TotalTokens: got %d, want %d", got, want)
	}
	if got, want := parts[7].FinishReason, fantasy.FinishReasonToolCalls; got != want {
		t.Errorf("FinishReason: got %q, want %q", got, want)
	}
	metadata, ok := parts[7].ProviderMetadata[Name].(*ProviderMetadata)
	if !ok {
		t.Fatalf("ProviderMetadata: got %T, want *ProviderMetadata", parts[7].ProviderMetadata[Name])
	}
	if got, want := metadata.Model, "response-model"; got != want {
		t.Errorf("ProviderMetadata.Model: got %q, want %q", got, want)
	}
	if got, want := metadata.SystemFingerprint, "fp_kronk"; got != want {
		t.Errorf("ProviderMetadata.SystemFingerprint: got %q, want %q", got, want)
	}
	if got, want := metadata.TimeToFirstTokenMS, float64(30); got != want {
		t.Errorf("ProviderMetadata.TimeToFirstTokenMS: got %v, want %v", got, want)
	}
	if got, want := metadata.DraftAcceptedTokens, int64(5); got != want {
		t.Errorf("ProviderMetadata.DraftAcceptedTokens: got %d, want %d", got, want)
	}
	if got := client.document["stream_options"]; !reflect.DeepEqual(got, model.D{"include_usage": true}) {
		t.Errorf("stream_options: got %#v, want include_usage", got)
	}
}

func TestStreamIncomplete(t *testing.T) {
	tests := []struct {
		name    string
		ctx     context.Context
		wantErr error
	}{
		{name: "incomplete", ctx: t.Context(), wantErr: io.ErrUnexpectedEOF},
		{name: "canceled", ctx: canceledContext(), wantErr: context.Canceled},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lm := newLanguageModel("model", Name, &fakeChatModel{})
			stream, err := lm.Stream(tt.ctx, fantasy.Call{})
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}

			var parts []fantasy.StreamPart
			for part := range stream {
				parts = append(parts, part)
			}
			if len(parts) != 1 || parts[0].Type != fantasy.StreamPartTypeError {
				t.Fatalf("parts: got %#v, want one error", parts)
			}
			if !errors.Is(parts[0].Error, tt.wantErr) {
				t.Errorf("Error: got %v, want %v", parts[0].Error, tt.wantErr)
			}
		})
	}
}

func TestStreamTerminalWithoutDelta(t *testing.T) {
	finishReason := model.FinishReasonLength
	client := fakeChatModel{
		streamResponse: []model.ChatResponse{
			{Choices: []model.Choice{{FinishReasonPtr: &finishReason}}},
		},
	}
	lm := newLanguageModel("model", Name, &client)

	stream, err := lm.Stream(t.Context(), fantasy.Call{})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	var parts []fantasy.StreamPart
	for part := range stream {
		parts = append(parts, part)
	}
	if len(parts) != 1 || parts[0].Type != fantasy.StreamPartTypeFinish {
		t.Fatalf("parts: got %#v, want one finish", parts)
	}
	if got, want := parts[0].FinishReason, fantasy.FinishReasonLength; got != want {
		t.Errorf("FinishReason: got %q, want %q", got, want)
	}
}

func canceledContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}
