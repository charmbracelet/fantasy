package kronk

import (
	"reflect"
	"testing"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

func TestPrepareDocumentSampling(t *testing.T) {
	lm := languageModel{
		provider:        Name,
		prepareCallFunc: DefaultPrepareCallFunc,
		toPromptFunc:    DefaultToPrompt,
	}
	call := fantasy.Call{
		TopK:             new(int64(20)),
		FrequencyPenalty: new(0.2),
		PresencePenalty:  new(0.1),
	}

	d, warnings, err := lm.prepareDocument(call)
	if err != nil {
		t.Fatalf("prepareDocument: %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings: got %v, want none", warnings)
	}

	tests := []struct {
		name string
		key  string
		want any
	}{
		{name: "top-k", key: "top_k", want: int64(20)},
		{name: "frequency penalty", key: "frequency_penalty", want: 0.2},
		{name: "presence penalty", key: "presence_penalty", want: 0.1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := d[tt.key]; got != tt.want {
				t.Errorf("%s: got %v, want %v", tt.key, got, tt.want)
			}
		})
	}
}

func TestDefaultPrepareCallFuncSampling(t *testing.T) {
	providerOptions := ProviderOptions{
		AdaptivePDecay:  new(0.9),
		AdaptivePTarget: new(0.1),
		DryAllowedLen:   new(int64(3)),
		DryBase:         new(1.75),
		DryMultiplier:   new(1.1),
		DryPenaltyLast:  new(int64(64)),
		Grammar:         new("root ::= \"yes\" | \"no\""),
		Logprobs:        new(true),
		MinP:            new(0.05),
		NumPredict:      new(int64(512)),
		ReasoningEffort: new("high"),
		RepeatLastN:     new(int64(32)),
		RepeatPenalty:   new(1.15),
		Seed:            new(int64(42)),
		Stop:            []string{"STOP"},
		Thinking:        new(false),
		TopK:            new(int64(20)),
		TopLogprobs:     new(int64(5)),
		XtcMinKeep:      new(int64(2)),
		XtcProbability:  new(0.5),
		XtcThreshold:    new(0.1),
	}
	call := fantasy.Call{
		ProviderOptions: NewProviderOptions(&providerOptions),
	}
	d := model.D{}

	warnings, err := DefaultPrepareCallFunc(nil, d, call)
	if err != nil {
		t.Fatalf("DefaultPrepareCallFunc: %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings: got %v, want none", warnings)
	}

	want := model.D{
		"adaptive_p_decay":   0.9,
		"adaptive_p_target":  0.1,
		"dry_allowed_length": int64(3),
		"dry_base":           1.75,
		"dry_multiplier":     1.1,
		"dry_penalty_last_n": int64(64),
		"enable_thinking":    false,
		"grammar":            "root ::= \"yes\" | \"no\"",
		"logprobs":           true,
		"max_tokens":         int64(512),
		"min_p":              0.05,
		"reasoning_effort":   "high",
		"repeat_last_n":      int64(32),
		"repeat_penalty":     1.15,
		"seed":               int64(42),
		"stop":               []string{"STOP"},
		"top_k":              int64(20),
		"top_logprobs":       int64(5),
		"xtc_min_keep":       int64(2),
		"xtc_probability":    0.5,
		"xtc_threshold":      0.1,
	}
	if !reflect.DeepEqual(d, want) {
		t.Errorf("document: got %#v, want %#v", d, want)
	}
}

func TestDefaultMapFinishReasonFunc(t *testing.T) {
	tests := []struct {
		name         string
		finishReason string
		want         fantasy.FinishReason
	}{
		{name: "stop", finishReason: model.FinishReasonStop, want: fantasy.FinishReasonStop},
		{name: "length", finishReason: model.FinishReasonLength, want: fantasy.FinishReasonLength},
		{name: "tool", finishReason: model.FinishReasonTool, want: fantasy.FinishReasonToolCalls},
		{name: "error", finishReason: model.FinishReasonError, want: fantasy.FinishReasonError},
		{name: "unknown", finishReason: "", want: fantasy.FinishReasonUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DefaultMapFinishReasonFunc(tt.finishReason); got != tt.want {
				t.Errorf("DefaultMapFinishReasonFunc: got %q, want %q", got, tt.want)
			}
		})
	}
}
