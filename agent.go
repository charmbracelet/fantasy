package ai

import (
	"context"
)

type StepResponse struct {
	Response
	// Messages generated during this step
	Messages []Message
}

type StepCondition = func(steps []StepResponse) bool

type PrepareStepFunctionOptions struct {
	Steps      []StepResponse
	StepNumber int
	Model      LanguageModel
	Messages   []Message
}

type PrepareStepResult struct {
	SystemPrompt string
	Model        LanguageModel
	Messages     []Message
}

type PrepareStepFunction = func(options PrepareStepFunctionOptions) PrepareStepResult

type OnStepFinishedFunction = func(step StepResponse)

type AgentSettings struct {
	Call
	Model LanguageModel

	StopWhen       []StepCondition
	PrepareStep    PrepareStepFunction
	OnStepFinished OnStepFinishedFunction
}

type Agent interface {
	Generate(context.Context, Call) (*Response, error)
	Stream(context.Context, Call) (StreamResponse, error)
}

type agentOption = func(*AgentSettings)

type agent struct {
	settings AgentSettings
}

func NewAgent(model LanguageModel, opts ...agentOption) Agent {
	settings := AgentSettings{
		Model: model,
	}
	for _, o := range opts {
		o(&settings)
	}
	return &agent{
		settings: settings,
	}
}

func mergeCall(agentOpts, opts Call) Call {
	if len(opts.Prompt) > 0 {
		agentOpts.Prompt = opts.Prompt
	}
	if opts.MaxOutputTokens != nil {
		agentOpts.MaxOutputTokens = opts.MaxOutputTokens
	}
	if opts.Temperature != nil {
		agentOpts.Temperature = opts.Temperature
	}
	if opts.TopP != nil {
		agentOpts.TopP = opts.TopP
	}
	if opts.TopK != nil {
		agentOpts.TopK = opts.TopK
	}
	if opts.PresencePenalty != nil {
		agentOpts.PresencePenalty = opts.PresencePenalty
	}
	if opts.FrequencyPenalty != nil {
		agentOpts.FrequencyPenalty = opts.FrequencyPenalty
	}
	if opts.Tools != nil {
		agentOpts.Tools = opts.Tools
	}
	if opts.Headers != nil {
		agentOpts.Headers = opts.Headers
	}
	if opts.ProviderOptions != nil {
		agentOpts.ProviderOptions = opts.ProviderOptions
	}
	return agentOpts
}

// Generate implements Agent.
func (a *agent) Generate(ctx context.Context, opts Call) (*Response, error) {
	// TODO: implement the agentic stuff
	return a.settings.Model.Generate(ctx, mergeCall(a.settings.Call, opts))
}

// Stream implements Agent.
func (a *agent) Stream(ctx context.Context, opts Call) (StreamResponse, error) {
	// TODO: implement the agentic stuff
	return a.settings.Model.Stream(ctx, mergeCall(a.settings.Call, opts))
}
