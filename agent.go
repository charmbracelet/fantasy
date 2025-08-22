package ai

import (
	"context"
	"errors"
	"maps"
	"slices"
	"sync"

	"github.com/charmbracelet/crush/internal/llm/tools"
)

type StepResult struct {
	Response
	// Messages generated during this step
	Messages []Message
}

type StopCondition = func(steps []StepResult) bool

type PrepareStepFunctionOptions struct {
	Steps      []StepResult
	StepNumber int
	Model      LanguageModel
	Messages   []Message
}

type PrepareStepResult struct {
	Model    LanguageModel
	Messages []Message
}

type (
	PrepareStepFunction    = func(options PrepareStepFunctionOptions) PrepareStepResult
	OnStepFinishedFunction = func(step StepResult)
	RepairToolCall         = func(ToolCallContent) ToolCallContent
)

type AgentSettings struct {
	systemPrompt     string
	maxOutputTokens  *int64
	temperature      *float64
	topP             *float64
	topK             *int64
	presencePenalty  *float64
	frequencyPenalty *float64
	headers          map[string]string
	providerOptions  ProviderOptions

	// TODO: add support for provider tools
	tools      []tools.BaseTool
	maxRetries *int

	model LanguageModel

	stopWhen       []StopCondition
	prepareStep    PrepareStepFunction
	repairToolCall RepairToolCall
	onStepFinished OnStepFinishedFunction
	onRetry        OnRetryCallback
}

type AgentCall struct {
	Prompt           string     `json:"prompt"`
	Files            []FilePart `json:"files"`
	Messages         []Message  `json:"messages"`
	MaxOutputTokens  *int64
	Temperature      *float64 `json:"temperature"`
	TopP             *float64 `json:"top_p"`
	TopK             *int64   `json:"top_k"`
	PresencePenalty  *float64 `json:"presence_penalty"`
	FrequencyPenalty *float64 `json:"frequency_penalty"`
	ActiveTools      []string `json:"active_tools"`
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	OnRetry          OnRetryCallback
	MaxRetries       *int

	StopWhen       []StopCondition
	PrepareStep    PrepareStepFunction
	RepairToolCall RepairToolCall
	OnStepFinished OnStepFinishedFunction
}

type AgentResult struct {
	Steps []StepResult
	// Final response
	Response   Response
	TotalUsage Usage
}

type Agent interface {
	Generate(context.Context, AgentCall) (*AgentResult, error)
	Stream(context.Context, AgentCall) (StreamResponse, error)
}

type agentOption = func(*AgentSettings)

type agent struct {
	settings AgentSettings
}

func NewAgent(model LanguageModel, opts ...agentOption) Agent {
	settings := AgentSettings{
		model: model,
	}
	for _, o := range opts {
		o(&settings)
	}
	return &agent{
		settings: settings,
	}
}

func (a *agent) prepareCall(call AgentCall) AgentCall {
	if call.MaxOutputTokens == nil && a.settings.maxOutputTokens != nil {
		call.MaxOutputTokens = a.settings.maxOutputTokens
	}
	if call.Temperature == nil && a.settings.temperature != nil {
		call.Temperature = a.settings.temperature
	}
	if call.TopP == nil && a.settings.topP != nil {
		call.TopP = a.settings.topP
	}
	if call.TopK == nil && a.settings.topK != nil {
		call.TopK = a.settings.topK
	}
	if call.PresencePenalty == nil && a.settings.presencePenalty != nil {
		call.PresencePenalty = a.settings.presencePenalty
	}
	if call.FrequencyPenalty == nil && a.settings.frequencyPenalty != nil {
		call.FrequencyPenalty = a.settings.frequencyPenalty
	}
	if len(call.StopWhen) == 0 && len(a.settings.stopWhen) > 0 {
		call.StopWhen = a.settings.stopWhen
	}
	if call.PrepareStep == nil && a.settings.prepareStep != nil {
		call.PrepareStep = a.settings.prepareStep
	}
	if call.RepairToolCall == nil && a.settings.repairToolCall != nil {
		call.RepairToolCall = a.settings.repairToolCall
	}
	if call.OnStepFinished == nil && a.settings.onStepFinished != nil {
		call.OnStepFinished = a.settings.onStepFinished
	}
	if call.OnRetry == nil && a.settings.onRetry != nil {
		call.OnRetry = a.settings.onRetry
	}
	if call.MaxRetries == nil && a.settings.maxRetries != nil {
		call.MaxRetries = a.settings.maxRetries
	}

	providerOptions := ProviderOptions{}
	if a.settings.providerOptions != nil {
		maps.Copy(providerOptions, a.settings.providerOptions)
	}
	if call.ProviderOptions != nil {
		maps.Copy(providerOptions, call.ProviderOptions)
	}
	call.ProviderOptions = providerOptions

	headers := map[string]string{}

	if a.settings.headers != nil {
		maps.Copy(headers, a.settings.headers)
	}

	if call.Headers != nil {
		maps.Copy(headers, call.Headers)
	}
	call.Headers = headers
	return call
}

// Generate implements Agent.
func (a *agent) Generate(ctx context.Context, opts AgentCall) (*AgentResult, error) {
	opts = a.prepareCall(opts)
	initialPrompt, err := a.createPrompt(a.settings.systemPrompt, opts.Prompt, opts.Messages, opts.Files...)
	if err != nil {
		return nil, err
	}
	var responseMessages []Message
	var steps []StepResult

	for {
		stepInputMessages := append(initialPrompt, responseMessages...)
		stepModel := a.settings.model
		if opts.PrepareStep != nil {
			prepared := opts.PrepareStep(PrepareStepFunctionOptions{
				Model:      stepModel,
				Steps:      steps,
				StepNumber: len(steps),
				Messages:   stepInputMessages,
			})
			stepInputMessages = prepared.Messages
			if prepared.Model != nil {
				stepModel = prepared.Model
			}
		}

		preparedTools := a.prepareTools(a.settings.tools, opts.ActiveTools)

		toolChoice := ToolChoiceAuto
		retryOptions := DefaultRetryOptions()
		retryOptions.OnRetry = opts.OnRetry
		retry := RetryWithExponentialBackoffRespectingRetryHeaders[*Response](retryOptions)

		result, err := retry(ctx, func() (*Response, error) {
			return stepModel.Generate(ctx, Call{
				Prompt:           stepInputMessages,
				MaxOutputTokens:  opts.MaxOutputTokens,
				Temperature:      opts.Temperature,
				TopP:             opts.TopP,
				TopK:             opts.TopK,
				PresencePenalty:  opts.PresencePenalty,
				FrequencyPenalty: opts.FrequencyPenalty,
				Tools:            preparedTools,
				ToolChoice:       &toolChoice,
				Headers:          opts.Headers,
				ProviderOptions:  opts.ProviderOptions,
			})
		})
		if err != nil {
			return nil, err
		}

		var stepToolCalls []ToolCallContent
		for _, content := range result.Content {
			if content.GetType() == ContentTypeToolCall {
				toolCall, ok := AsContentType[ToolCallContent](content)
				if !ok {
					continue
				}
				stepToolCalls = append(stepToolCalls, toolCall)
			}
		}

		toolResults, err := a.executeTools(ctx, a.settings.tools, stepToolCalls)

		stepContent := result.Content
		for _, result := range toolResults {
			stepContent = append(stepContent, result)
		}
		currentStepMessages := toResponseMessages(stepContent)
		responseMessages = append(responseMessages, currentStepMessages...)

		stepResult := StepResult{
			Response: Response{
				Content:          stepContent,
				FinishReason:     result.FinishReason,
				Usage:            result.Usage,
				Warnings:         result.Warnings,
				ProviderMetadata: result.ProviderMetadata,
			},
			Messages: currentStepMessages,
		}
		steps = append(steps, stepResult)
		if opts.OnStepFinished != nil {
			opts.OnStepFinished(stepResult)
		}

		shouldStop := isStopConditionMet(opts.StopWhen, steps)

		if shouldStop || err != nil || len(stepToolCalls) == 0 || result.FinishReason != FinishReasonToolCalls {
			break
		}
	}

	totalUsage := Usage{}

	for _, step := range steps {
		usage := step.Usage
		totalUsage.InputTokens += usage.InputTokens
		totalUsage.OutputTokens += usage.OutputTokens
		totalUsage.ReasoningTokens += usage.ReasoningTokens
		totalUsage.CacheCreationTokens += usage.CacheCreationTokens
		totalUsage.CacheReadTokens += usage.CacheReadTokens
		totalUsage.TotalTokens += usage.TotalTokens
	}

	agentResult := &AgentResult{
		Steps:      steps,
		Response:   steps[len(steps)-1].Response,
		TotalUsage: totalUsage,
	}
	return agentResult, nil
}

func isStopConditionMet(conditions []StopCondition, steps []StepResult) bool {
	if len(conditions) == 0 {
		return false
	}

	for _, condition := range conditions {
		if condition(steps) {
			return true
		}
	}
	return false
}

func toResponseMessages(content []Content) []Message {
	var assistantParts []MessagePart
	var toolParts []MessagePart

	for _, c := range content {
		switch c.GetType() {
		case ContentTypeText:
			text, ok := AsContentType[TextContent](c)
			if !ok {
				continue
			}
			assistantParts = append(assistantParts, TextPart{
				Text:            text.Text,
				ProviderOptions: ProviderOptions(text.ProviderMetadata),
			})
		case ContentTypeReasoning:
			reasoning, ok := AsContentType[ReasoningContent](c)
			if !ok {
				continue
			}
			assistantParts = append(assistantParts, ReasoningPart{
				Text:            reasoning.Text,
				ProviderOptions: ProviderOptions(reasoning.ProviderMetadata),
			})
		case ContentTypeToolCall:
			toolCall, ok := AsContentType[ToolCallContent](c)
			if !ok {
				continue
			}
			assistantParts = append(assistantParts, ToolCallPart{
				ToolCallID:       toolCall.ToolCallID,
				ToolName:         toolCall.ToolName,
				Input:            toolCall.Input,
				ProviderExecuted: toolCall.ProviderExecuted,
				ProviderOptions:  ProviderOptions(toolCall.ProviderMetadata),
			})
		case ContentTypeFile:
			file, ok := AsContentType[FileContent](c)
			if !ok {
				continue
			}
			assistantParts = append(assistantParts, FilePart{
				Data:            file.Data,
				MediaType:       file.MediaType,
				ProviderOptions: ProviderOptions(file.ProviderMetadata),
			})
		case ContentTypeToolResult:
			result, ok := AsContentType[ToolResultContent](c)
			if !ok {
				continue
			}
			toolParts = append(toolParts, ToolResultPart{
				ToolCallID:      result.ToolCallID,
				Output:          result.Result,
				ProviderOptions: ProviderOptions(result.ProviderMetadata),
			})
		}
	}

	var messages []Message
	if len(assistantParts) > 0 {
		messages = append(messages, Message{
			Role:    MessageRoleAssistant,
			Content: assistantParts,
		})
	}
	if len(toolParts) > 0 {
		messages = append(messages, Message{
			Role:    MessageRoleTool,
			Content: toolParts,
		})
	}
	return messages
}

func (a *agent) executeTools(ctx context.Context, allTools []tools.BaseTool, toolCalls []ToolCallContent) ([]ToolResultContent, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}

	// Create a map for quick tool lookup
	toolMap := make(map[string]tools.BaseTool)
	for _, tool := range allTools {
		toolMap[tool.Info().Name] = tool
	}

	// Execute all tool calls in parallel
	results := make([]ToolResultContent, len(toolCalls))
	var toolExecutionError error
	var wg sync.WaitGroup

	for i, toolCall := range toolCalls {
		wg.Add(1)
		go func(index int, call ToolCallContent) {
			defer wg.Done()

			tool, exists := toolMap[call.ToolName]
			if !exists {
				results[index] = ToolResultContent{
					ToolCallID: call.ToolCallID,
					ToolName:   call.ToolName,
					Result: ToolResultOutputContentError{
						Error: errors.New("Error: Tool not found: " + call.ToolName),
					},
					ProviderExecuted: false,
				}
				return
			}

			// Execute the tool
			result, err := tool.Run(ctx, tools.ToolCall{
				ID:    call.ToolCallID,
				Name:  call.ToolName,
				Input: call.Input,
			})
			if err != nil {
				results[index] = ToolResultContent{
					ToolCallID: call.ToolCallID,
					ToolName:   call.ToolName,
					Result: ToolResultOutputContentError{
						Error: err,
					},
					ProviderExecuted: false,
				}
				toolExecutionError = err
				return
			}

			if result.IsError {
				results[index] = ToolResultContent{
					ToolCallID: call.ToolCallID,
					ToolName:   call.ToolName,
					Result: ToolResultOutputContentError{
						Error: errors.New(result.Content),
					},
					ProviderExecuted: false,
				}
			} else {
				results[index] = ToolResultContent{
					ToolCallID: call.ToolCallID,
					ToolName:   toolCall.ToolName,
					Result: ToolResultOutputContentText{
						Text: result.Content,
					},
					ProviderExecuted: false,
				}
			}
		}(i, toolCall)
	}

	// Wait for all tool executions to complete
	wg.Wait()

	return results, toolExecutionError
}

// Stream implements Agent.
func (a *agent) Stream(ctx context.Context, opts AgentCall) (StreamResponse, error) {
	// TODO: implement the agentic stuff
	panic("not implemented")
}

func (a *agent) prepareTools(tools []tools.BaseTool, activeTools []string) []Tool {
	var preparedTools []Tool
	for _, tool := range tools {
		if len(activeTools) > 0 && !slices.Contains(activeTools, tool.Info().Name) {
			continue
		}
		info := tool.Info()
		preparedTools = append(preparedTools, FunctionTool{
			Name:        info.Name,
			Description: info.Description,
			InputSchema: map[string]any{
				"type":       "object",
				"properties": info.Parameters,
				"required":   info.Required,
			},
		})
	}
	return preparedTools
}

func (a *agent) createPrompt(system, prompt string, messages []Message, files ...FilePart) (Prompt, error) {
	if prompt == "" {
		return nil, NewInvalidPromptError(prompt, "Prompt can't be empty", nil)
	}

	var preparedPrompt Prompt

	if system != "" {
		preparedPrompt = append(preparedPrompt, NewSystemMessage(system))
	}

	preparedPrompt = append(preparedPrompt, NewUserMessage(prompt, files...))
	preparedPrompt = append(preparedPrompt, messages...)
	return preparedPrompt, nil
}

func WithSystemPrompt(prompt string) agentOption {
	return func(s *AgentSettings) {
		s.systemPrompt = prompt
	}
}

func WithMaxOutputTokens(tokens int64) agentOption {
	return func(s *AgentSettings) {
		s.maxOutputTokens = &tokens
	}
}

func WithTemperature(temp float64) agentOption {
	return func(s *AgentSettings) {
		s.temperature = &temp
	}
}

func WithTopP(topP float64) agentOption {
	return func(s *AgentSettings) {
		s.topP = &topP
	}
}

func WithTopK(topK int64) agentOption {
	return func(s *AgentSettings) {
		s.topK = &topK
	}
}

func WithPresencePenalty(penalty float64) agentOption {
	return func(s *AgentSettings) {
		s.presencePenalty = &penalty
	}
}

func WithFrequencyPenalty(penalty float64) agentOption {
	return func(s *AgentSettings) {
		s.frequencyPenalty = &penalty
	}
}

func WithTools(tools ...tools.BaseTool) agentOption {
	return func(s *AgentSettings) {
		s.tools = append(s.tools, tools...)
	}
}

func WithStopConditions(conditions ...StopCondition) agentOption {
	return func(s *AgentSettings) {
		s.stopWhen = append(s.stopWhen, conditions...)
	}
}

func WithPrepareStep(fn PrepareStepFunction) agentOption {
	return func(s *AgentSettings) {
		s.prepareStep = fn
	}
}

func WithRepairToolCall(fn RepairToolCall) agentOption {
	return func(s *AgentSettings) {
		s.repairToolCall = fn
	}
}

func WithOnStepFinished(fn OnStepFinishedFunction) agentOption {
	return func(s *AgentSettings) {
		s.onStepFinished = fn
	}
}

func WithHeaders(headers map[string]string) agentOption {
	return func(s *AgentSettings) {
		s.headers = headers
	}
}

func WithProviderOptions(providerOptions ProviderOptions) agentOption {
	return func(s *AgentSettings) {
		s.providerOptions = providerOptions
	}
}
