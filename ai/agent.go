package ai

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
)

type StepResult struct {
	Response
	Messages []Message
}

type StopCondition = func(steps []StepResult) bool

// StepCountIs returns a stop condition that stops after the specified number of steps.
func StepCountIs(stepCount int) StopCondition {
	return func(steps []StepResult) bool {
		return len(steps) >= stepCount
	}
}

// HasToolCall returns a stop condition that stops when the specified tool is called in the last step.
func HasToolCall(toolName string) StopCondition {
	return func(steps []StepResult) bool {
		if len(steps) == 0 {
			return false
		}
		lastStep := steps[len(steps)-1]
		toolCalls := lastStep.Content.ToolCalls()
		for _, toolCall := range toolCalls {
			if toolCall.ToolName == toolName {
				return true
			}
		}
		return false
	}
}

// HasContent returns a stop condition that stops when the specified content type appears in the last step.
func HasContent(contentType ContentType) StopCondition {
	return func(steps []StepResult) bool {
		if len(steps) == 0 {
			return false
		}
		lastStep := steps[len(steps)-1]
		for _, content := range lastStep.Content {
			if content.GetType() == contentType {
				return true
			}
		}
		return false
	}
}

// FinishReasonIs returns a stop condition that stops when the specified finish reason occurs.
func FinishReasonIs(reason FinishReason) StopCondition {
	return func(steps []StepResult) bool {
		if len(steps) == 0 {
			return false
		}
		lastStep := steps[len(steps)-1]
		return lastStep.FinishReason == reason
	}
}

// MaxTokensUsed returns a stop condition that stops when total token usage exceeds the specified limit.
func MaxTokensUsed(maxTokens int64) StopCondition {
	return func(steps []StepResult) bool {
		var totalTokens int64
		for _, step := range steps {
			totalTokens += step.Usage.TotalTokens
		}
		return totalTokens >= maxTokens
	}
}

type PrepareStepFunctionOptions struct {
	Steps      []StepResult
	StepNumber int
	Model      LanguageModel
	Messages   []Message
}

type PrepareStepResult struct {
	Model           LanguageModel
	Messages        []Message
	System          *string
	ToolChoice      *ToolChoice
	ActiveTools     []string
	DisableAllTools bool
}

type ToolCallRepairOptions struct {
	OriginalToolCall ToolCallContent
	ValidationError  error
	AvailableTools   []AgentTool
	SystemPrompt     string
	Messages         []Message
}

type (
	PrepareStepFunction    = func(ctx context.Context, options PrepareStepFunctionOptions) (context.Context, PrepareStepResult, error)
	OnStepFinishedFunction = func(step StepResult)
	RepairToolCallFunction = func(ctx context.Context, options ToolCallRepairOptions) (*ToolCallContent, error)
)

type agentSettings struct {
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
	tools      []AgentTool
	maxRetries *int

	model LanguageModel

	stopWhen       []StopCondition
	prepareStep    PrepareStepFunction
	repairToolCall RepairToolCallFunction
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
	ProviderOptions  ProviderOptions
	OnRetry          OnRetryCallback
	MaxRetries       *int

	StopWhen       []StopCondition
	PrepareStep    PrepareStepFunction
	RepairToolCall RepairToolCallFunction
}

// Agent-level callbacks.
type (
	// OnAgentStartFunc is called when agent starts.
	OnAgentStartFunc func()

	// OnAgentFinishFunc is called when agent finishes.
	OnAgentFinishFunc func(result *AgentResult) error

	// OnStepStartFunc is called when a step starts.
	OnStepStartFunc func(stepNumber int) error

	// OnStepFinishFunc is called when a step finishes.
	OnStepFinishFunc func(stepResult StepResult) error

	// OnFinishFunc is called when entire agent completes.
	OnFinishFunc func(result *AgentResult)

	// OnErrorFunc is called when an error occurs.
	OnErrorFunc func(error)
)

// Stream part callbacks - called for each corresponding stream part type.
type (
	// OnChunkFunc is called for each stream part (catch-all).
	OnChunkFunc func(StreamPart) error

	// OnWarningsFunc is called for warnings.
	OnWarningsFunc func(warnings []CallWarning) error

	// OnTextStartFunc is called when text starts.
	OnTextStartFunc func(id string) error

	// OnTextDeltaFunc is called for text deltas.
	OnTextDeltaFunc func(id, text string) error

	// OnTextEndFunc is called when text ends.
	OnTextEndFunc func(id string) error

	// OnReasoningStartFunc is called when reasoning starts.
	OnReasoningStartFunc func(id string) error

	// OnReasoningDeltaFunc is called for reasoning deltas.
	OnReasoningDeltaFunc func(id, text string) error

	// OnReasoningEndFunc is called when reasoning ends.
	OnReasoningEndFunc func(id string, reasoning ReasoningContent) error

	// OnToolInputStartFunc is called when tool input starts.
	OnToolInputStartFunc func(id, toolName string) error

	// OnToolInputDeltaFunc is called for tool input deltas.
	OnToolInputDeltaFunc func(id, delta string) error

	// OnToolInputEndFunc is called when tool input ends.
	OnToolInputEndFunc func(id string) error

	// OnToolCallFunc is called when tool call is complete.
	OnToolCallFunc func(toolCall ToolCallContent) error

	// OnToolResultFunc is called when tool execution completes.
	OnToolResultFunc func(result ToolResultContent) error

	// OnSourceFunc is called for source references.
	OnSourceFunc func(source SourceContent) error

	// OnStreamFinishFunc is called when stream finishes.
	OnStreamFinishFunc func(usage Usage, finishReason FinishReason, providerMetadata ProviderMetadata) error
)

type AgentStreamCall struct {
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
	RepairToolCall RepairToolCallFunction

	// Agent-level callbacks
	OnAgentStart  OnAgentStartFunc  // Called when agent starts
	OnAgentFinish OnAgentFinishFunc // Called when agent finishes
	OnStepStart   OnStepStartFunc   // Called when a step starts
	OnStepFinish  OnStepFinishFunc  // Called when a step finishes
	OnFinish      OnFinishFunc      // Called when entire agent completes
	OnError       OnErrorFunc       // Called when an error occurs

	// Stream part callbacks - called for each corresponding stream part type
	OnChunk          OnChunkFunc          // Called for each stream part (catch-all)
	OnWarnings       OnWarningsFunc       // Called for warnings
	OnTextStart      OnTextStartFunc      // Called when text starts
	OnTextDelta      OnTextDeltaFunc      // Called for text deltas
	OnTextEnd        OnTextEndFunc        // Called when text ends
	OnReasoningStart OnReasoningStartFunc // Called when reasoning starts
	OnReasoningDelta OnReasoningDeltaFunc // Called for reasoning deltas
	OnReasoningEnd   OnReasoningEndFunc   // Called when reasoning ends
	OnToolInputStart OnToolInputStartFunc // Called when tool input starts
	OnToolInputDelta OnToolInputDeltaFunc // Called for tool input deltas
	OnToolInputEnd   OnToolInputEndFunc   // Called when tool input ends
	OnToolCall       OnToolCallFunc       // Called when tool call is complete
	OnToolResult     OnToolResultFunc     // Called when tool execution completes
	OnSource         OnSourceFunc         // Called for source references
	OnStreamFinish   OnStreamFinishFunc   // Called when stream finishes
}

type AgentResult struct {
	Steps []StepResult
	// Final response
	Response   Response
	TotalUsage Usage
}

type Agent interface {
	Generate(context.Context, AgentCall) (*AgentResult, error)
	Stream(context.Context, AgentStreamCall) (*AgentResult, error)
}

type AgentOption = func(*agentSettings)

type agent struct {
	settings agentSettings
}

func NewAgent(model LanguageModel, opts ...AgentOption) Agent {
	settings := agentSettings{
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
	call.MaxOutputTokens = cmp.Or(call.MaxOutputTokens, a.settings.maxOutputTokens)
	call.Temperature = cmp.Or(call.Temperature, a.settings.temperature)
	call.TopP = cmp.Or(call.TopP, a.settings.topP)
	call.TopK = cmp.Or(call.TopK, a.settings.topK)
	call.PresencePenalty = cmp.Or(call.PresencePenalty, a.settings.presencePenalty)
	call.FrequencyPenalty = cmp.Or(call.FrequencyPenalty, a.settings.frequencyPenalty)
	call.MaxRetries = cmp.Or(call.MaxRetries, a.settings.maxRetries)

	if len(call.StopWhen) == 0 && len(a.settings.stopWhen) > 0 {
		call.StopWhen = a.settings.stopWhen
	}
	if call.PrepareStep == nil && a.settings.prepareStep != nil {
		call.PrepareStep = a.settings.prepareStep
	}
	if call.RepairToolCall == nil && a.settings.repairToolCall != nil {
		call.RepairToolCall = a.settings.repairToolCall
	}
	if call.OnRetry == nil && a.settings.onRetry != nil {
		call.OnRetry = a.settings.onRetry
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
		stepSystemPrompt := a.settings.systemPrompt
		stepActiveTools := opts.ActiveTools
		stepToolChoice := ToolChoiceAuto
		disableAllTools := false

		if opts.PrepareStep != nil {
			updatedCtx, prepared, err := opts.PrepareStep(ctx, PrepareStepFunctionOptions{
				Model:      stepModel,
				Steps:      steps,
				StepNumber: len(steps),
				Messages:   stepInputMessages,
			})
			if err != nil {
				return nil, err
			}

			ctx = updatedCtx

			// Apply prepared step modifications
			if prepared.Messages != nil {
				stepInputMessages = prepared.Messages
			}
			if prepared.Model != nil {
				stepModel = prepared.Model
			}
			if prepared.System != nil {
				stepSystemPrompt = *prepared.System
			}
			if prepared.ToolChoice != nil {
				stepToolChoice = *prepared.ToolChoice
			}
			if len(prepared.ActiveTools) > 0 {
				stepActiveTools = prepared.ActiveTools
			}
			disableAllTools = prepared.DisableAllTools
		}

		// Recreate prompt with potentially modified system prompt
		if stepSystemPrompt != a.settings.systemPrompt {
			stepPrompt, err := a.createPrompt(stepSystemPrompt, opts.Prompt, opts.Messages, opts.Files...)
			if err != nil {
				return nil, err
			}
			// Replace system message part, keep the rest
			if len(stepInputMessages) > 0 && len(stepPrompt) > 0 {
				stepInputMessages[0] = stepPrompt[0] // Replace system message
			}
		}

		preparedTools := a.prepareTools(a.settings.tools, stepActiveTools, disableAllTools)

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
				ToolChoice:       &stepToolChoice,
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

				// Validate and potentially repair the tool call
				validatedToolCall := a.validateAndRepairToolCall(ctx, toolCall, a.settings.tools, stepSystemPrompt, stepInputMessages, a.settings.repairToolCall)
				stepToolCalls = append(stepToolCalls, validatedToolCall)
			}
		}

		toolResults, err := a.executeTools(ctx, a.settings.tools, stepToolCalls, nil)

		// Build step content with validated tool calls and tool results
		stepContent := []Content{}
		toolCallIndex := 0
		for _, content := range result.Content {
			if content.GetType() == ContentTypeToolCall {
				// Replace with validated tool call
				if toolCallIndex < len(stepToolCalls) {
					stepContent = append(stepContent, stepToolCalls[toolCallIndex])
					toolCallIndex++
				}
			} else {
				// Keep other content as-is
				stepContent = append(stepContent, content)
			}
		}
		// Add tool results
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
		case ContentTypeSource:
			// Sources are metadata about references used to generate the response.
			// They don't need to be included in the conversation messages.
			continue
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

func (a *agent) executeTools(ctx context.Context, allTools []AgentTool, toolCalls []ToolCallContent, toolResultCallback func(result ToolResultContent) error) ([]ToolResultContent, error) {
	if len(toolCalls) == 0 {
		return nil, nil
	}

	// Create a map for quick tool lookup
	toolMap := make(map[string]AgentTool)
	for _, tool := range allTools {
		toolMap[tool.Info().Name] = tool
	}

	// Execute all tool calls in parallel
	results := make([]ToolResultContent, len(toolCalls))
	executeErrors := make([]error, len(toolCalls))

	var wg sync.WaitGroup

	for i, toolCall := range toolCalls {
		wg.Add(1)
		go func(index int, call ToolCallContent) {
			defer wg.Done()

			// Skip invalid tool calls - create error result
			if call.Invalid {
				results[index] = ToolResultContent{
					ToolCallID: call.ToolCallID,
					ToolName:   call.ToolName,
					Result: ToolResultOutputContentError{
						Error: call.ValidationError,
					},
					ProviderExecuted: false,
				}
				if toolResultCallback != nil {
					err := toolResultCallback(results[index])
					if err != nil {
						executeErrors[index] = err
					}
				}

				return
			}

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

				if toolResultCallback != nil {
					err := toolResultCallback(results[index])
					if err != nil {
						executeErrors[index] = err
					}
				}
				return
			}

			// Execute the tool
			result, err := tool.Run(ctx, ToolCall{
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
					ClientMetadata:   result.Metadata,
					ProviderExecuted: false,
				}
				if toolResultCallback != nil {
					cbErr := toolResultCallback(results[index])
					if cbErr != nil {
						executeErrors[index] = cbErr
					}
				}
				executeErrors[index] = err
				return
			}

			if result.IsError {
				results[index] = ToolResultContent{
					ToolCallID: call.ToolCallID,
					ToolName:   call.ToolName,
					Result: ToolResultOutputContentError{
						Error: errors.New(result.Content),
					},
					ClientMetadata:   result.Metadata,
					ProviderExecuted: false,
				}

				if toolResultCallback != nil {
					err := toolResultCallback(results[index])
					if err != nil {
						executeErrors[index] = err
					}
				}
			} else {
				results[index] = ToolResultContent{
					ToolCallID: call.ToolCallID,
					ToolName:   toolCall.ToolName,
					Result: ToolResultOutputContentText{
						Text: result.Content,
					},
					ClientMetadata:   result.Metadata,
					ProviderExecuted: false,
				}
				if toolResultCallback != nil {
					err := toolResultCallback(results[index])
					if err != nil {
						executeErrors[index] = err
					}
				}
			}
		}(i, toolCall)
	}

	// Wait for all tool executions to complete
	wg.Wait()

	for _, err := range executeErrors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// Stream implements Agent.
func (a *agent) Stream(ctx context.Context, opts AgentStreamCall) (*AgentResult, error) {
	// Convert AgentStreamCall to AgentCall for preparation
	call := AgentCall{
		Prompt:           opts.Prompt,
		Files:            opts.Files,
		Messages:         opts.Messages,
		MaxOutputTokens:  opts.MaxOutputTokens,
		Temperature:      opts.Temperature,
		TopP:             opts.TopP,
		TopK:             opts.TopK,
		PresencePenalty:  opts.PresencePenalty,
		FrequencyPenalty: opts.FrequencyPenalty,
		ActiveTools:      opts.ActiveTools,
		ProviderOptions:  opts.ProviderOptions,
		MaxRetries:       opts.MaxRetries,
		StopWhen:         opts.StopWhen,
		PrepareStep:      opts.PrepareStep,
		RepairToolCall:   opts.RepairToolCall,
	}

	call = a.prepareCall(call)

	initialPrompt, err := a.createPrompt(a.settings.systemPrompt, call.Prompt, call.Messages, call.Files...)
	if err != nil {
		return nil, err
	}

	var responseMessages []Message
	var steps []StepResult
	var totalUsage Usage

	// Start agent stream
	if opts.OnAgentStart != nil {
		opts.OnAgentStart()
	}

	for stepNumber := 0; ; stepNumber++ {
		stepInputMessages := append(initialPrompt, responseMessages...)
		stepModel := a.settings.model
		stepSystemPrompt := a.settings.systemPrompt
		stepActiveTools := call.ActiveTools
		stepToolChoice := ToolChoiceAuto
		disableAllTools := false

		// Apply step preparation if provided
		if call.PrepareStep != nil {
			updatedCtx, prepared, err := call.PrepareStep(ctx, PrepareStepFunctionOptions{
				Model:      stepModel,
				Steps:      steps,
				StepNumber: stepNumber,
				Messages:   stepInputMessages,
			})
			if err != nil {
				return nil, err
			}

			ctx = updatedCtx

			if prepared.Messages != nil {
				stepInputMessages = prepared.Messages
			}
			if prepared.Model != nil {
				stepModel = prepared.Model
			}
			if prepared.System != nil {
				stepSystemPrompt = *prepared.System
			}
			if prepared.ToolChoice != nil {
				stepToolChoice = *prepared.ToolChoice
			}
			if len(prepared.ActiveTools) > 0 {
				stepActiveTools = prepared.ActiveTools
			}
			disableAllTools = prepared.DisableAllTools
		}

		// Recreate prompt with potentially modified system prompt
		if stepSystemPrompt != a.settings.systemPrompt {
			stepPrompt, err := a.createPrompt(stepSystemPrompt, call.Prompt, call.Messages, call.Files...)
			if err != nil {
				return nil, err
			}
			if len(stepInputMessages) > 0 && len(stepPrompt) > 0 {
				stepInputMessages[0] = stepPrompt[0]
			}
		}

		preparedTools := a.prepareTools(a.settings.tools, stepActiveTools, disableAllTools)

		// Start step stream
		if opts.OnStepStart != nil {
			_ = opts.OnStepStart(stepNumber)
		}

		// Create streaming call
		streamCall := Call{
			Prompt:           stepInputMessages,
			MaxOutputTokens:  call.MaxOutputTokens,
			Temperature:      call.Temperature,
			TopP:             call.TopP,
			TopK:             call.TopK,
			PresencePenalty:  call.PresencePenalty,
			FrequencyPenalty: call.FrequencyPenalty,
			Tools:            preparedTools,
			ToolChoice:       &stepToolChoice,
			ProviderOptions:  call.ProviderOptions,
		}

		// Get streaming response
		stream, err := stepModel.Stream(ctx, streamCall)
		if err != nil {
			if opts.OnError != nil {
				opts.OnError(err)
			}
			return nil, err
		}

		// Process stream with tool execution
		stepResult, shouldContinue, err := a.processStepStream(ctx, stream, opts, steps)
		if err != nil {
			if opts.OnError != nil {
				opts.OnError(err)
			}
			return nil, err
		}

		steps = append(steps, stepResult)
		totalUsage = addUsage(totalUsage, stepResult.Usage)

		// Call step finished callback
		if opts.OnStepFinish != nil {
			_ = opts.OnStepFinish(stepResult)
		}

		// Add step messages to response messages
		stepMessages := toResponseMessages(stepResult.Content)
		responseMessages = append(responseMessages, stepMessages...)

		// Check stop conditions
		shouldStop := isStopConditionMet(call.StopWhen, steps)
		if shouldStop || !shouldContinue {
			break
		}
	}

	// Finish agent stream
	agentResult := &AgentResult{
		Steps:      steps,
		Response:   steps[len(steps)-1].Response,
		TotalUsage: totalUsage,
	}

	if opts.OnFinish != nil {
		opts.OnFinish(agentResult)
	}

	if opts.OnAgentFinish != nil {
		_ = opts.OnAgentFinish(agentResult)
	}

	return agentResult, nil
}

func (a *agent) prepareTools(tools []AgentTool, activeTools []string, disableAllTools bool) []Tool {
	preparedTools := make([]Tool, 0, len(tools))

	// If explicitly disabling all tools, return no tools
	if disableAllTools {
		return preparedTools
	}

	for _, tool := range tools {
		// If activeTools has items, only include tools in the list
		// If activeTools is empty, include all tools
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
			ProviderOptions: tool.ProviderOptions(),
		})
	}
	return preparedTools
}

// validateAndRepairToolCall validates a tool call and attempts repair if validation fails.
func (a *agent) validateAndRepairToolCall(ctx context.Context, toolCall ToolCallContent, availableTools []AgentTool, systemPrompt string, messages []Message, repairFunc RepairToolCallFunction) ToolCallContent {
	if err := a.validateToolCall(toolCall, availableTools); err == nil {
		return toolCall
	} else {
		if repairFunc != nil {
			repairOptions := ToolCallRepairOptions{
				OriginalToolCall: toolCall,
				ValidationError:  err,
				AvailableTools:   availableTools,
				SystemPrompt:     systemPrompt,
				Messages:         messages,
			}

			if repairedToolCall, repairErr := repairFunc(ctx, repairOptions); repairErr == nil && repairedToolCall != nil {
				if validateErr := a.validateToolCall(*repairedToolCall, availableTools); validateErr == nil {
					return *repairedToolCall
				}
			}
		}

		invalidToolCall := toolCall
		invalidToolCall.Invalid = true
		invalidToolCall.ValidationError = err
		return invalidToolCall
	}
}

// validateToolCall validates a tool call against available tools and their schemas.
func (a *agent) validateToolCall(toolCall ToolCallContent, availableTools []AgentTool) error {
	var tool AgentTool
	for _, t := range availableTools {
		if t.Info().Name == toolCall.ToolName {
			tool = t
			break
		}
	}

	if tool == nil {
		return fmt.Errorf("tool not found: %s", toolCall.ToolName)
	}

	// Validate JSON parsing
	var input map[string]any
	if err := json.Unmarshal([]byte(toolCall.Input), &input); err != nil {
		return fmt.Errorf("invalid JSON input: %w", err)
	}

	// Basic schema validation (check required fields)
	// TODO: more robust schema validation using JSON Schema or similar
	toolInfo := tool.Info()
	for _, required := range toolInfo.Required {
		if _, exists := input[required]; !exists {
			return fmt.Errorf("missing required parameter: %s", required)
		}
	}
	return nil
}

func (a *agent) createPrompt(system, prompt string, messages []Message, files ...FilePart) (Prompt, error) {
	if prompt == "" {
		return nil, NewInvalidPromptError(prompt, "Prompt can't be empty", nil)
	}

	var preparedPrompt Prompt

	if system != "" {
		preparedPrompt = append(preparedPrompt, NewSystemMessage(system))
	}
	preparedPrompt = append(preparedPrompt, messages...)
	preparedPrompt = append(preparedPrompt, NewUserMessage(prompt, files...))
	return preparedPrompt, nil
}

func WithSystemPrompt(prompt string) AgentOption {
	return func(s *agentSettings) {
		s.systemPrompt = prompt
	}
}

func WithMaxOutputTokens(tokens int64) AgentOption {
	return func(s *agentSettings) {
		s.maxOutputTokens = &tokens
	}
}

func WithTemperature(temp float64) AgentOption {
	return func(s *agentSettings) {
		s.temperature = &temp
	}
}

func WithTopP(topP float64) AgentOption {
	return func(s *agentSettings) {
		s.topP = &topP
	}
}

func WithTopK(topK int64) AgentOption {
	return func(s *agentSettings) {
		s.topK = &topK
	}
}

func WithPresencePenalty(penalty float64) AgentOption {
	return func(s *agentSettings) {
		s.presencePenalty = &penalty
	}
}

func WithFrequencyPenalty(penalty float64) AgentOption {
	return func(s *agentSettings) {
		s.frequencyPenalty = &penalty
	}
}

func WithTools(tools ...AgentTool) AgentOption {
	return func(s *agentSettings) {
		s.tools = append(s.tools, tools...)
	}
}

func WithStopConditions(conditions ...StopCondition) AgentOption {
	return func(s *agentSettings) {
		s.stopWhen = append(s.stopWhen, conditions...)
	}
}

func WithPrepareStep(fn PrepareStepFunction) AgentOption {
	return func(s *agentSettings) {
		s.prepareStep = fn
	}
}

func WithRepairToolCall(fn RepairToolCallFunction) AgentOption {
	return func(s *agentSettings) {
		s.repairToolCall = fn
	}
}

// processStepStream processes a single step's stream and returns the step result.
func (a *agent) processStepStream(ctx context.Context, stream StreamResponse, opts AgentStreamCall, _ []StepResult) (StepResult, bool, error) {
	var stepContent []Content
	var stepToolCalls []ToolCallContent
	var stepUsage Usage
	stepFinishReason := FinishReasonUnknown
	var stepWarnings []CallWarning
	var stepProviderMetadata ProviderMetadata

	activeToolCalls := make(map[string]*ToolCallContent)
	activeTextContent := make(map[string]string)
	type reasoningContent struct {
		content string
		options ProviderMetadata
	}
	activeReasoningContent := make(map[string]reasoningContent)

	// Process stream parts
	for part := range stream {
		// Forward all parts to chunk callback
		if opts.OnChunk != nil {
			err := opts.OnChunk(part)
			if err != nil {
				return StepResult{}, false, err
			}
		}

		switch part.Type {
		case StreamPartTypeWarnings:
			stepWarnings = part.Warnings
			if opts.OnWarnings != nil {
				err := opts.OnWarnings(part.Warnings)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeTextStart:
			activeTextContent[part.ID] = ""
			if opts.OnTextStart != nil {
				err := opts.OnTextStart(part.ID)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeTextDelta:
			if _, exists := activeTextContent[part.ID]; exists {
				activeTextContent[part.ID] += part.Delta
			}
			if opts.OnTextDelta != nil {
				err := opts.OnTextDelta(part.ID, part.Delta)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeTextEnd:
			if text, exists := activeTextContent[part.ID]; exists {
				stepContent = append(stepContent, TextContent{
					Text:             text,
					ProviderMetadata: part.ProviderMetadata,
				})
				delete(activeTextContent, part.ID)
			}
			if opts.OnTextEnd != nil {
				err := opts.OnTextEnd(part.ID)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeReasoningStart:
			activeReasoningContent[part.ID] = reasoningContent{content: ""}
			if opts.OnReasoningStart != nil {
				err := opts.OnReasoningStart(part.ID)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeReasoningDelta:
			if active, exists := activeReasoningContent[part.ID]; exists {
				active.content += part.Delta
				active.options = part.ProviderMetadata
				activeReasoningContent[part.ID] = active
			}
			if opts.OnReasoningDelta != nil {
				err := opts.OnReasoningDelta(part.ID, part.Delta)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeReasoningEnd:
			if active, exists := activeReasoningContent[part.ID]; exists {
				content := ReasoningContent{
					Text:             active.content,
					ProviderMetadata: active.options,
				}
				stepContent = append(stepContent, content)
				if opts.OnReasoningEnd != nil {
					err := opts.OnReasoningEnd(part.ID, content)
					if err != nil {
						return StepResult{}, false, err
					}
				}
				delete(activeReasoningContent, part.ID)
			}

		case StreamPartTypeToolInputStart:
			activeToolCalls[part.ID] = &ToolCallContent{
				ToolCallID:       part.ID,
				ToolName:         part.ToolCallName,
				Input:            "",
				ProviderExecuted: part.ProviderExecuted,
			}
			if opts.OnToolInputStart != nil {
				err := opts.OnToolInputStart(part.ID, part.ToolCallName)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeToolInputDelta:
			if toolCall, exists := activeToolCalls[part.ID]; exists {
				toolCall.Input += part.Delta
			}
			if opts.OnToolInputDelta != nil {
				err := opts.OnToolInputDelta(part.ID, part.Delta)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeToolInputEnd:
			if opts.OnToolInputEnd != nil {
				err := opts.OnToolInputEnd(part.ID)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeToolCall:
			toolCall := ToolCallContent{
				ToolCallID:       part.ID,
				ToolName:         part.ToolCallName,
				Input:            part.ToolCallInput,
				ProviderExecuted: part.ProviderExecuted,
				ProviderMetadata: part.ProviderMetadata,
			}

			// Validate and potentially repair the tool call
			validatedToolCall := a.validateAndRepairToolCall(ctx, toolCall, a.settings.tools, a.settings.systemPrompt, nil, opts.RepairToolCall)
			stepToolCalls = append(stepToolCalls, validatedToolCall)
			stepContent = append(stepContent, validatedToolCall)

			if opts.OnToolCall != nil {
				err := opts.OnToolCall(validatedToolCall)
				if err != nil {
					return StepResult{}, false, err
				}
			}

			// Clean up active tool call
			delete(activeToolCalls, part.ID)

		case StreamPartTypeSource:
			sourceContent := SourceContent{
				SourceType:       part.SourceType,
				ID:               part.ID,
				URL:              part.URL,
				Title:            part.Title,
				ProviderMetadata: part.ProviderMetadata,
			}
			stepContent = append(stepContent, sourceContent)
			if opts.OnSource != nil {
				err := opts.OnSource(sourceContent)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeFinish:
			stepUsage = part.Usage
			stepFinishReason = part.FinishReason
			stepProviderMetadata = part.ProviderMetadata
			if opts.OnStreamFinish != nil {
				err := opts.OnStreamFinish(part.Usage, part.FinishReason, part.ProviderMetadata)
				if err != nil {
					return StepResult{}, false, err
				}
			}

		case StreamPartTypeError:
			if opts.OnError != nil {
				opts.OnError(part.Error)
			}
			return StepResult{}, false, part.Error
		}
	}

	// Execute tools if any
	var toolResults []ToolResultContent
	if len(stepToolCalls) > 0 {
		var err error
		toolResults, err = a.executeTools(ctx, a.settings.tools, stepToolCalls, opts.OnToolResult)
		if err != nil {
			return StepResult{}, false, err
		}
		// Add tool results to content
		for _, result := range toolResults {
			stepContent = append(stepContent, result)
		}
	}

	stepResult := StepResult{
		Response: Response{
			Content:          stepContent,
			FinishReason:     stepFinishReason,
			Usage:            stepUsage,
			Warnings:         stepWarnings,
			ProviderMetadata: stepProviderMetadata,
		},
		Messages: toResponseMessages(stepContent),
	}

	// Determine if we should continue (has tool calls and not stopped)
	shouldContinue := len(stepToolCalls) > 0 && stepFinishReason == FinishReasonToolCalls

	return stepResult, shouldContinue, nil
}

func addUsage(a, b Usage) Usage {
	return Usage{
		InputTokens:         a.InputTokens + b.InputTokens,
		OutputTokens:        a.OutputTokens + b.OutputTokens,
		TotalTokens:         a.TotalTokens + b.TotalTokens,
		ReasoningTokens:     a.ReasoningTokens + b.ReasoningTokens,
		CacheCreationTokens: a.CacheCreationTokens + b.CacheCreationTokens,
		CacheReadTokens:     a.CacheReadTokens + b.CacheReadTokens,
	}
}

func WithHeaders(headers map[string]string) AgentOption {
	return func(s *agentSettings) {
		s.headers = headers
	}
}

func WithProviderOptions(providerOptions ProviderOptions) AgentOption {
	return func(s *agentSettings) {
		s.providerOptions = providerOptions
	}
}
