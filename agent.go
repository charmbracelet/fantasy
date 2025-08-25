package ai

import (
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
	PrepareStepFunction    = func(options PrepareStepFunctionOptions) (PrepareStepResult, error)
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
	Headers          map[string]string
	ProviderOptions  ProviderOptions
	OnRetry          OnRetryCallback
	MaxRetries       *int

	StopWhen       []StopCondition
	PrepareStep    PrepareStepFunction
	RepairToolCall RepairToolCallFunction
}

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
	OnAgentStart  func()                      // Called when agent starts
	OnAgentFinish func(result *AgentResult)   // Called when agent finishes
	OnStepStart   func(stepNumber int)        // Called when a step starts
	OnStepFinish  func(stepResult StepResult) // Called when a step finishes
	OnFinish      func(result *AgentResult)   // Called when entire agent completes
	OnError       func(error)                 // Called when an error occurs

	// Stream part callbacks - called for each corresponding stream part type
	OnChunk          func(StreamPart)                                                                // Called for each stream part (catch-all)
	OnWarnings       func(warnings []CallWarning)                                                    // Called for warnings
	OnTextStart      func(id string)                                                                 // Called when text starts
	OnTextDelta      func(id, text string)                                                           // Called for text deltas
	OnTextEnd        func(id string)                                                                 // Called when text ends
	OnReasoningStart func(id string)                                                                 // Called when reasoning starts
	OnReasoningDelta func(id, text string)                                                           // Called for reasoning deltas
	OnReasoningEnd   func(id string, reasoning ReasoningContent)                                     // Called when reasoning ends
	OnToolInputStart func(id, toolName string)                                                       // Called when tool input starts
	OnToolInputDelta func(id, delta string)                                                          // Called for tool input deltas
	OnToolInputEnd   func(id string)                                                                 // Called when tool input ends
	OnToolCall       func(toolCall ToolCallContent)                                                  // Called when tool call is complete
	OnToolResult     func(result ToolResultContent)                                                  // Called when tool execution completes
	OnSource         func(source SourceContent)                                                      // Called for source references
	OnStreamFinish   func(usage Usage, finishReason FinishReason, providerMetadata ProviderMetadata) // Called when stream finishes
	OnStreamError    func(error)                                                                     // Called when stream error occurs
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
		stepSystemPrompt := a.settings.systemPrompt
		stepActiveTools := opts.ActiveTools
		stepToolChoice := ToolChoiceAuto
		disableAllTools := false

		if opts.PrepareStep != nil {
			prepared, err := opts.PrepareStep(PrepareStepFunctionOptions{
				Model:      stepModel,
				Steps:      steps,
				StepNumber: len(steps),
				Messages:   stepInputMessages,
			})
			if err != nil {
				return nil, err
			}

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

func (a *agent) executeTools(ctx context.Context, allTools []AgentTool, toolCalls []ToolCallContent, toolResultCallback func(result ToolResultContent)) ([]ToolResultContent, error) {
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
	var toolExecutionError error
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
					toolResultCallback(results[index])
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
					toolResultCallback(results[index])
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
					toolResultCallback(results[index])
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
					ClientMetadata:   result.Metadata,
					ProviderExecuted: false,
				}

				if toolResultCallback != nil {
					toolResultCallback(results[index])
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
					toolResultCallback(results[index])
				}
			}
		}(i, toolCall)
	}

	// Wait for all tool executions to complete
	wg.Wait()

	return results, toolExecutionError
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
		Headers:          opts.Headers,
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
			prepared, err := call.PrepareStep(PrepareStepFunctionOptions{
				Model:      stepModel,
				Steps:      steps,
				StepNumber: stepNumber,
				Messages:   stepInputMessages,
			})
			if err != nil {
				return nil, err
			}

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
			opts.OnStepStart(stepNumber)
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
			Headers:          call.Headers,
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
			opts.OnStepFinish(stepResult)
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
		opts.OnAgentFinish(agentResult)
	}

	return agentResult, nil
}

func (a *agent) prepareTools(tools []AgentTool, activeTools []string, disableAllTools bool) []Tool {
	var preparedTools []Tool

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
		})
	}
	return preparedTools
}

// validateAndRepairToolCall validates a tool call and attempts repair if validation fails
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

// validateToolCall validates a tool call against available tools and their schemas
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

	preparedPrompt = append(preparedPrompt, NewUserMessage(prompt, files...))
	preparedPrompt = append(preparedPrompt, messages...)
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

// processStepStream processes a single step's stream and returns the step result
func (a *agent) processStepStream(ctx context.Context, stream StreamResponse, opts AgentStreamCall, _ []StepResult) (StepResult, bool, error) {
	var stepContent []Content
	var stepToolCalls []ToolCallContent
	var stepUsage Usage
	stepFinishReason := FinishReasonUnknown
	var stepWarnings []CallWarning
	var stepProviderMetadata ProviderMetadata

	activeToolCalls := make(map[string]*ToolCallContent)
	activeTextContent := make(map[string]string)

	// Process stream parts
	for part := range stream {
		// Forward all parts to chunk callback
		if opts.OnChunk != nil {
			opts.OnChunk(part)
		}

		switch part.Type {
		case StreamPartTypeWarnings:
			stepWarnings = part.Warnings
			if opts.OnWarnings != nil {
				opts.OnWarnings(part.Warnings)
			}

		case StreamPartTypeTextStart:
			activeTextContent[part.ID] = ""
			if opts.OnTextStart != nil {
				opts.OnTextStart(part.ID)
			}

		case StreamPartTypeTextDelta:
			if _, exists := activeTextContent[part.ID]; exists {
				activeTextContent[part.ID] += part.Delta
			}
			if opts.OnTextDelta != nil {
				opts.OnTextDelta(part.ID, part.Delta)
			}

		case StreamPartTypeTextEnd:
			if text, exists := activeTextContent[part.ID]; exists {
				stepContent = append(stepContent, TextContent{
					Text:             text,
					ProviderMetadata: ProviderMetadata(part.ProviderMetadata),
				})
				delete(activeTextContent, part.ID)
			}
			if opts.OnTextEnd != nil {
				opts.OnTextEnd(part.ID)
			}

		case StreamPartTypeReasoningStart:
			activeTextContent[part.ID] = ""
			if opts.OnReasoningStart != nil {
				opts.OnReasoningStart(part.ID)
			}

		case StreamPartTypeReasoningDelta:
			if _, exists := activeTextContent[part.ID]; exists {
				activeTextContent[part.ID] += part.Delta
			}
			if opts.OnReasoningDelta != nil {
				opts.OnReasoningDelta(part.ID, part.Delta)
			}

		case StreamPartTypeReasoningEnd:
			if text, exists := activeTextContent[part.ID]; exists {
				stepContent = append(stepContent, ReasoningContent{
					Text:             text,
					ProviderMetadata: ProviderMetadata(part.ProviderMetadata),
				})
				if opts.OnReasoningEnd != nil {
					opts.OnReasoningEnd(part.ID, ReasoningContent{
						Text:             text,
						ProviderMetadata: ProviderMetadata(part.ProviderMetadata),
					})
				}
				delete(activeTextContent, part.ID)
			}

		case StreamPartTypeToolInputStart:
			activeToolCalls[part.ID] = &ToolCallContent{
				ToolCallID:       part.ID,
				ToolName:         part.ToolCallName,
				Input:            "",
				ProviderExecuted: part.ProviderExecuted,
			}
			if opts.OnToolInputStart != nil {
				opts.OnToolInputStart(part.ID, part.ToolCallName)
			}

		case StreamPartTypeToolInputDelta:
			if toolCall, exists := activeToolCalls[part.ID]; exists {
				toolCall.Input += part.Delta
			}
			if opts.OnToolInputDelta != nil {
				opts.OnToolInputDelta(part.ID, part.Delta)
			}

		case StreamPartTypeToolInputEnd:
			if opts.OnToolInputEnd != nil {
				opts.OnToolInputEnd(part.ID)
			}

		case StreamPartTypeToolCall:
			toolCall := ToolCallContent{
				ToolCallID:       part.ID,
				ToolName:         part.ToolCallName,
				Input:            part.ToolCallInput,
				ProviderExecuted: part.ProviderExecuted,
				ProviderMetadata: ProviderMetadata(part.ProviderMetadata),
			}

			// Validate and potentially repair the tool call
			validatedToolCall := a.validateAndRepairToolCall(ctx, toolCall, a.settings.tools, a.settings.systemPrompt, nil, opts.RepairToolCall)
			stepToolCalls = append(stepToolCalls, validatedToolCall)
			stepContent = append(stepContent, validatedToolCall)

			if opts.OnToolCall != nil {
				opts.OnToolCall(validatedToolCall)
			}

			// Clean up active tool call
			delete(activeToolCalls, part.ID)

		case StreamPartTypeSource:
			sourceContent := SourceContent{
				SourceType:       part.SourceType,
				ID:               part.ID,
				URL:              part.URL,
				Title:            part.Title,
				ProviderMetadata: ProviderMetadata(part.ProviderMetadata),
			}
			stepContent = append(stepContent, sourceContent)
			if opts.OnSource != nil {
				opts.OnSource(sourceContent)
			}

		case StreamPartTypeFinish:
			stepUsage = part.Usage
			stepFinishReason = part.FinishReason
			stepProviderMetadata = ProviderMetadata(part.ProviderMetadata)
			if opts.OnStreamFinish != nil {
				opts.OnStreamFinish(part.Usage, part.FinishReason, part.ProviderMetadata)
			}

		case StreamPartTypeError:
			if opts.OnStreamError != nil {
				opts.OnStreamError(part.Error)
			}
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
