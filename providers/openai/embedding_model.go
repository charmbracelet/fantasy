package openai

import (
	"context"

	"charm.land/fantasy"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/param"
)

type embeddingModel struct {
	provider string
	modelID  string
	client   openai.Client
}

// Model implements fantasy.EmbeddingModel.
func (e embeddingModel) Model() string {
	return e.modelID
}

// Provider implements fantasy.EmbeddingModel.
func (e embeddingModel) Provider() string {
	return e.provider
}

// Embed implements fantasy.EmbeddingModel.
func (e embeddingModel) Embed(ctx context.Context, call fantasy.EmbeddingCall) (*fantasy.EmbeddingResponse, error) {
	if err := fantasy.ValidateEmbeddingCall(call); err != nil {
		return nil, err
	}

	params := openai.EmbeddingNewParams{
		Model: e.modelID,
	}

	if call.ProviderOptions != nil {
		if v, ok := call.ProviderOptions[Name]; ok {
			providerOptions, ok := v.(*ProviderOptions)
			if !ok {
				return nil, &fantasy.Error{Title: "invalid argument", Message: "openai provider options should be *openai.ProviderOptions"}
			}
			if providerOptions.User != nil {
				params.User = param.NewOpt(*providerOptions.User)
			}
		}
	}

	if call.Dimensions != nil {
		params.Dimensions = param.NewOpt(*call.Dimensions)
	}

	if call.Input != nil {
		params.Input = openai.EmbeddingNewParamsInputUnion{
			OfString: param.NewOpt(*call.Input),
		}
	} else {
		params.Input = openai.EmbeddingNewParamsInputUnion{
			OfArrayOfStrings: call.Inputs,
		}
	}

	response, err := e.client.Embeddings.New(ctx, params)
	if err != nil {
		return nil, toProviderErr(err)
	}

	embeddings := make([]fantasy.Embedding, 0, len(response.Data))
	for _, embedding := range response.Data {
		vector := make([]float32, len(embedding.Embedding))
		for i, value := range embedding.Embedding {
			vector[i] = float32(value)
		}
		embeddings = append(embeddings, fantasy.Embedding{
			Index:  int(embedding.Index),
			Vector: vector,
		})
	}

	usage := fantasy.Usage{
		InputTokens:  response.Usage.PromptTokens,
		TotalTokens:  response.Usage.TotalTokens,
		OutputTokens: 0,
	}

	return &fantasy.EmbeddingResponse{
		Model:      response.Model,
		Usage:      usage,
		Embeddings: embeddings,
	}, nil
}
