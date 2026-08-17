package kronk

import (
	"context"
	"fmt"
	"time"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
)

// Option defines a function that configures Kronk provider options.
type Option func(*options)

// Logger is the function signature for logging download progress.
type Logger func(ctx context.Context, msg string, args ...any)

type options struct {
	name                 string
	modelOptions         []model.Option
	logger               Logger
	objectMode           fantasy.ObjectMode
	languageModelOptions []LanguageModelOption
}

// WithName sets the name for the Kronk provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithModelConfig replaces the model configuration at this point in the option sequence.
// The provider replaces ModelFiles with the files resolved by LanguageModel.
func WithModelConfig(cfg model.Config) Option {
	return withModelOption(model.WithConfig(cfg))
}

// WithConfig replaces the model configuration at this point in the option sequence.
// The provider replaces ModelFiles with the files resolved by LanguageModel.
func WithConfig(cfg model.Config) Option {
	return withModelOption(model.WithConfig(cfg))
}

// WithAdapters configures model adapters.
func WithAdapters(v []model.AdapterConfig) Option {
	return withModelOption(model.WithAdapters(v))
}

// WithAdmissionTimeout configures the model admission timeout.
func WithAdmissionTimeout(v time.Duration) Option {
	return withModelOption(model.WithAdmissionTimeout(v))
}

// WithAutoTune enables hardware-aware model configuration.
func WithAutoTune(v bool) Option {
	return withModelOption(model.WithAutoTune(v))
}

// WithCacheMinTokens configures the minimum number of tokens required for caching.
func WithCacheMinTokens(v int) Option {
	return withModelOption(model.WithCacheMinTokens(v))
}

// WithCacheTypeK configures the key cache data type.
func WithCacheTypeK(v model.GGMLType) Option {
	return withModelOption(model.WithCacheTypeK(v))
}

// WithCacheTypeV configures the value cache data type.
func WithCacheTypeV(v model.GGMLType) Option {
	return withModelOption(model.WithCacheTypeV(v))
}

// WithContextWindow configures the model context window.
func WithContextWindow(v int) Option {
	return withModelOption(model.WithContextWindow(v))
}

// WithDefaultParams configures the default model parameters.
func WithDefaultParams(v model.Params) Option {
	return withModelOption(model.WithDefaultParams(v))
}

// WithChatTemplateKwargs configures the model chat template arguments.
func WithChatTemplateKwargs(v model.D) Option {
	return withModelOption(model.WithChatTemplateKwargs(v))
}

// WithDevices configures the devices used for model execution.
func WithDevices(v []string) Option {
	return withModelOption(model.WithDevices(v))
}

// WithDraftModel configures speculative decoding with a draft model.
func WithDraftModel(v *model.DraftModelConfig) Option {
	return withModelOption(model.WithDraftModel(v))
}

// WithFlashAttention configures Flash Attention.
func WithFlashAttention(v model.FlashAttentionType) Option {
	return withModelOption(model.WithFlashAttention(v))
}

// WithIMCSessionCapacity configures the incremental message cache session capacity.
func WithIMCSessionCapacity(v int) Option {
	return withModelOption(model.WithIMCSessionCapacity(v))
}

// WithIncrementalCache configures incremental message caching.
func WithIncrementalCache(v bool) Option {
	return withModelOption(model.WithIncrementalCache(v))
}

// WithInsecureLogging configures logging of potentially sensitive model data.
func WithInsecureLogging(v bool) Option {
	return withModelOption(model.WithInsecureLogging(v))
}

// WithJinjaFile configures the model's Jinja template file.
func WithJinjaFile(v string) Option {
	return withModelOption(model.WithJinjaFile(v))
}

// WithLoadMode configures how model weights are loaded.
func WithLoadMode(v model.LoadMode) Option {
	return withModelOption(model.WithLoadMode(v))
}

// WithLog configures the logger used by model operations.
func WithLog(v model.Logger) Option {
	return withModelOption(model.WithLog(v))
}

// WithMainGPU configures the primary GPU.
func WithMainGPU(v int) Option {
	return withModelOption(model.WithMainGPU(v))
}

// WithMoE configures mixture-of-experts execution.
func WithMoE(v *model.MoEConfig) Option {
	return withModelOption(model.WithMoE(v))
}

// WithModelFiles configures the model files at this point in the option sequence.
// The provider replaces them with the files resolved by LanguageModel.
func WithModelFiles(v []string) Option {
	return withModelOption(model.WithModelFiles(v))
}

// WithNGpuLayers configures the number of model layers offloaded to the GPU.
func WithNGpuLayers(v int) Option {
	return withModelOption(model.WithNGpuLayers(v))
}

// WithNSeqMax configures the maximum number of concurrent sequences.
func WithNSeqMax(v int) Option {
	return withModelOption(model.WithNSeqMax(v))
}

// WithNThreads configures the number of generation threads.
func WithNThreads(v int) Option {
	return withModelOption(model.WithNThreads(v))
}

// WithNThreadsBatch configures the number of batch-processing threads.
func WithNThreadsBatch(v int) Option {
	return withModelOption(model.WithNThreadsBatch(v))
}

// WithNUMA configures the NUMA strategy.
func WithNUMA(v string) Option {
	return withModelOption(model.WithNUMA(v))
}

// WithOffloadKQV configures KV cache offloading.
func WithOffloadKQV(v bool) Option {
	return withModelOption(model.WithOffloadKQV(v))
}

// WithOpOffload configures host tensor operation offloading.
func WithOpOffload(v bool) Option {
	return withModelOption(model.WithOpOffload(v))
}

// WithOpOffloadMinBatch configures the minimum batch size for operation offloading.
func WithOpOffloadMinBatch(v int) Option {
	return withModelOption(model.WithOpOffloadMinBatch(v))
}

// WithPrefillBatchSize configures the maximum number of prompt tokens processed per decode iteration.
func WithPrefillBatchSize(v int) Option {
	return withModelOption(model.WithPrefillBatchSize(v))
}

// WithProjFile configures the multimodal projection file.
func WithProjFile(v string) Option {
	return withModelOption(model.WithProjFile(v))
}

// WithMTPDrafterFile configures the MTP drafter file.
func WithMTPDrafterFile(v string) Option {
	return withModelOption(model.WithMTPDrafterFile(v))
}

// WithProjOnCPU configures projection execution on the CPU.
func WithProjOnCPU(v bool) Option {
	return withModelOption(model.WithProjOnCPU(v))
}

// WithRopeFreqBase configures the RoPE base frequency.
func WithRopeFreqBase(v float32) Option {
	return withModelOption(model.WithRopeFreqBase(v))
}

// WithRopeFreqScale configures the RoPE frequency scale.
func WithRopeFreqScale(v float32) Option {
	return withModelOption(model.WithRopeFreqScale(v))
}

// WithRopeScaling configures the RoPE scaling strategy.
func WithRopeScaling(v model.RopeScalingType) Option {
	return withModelOption(model.WithRopeScaling(v))
}

// WithSplitMode configures how the model is split across devices.
func WithSplitMode(v model.SplitMode) Option {
	return withModelOption(model.WithSplitMode(v))
}

// WithSWAFull configures full-size sliding-window-attention caching.
func WithSWAFull(v bool) Option {
	return withModelOption(model.WithSWAFull(v))
}

// WithTensorBuftOverrides configures tensor buffer type overrides.
func WithTensorBuftOverrides(v []string) Option {
	return withModelOption(model.WithTensorBuftOverrides(v))
}

// WithTensorSplit configures model distribution across devices.
func WithTensorSplit(v []float32) Option {
	return withModelOption(model.WithTensorSplit(v))
}

// WithYarnAttnFactor configures the YaRN attention factor.
func WithYarnAttnFactor(v float32) Option {
	return withModelOption(model.WithYarnAttnFactor(v))
}

// WithYarnBetaFast configures the YaRN low correction dimension.
func WithYarnBetaFast(v float32) Option {
	return withModelOption(model.WithYarnBetaFast(v))
}

// WithYarnBetaSlow configures the YaRN high correction dimension.
func WithYarnBetaSlow(v float32) Option {
	return withModelOption(model.WithYarnBetaSlow(v))
}

// WithYarnExtFactor configures the YaRN extrapolation mix factor.
func WithYarnExtFactor(v float32) Option {
	return withModelOption(model.WithYarnExtFactor(v))
}

// WithYarnOrigCtx configures the original YaRN training context size.
func WithYarnOrigCtx(v int) Option {
	return withModelOption(model.WithYarnOrigCtx(v))
}

// WithQueueDepth configures the model admission queue depth.
func WithQueueDepth(v int) Option {
	return withModelOption(model.WithQueueDepth(v))
}

// WithSessionStoreFactory configures the model session store factory.
func WithSessionStoreFactory(v model.SessionStoreFactory) Option {
	return withModelOption(model.WithSessionStoreFactory(v))
}

func withModelOption(opt model.Option) Option {
	return func(o *options) {
		o.modelOptions = append(o.modelOptions, opt)
	}
}

// WithLogger sets the logger function for download progress.
func WithLogger(logger Logger) Option {
	return func(o *options) {
		o.logger = logger
	}
}

// WithLanguageModelOptions sets the language model options for the Kronk provider.
func WithLanguageModelOptions(opts ...LanguageModelOption) Option {
	return func(o *options) {
		o.languageModelOptions = append(o.languageModelOptions, opts...)
	}
}

// WithObjectMode sets the object generation mode.
func WithObjectMode(om fantasy.ObjectMode) Option {
	return func(o *options) {
		o.objectMode = om
	}
}

// FmtLogger is a simple logger that prints to stdout using fmt.Printf.
func FmtLogger(_ context.Context, msg string, args ...any) {
	fmt.Printf("%s:", msg)

	for i := 0; i < len(args); i += 2 {
		if i+1 < len(args) {
			fmt.Printf(" %v[%v]", args[i], args[i+1])
		}
	}

	fmt.Println()
}
