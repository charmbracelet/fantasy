package providertests

import (
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/yzma"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

var (
	yzmaTestModels = []testModel{
		{"Qwen2.5-VL-3B-Instruct-Q8_0.gguf", "Qwen2.5-VL-3B-Instruct-Q8_0.gguf", false},
	}
)

func TestYzmaCommon(t *testing.T) {
	var pairs []builderPair
	for _, m := range yzmaTestModels {
		pairs = append(pairs, builderPair{m.name, yzmaBuilder(t, m.model), nil, nil})
	}
	testCommon(t, pairs)
}

func yzmaBuilder(t *testing.T, model string) builderFunc {
	provider, err := yzma.New()
	if err != nil {
		panic(err)
	}

	mdl, err := provider.LanguageModel(t.Context(), model)
	if err != nil {
		panic(err)
	}

	return func(t *testing.T, r *recorder.Recorder) (fantasy.LanguageModel, error) {
		return mdl, nil
	}
}
