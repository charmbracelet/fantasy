package providertests

import (
	"os"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/yzma"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestYzmaCommon(t *testing.T) {
	if os.Getenv("YZMA_TEST_MODEL") == "" {
		t.Skip("YZMA_TEST_MODEL not set; skipping yzma provider tests")
	}

	yzmaTestModels := []testModel{
		{os.Getenv("YZMA_TEST_MODEL"), os.Getenv("YZMA_TEST_MODEL"), false},
	}
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
