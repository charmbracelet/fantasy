package providertests

import (
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/anthropic"
	"charm.land/fantasy/providers/openaicompat"
	"charm.land/fantasy/replaytest"
	"github.com/stretchr/testify/require"
)

type shapeMeta struct {
	Provider string `json:"provider"`
}

func TestFixtureShapes(t *testing.T) {
	root := filepath.Join("testdata", "shapes")

	var cases []string
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if _, statErr := os.Stat(filepath.Join(path, "meta.json")); statErr == nil {
				cases = append(cases, path)
				return fs.SkipDir
			}
		}
		return nil
	})
	require.NoError(t, err)
	require.NotEmpty(t, cases, "no shape fixtures found under %s", root)
	sort.Strings(cases)

	for _, dir := range cases {
		t.Run(strings.TrimPrefix(filepath.ToSlash(dir), filepath.ToSlash(root)+"/"), func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(dir, "meta.json"))
			require.NoError(t, err)
			var meta shapeMeta
			require.NoError(t, json.Unmarshal(data, &meta))

			fixture, err := replaytest.Load(dir)
			require.NoError(t, err)
			server := replaytest.Serve(t, fixture)

			model := shapeLanguageModel(t, meta.Provider, server.URL())
			stream, err := model.Stream(t.Context(), fantasy.Call{
				Prompt: fantasy.Prompt{
					{Role: fantasy.MessageRoleUser, Content: []fantasy.MessagePart{fantasy.TextPart{Text: "hi"}}},
				},
			})
			require.NoError(t, err)

			records := replaytest.Collect(stream)
			replaytest.AssertGolden(t, filepath.Join(dir, "parts.golden.json"), records)
		})
	}
}

func shapeLanguageModel(t *testing.T, provider, baseURL string) fantasy.LanguageModel {
	t.Helper()
	switch provider {
	case "openaicompat":
		p, err := openaicompat.New(
			openaicompat.WithBaseURL(baseURL),
			openaicompat.WithAPIKey("test-key"),
		)
		require.NoError(t, err)
		lm, err := p.LanguageModel(t.Context(), "test-model")
		require.NoError(t, err)
		return lm
	case "anthropic":
		p, err := anthropic.New(
			anthropic.WithBaseURL(baseURL),
			anthropic.WithAPIKey("test-key"),
		)
		require.NoError(t, err)
		lm, err := p.LanguageModel(t.Context(), "test-model")
		require.NoError(t, err)
		return lm
	default:
		t.Fatalf("unknown provider %q in meta.json", provider)
		return nil
	}
}
