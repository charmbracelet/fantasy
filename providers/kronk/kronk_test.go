package kronk

import (
	"reflect"
	"testing"

	"charm.land/fantasy"
	"github.com/ardanlabs/kronk/sdk/kronk/model"
	"github.com/ardanlabs/kronk/sdk/tools/models"
)

var _ func(...Option) (fantasy.Provider, error) = New

func TestModelConfigArtifacts(t *testing.T) {
	tests := []struct {
		name         string
		modelOptions []model.Option
		wantProj     string
		wantMTP      string
	}{
		{
			name:     "downloaded companions",
			wantProj: "downloaded-proj.gguf",
			wantMTP:  "downloaded-mtp.gguf",
		},
		{
			name: "configured companions",
			modelOptions: []model.Option{
				model.WithProjFile("custom-proj.gguf"),
				model.WithMTPDrafterFile("custom-mtp.gguf"),
			},
			wantProj: "custom-proj.gguf",
			wantMTP:  "custom-mtp.gguf",
		},
	}
	mp := models.Path{
		ModelFiles: []string{"model.gguf"},
		ProjFile:   "downloaded-proj.gguf",
		MTPFile:    "downloaded-mtp.gguf",
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := provider{options: options{modelOptions: tt.modelOptions}}
			cfg := p.modelConfig(mp)
			if got, want := cfg.ModelFiles, mp.ModelFiles; !reflect.DeepEqual(got, want) {
				t.Errorf("ModelFiles: got %v, want %v", got, want)
			}
			if got := cfg.ProjFile; got != tt.wantProj {
				t.Errorf("ProjFile: got %q, want %q", got, tt.wantProj)
			}
			if got := cfg.MTPDrafterFile; got != tt.wantMTP {
				t.Errorf("MTPDrafterFile: got %q, want %q", got, tt.wantMTP)
			}
		})
	}
}
