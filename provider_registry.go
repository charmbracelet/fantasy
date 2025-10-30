package fantasy

import (
	"encoding/json"
	"fmt"
	"sync"
)

// providerDataJSON is the serialized wrapper used by the registry.
type providerDataJSON struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data"`
}

// UnmarshalFunc converts raw JSON into a ProviderOptionsData implementation.
type UnmarshalFunc func([]byte) (ProviderOptionsData, error)

var (
	providerRegistry = make(map[string]UnmarshalFunc)
	registryMutex    sync.RWMutex
)

// RegisterProviderType registers a provider type ID with its unmarshal function.
// Type IDs must be globally unique (e.g. "openai.options").
func RegisterProviderType(typeID string, unmarshalFn UnmarshalFunc) {
	registryMutex.Lock()
	defer registryMutex.Unlock()
	providerRegistry[typeID] = unmarshalFn
}

// unmarshalProviderData routes a typed payload to the correct constructor.
func unmarshalProviderData(data []byte) (ProviderOptionsData, error) {
	var pj providerDataJSON
	if err := json.Unmarshal(data, &pj); err != nil {
		return nil, err
	}

	registryMutex.RLock()
	unmarshalFn, exists := providerRegistry[pj.Type]
	registryMutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("unknown provider data type: %s", pj.Type)
	}

	return unmarshalFn(pj.Data)
}

// unmarshalProviderDataMap is a helper for unmarshaling maps of provider data.
func unmarshalProviderDataMap(data map[string]json.RawMessage) (map[string]ProviderOptionsData, error) {
	result := make(map[string]ProviderOptionsData)
	for provider, rawData := range data {
		providerData, err := unmarshalProviderData(rawData)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal provider data for %s: %w", provider, err)
		}
		result[provider] = providerData
	}
	return result, nil
}

// UnmarshalProviderOptions unmarshals a map of provider options by type.
func UnmarshalProviderOptions(data map[string]json.RawMessage) (ProviderOptions, error) {
	return unmarshalProviderDataMap(data)
}

// UnmarshalProviderMetadata unmarshals a map of provider metadata by type.
func UnmarshalProviderMetadata(data map[string]json.RawMessage) (ProviderMetadata, error) {
	return unmarshalProviderDataMap(data)
}
