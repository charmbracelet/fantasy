package fantasy

import "github.com/google/uuid"

// NewID generates identifiers for provider-generated content parts, such as
// streamed source blocks. It is a package-level variable so tests can swap
// in a deterministic generator; production code should not reassign it.
var NewID = uuid.NewString
