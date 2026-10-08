package vercel

import openaipkg "charm.land/fantasy/providers/openai"

// ToPrompt is the prompt converter this provider installs on its language
// models, exported for the shared corpus in providers/internal/prompttest.
var ToPrompt openaipkg.LanguageModelToPromptFunc = languageModelToPrompt
