package vercel

import openaipkg "charm.land/fantasy/providers/openai"

// ToPrompt is the prompt converter this provider installs on its language
// models. Exported so the shared prompt corpus can record every
// chat-completions converter side by side; see providers/internal/prompttest.
var ToPrompt openaipkg.LanguageModelToPromptFunc = languageModelToPrompt
