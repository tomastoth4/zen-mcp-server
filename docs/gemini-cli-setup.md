# Using Gemini CLI with Zen MCP Server

The Gemini CLI provider allows you to use Google's Gemini models through the locally installed Gemini CLI without requiring an API key. This is particularly useful for users who already have access to Gemini through their Google account.

## Prerequisites

1. Install Gemini CLI:
```bash
npm install -g @google/generative-ai-cli
```

2. Authenticate with your Google account:
```bash
gemini auth
```

## How It Works

When Zen MCP Server starts:
1. It checks if the `gemini` command is available in your PATH
2. If found, it automatically registers the Gemini CLI provider
3. You can then use Gemini models without setting a `GEMINI_API_KEY`

## Available Models

The following models are available through the CLI provider:
- `gemini-cli` (default, uses your CLI's configured model)
- `gcli` (alias for gemini-cli)
- `gemini-2.5-flash` (if available in your CLI)
- `gemini-2.5-pro` (if available in your CLI)

## Usage Examples

### Basic Usage
```
"Use gemini-cli to analyze this code for performance issues"
"Ask gcli to explain this error message"
```

### With File References
The Gemini CLI provider automatically converts embedded file content to temporary files and uses Gemini's `@` syntax:
```
"Use gemini-cli to review the architecture of this project"
"Ask gcli to analyze these test files for coverage gaps"
```

### With Images
```
"Use gemini-cli to describe what's in this screenshot"
"Ask gcli to analyze this architecture diagram"
```

## Features and Limitations

### Features
- ✅ No API key required
- ✅ Access to Gemini's large context window (1M tokens)
- ✅ Automatic file reference handling
- ✅ Image analysis support
- ✅ Sandbox mode support (use `-s` flag)

### Limitations
- ❌ No temperature control (fixed at 1.0)
- ❌ No streaming support
- ❌ No token usage statistics
- ❌ No system prompt support (combined with user prompt)
- ❌ No JSON mode
- ❌ Performance overhead from subprocess spawning

## Configuration

No special configuration is needed. The provider is automatically enabled when the Gemini CLI is detected.

To disable the Gemini CLI provider, you can use model restrictions in your `.env` file:
```bash
RESTRICTED_MODELS=gemini-cli,gcli
```

## Troubleshooting

### "Gemini CLI not found" Error
Make sure the Gemini CLI is installed globally:
```bash
which gemini  # Should return the path to gemini
```

### Quota Errors
The CLI provider detects quota exhaustion and provides clear error messages. If you hit quota limits, you'll need to wait or upgrade your Google account.

### Timeout Errors
Large prompts or complex operations may timeout (5-minute limit). Consider breaking down your requests into smaller chunks.

## Integration Priority

The Gemini CLI provider sits between the API-based Gemini provider and custom providers in the priority order:
1. Google Gemini API (if GEMINI_API_KEY is set)
2. **Gemini CLI (if CLI is installed)**
3. OpenAI, X.AI, DIAL providers
4. Custom providers (Ollama, etc.)
5. OpenRouter (catch-all)

This means if you have both a Gemini API key and the CLI installed, the API-based provider will be preferred for better performance.