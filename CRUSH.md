# CRUSH.md - Fantasy AI SDK

## Build/Test/Lint Commands
- **Build**: `go build ./...`
- **Test all**: `task test` or `go test ./... -count=1` (outputs to `test-results/test-output.log`)
- **Test single**: `go test -run TestName ./package -v`
- **Test with args**: `task test -- -v -run TestName`
- **Test verbose**: `task test-verbose` (outputs to console)
- **Lint**: `task lint` or `golangci-lint run --no-config --enable=gosec,misspell,errcheck,gosimple` (outputs to `test-results/lint.log`)
- **Format**: `task fmt` or `gofmt -s -w .`
- **Modernize**: `task modernize` or `modernize -fix ./...`
- **Clean artifacts**: `task clean` (removes test-results directory)

## Test Output Management
- Test results are saved to `test-results/` directory
- Main tests: `test-results/test-output.log`
- Pre-commit tests: `test-results/pre-commit-test.log`
- Lint results: `test-results/lint.log`
- Coverage: `test-results/coverage.txt` and `test-results/coverage.html`
- Use `task test-verbose` for console output during development

## CI Pipeline Configuration
- **GitLab CI** with Go 1.25, golangci-lint v1.64.8
- **Test output**: Redirected to `test-results/` directory in CI
- **Lint**: Uses `--no-config` with explicit linters (gosec, misspell, errcheck, gosimple)
- **Private modules**: Configured for `gitlab.com/tinyland/*`
- **Artifacts**: Test results and coverage stored for 1 week

## Code Style Guidelines
- **Package naming**: lowercase, single word (ai, openai, anthropic, google)
- **Imports**: standard library first, then third-party, then local packages
- **Error handling**: Use custom error types with structured fields, wrap with context
- **Types**: Use type aliases for function signatures (`type Option = func(*options)`)
- **Naming**: CamelCase for exported, camelCase for unexported
- **Constants**: Use const blocks with descriptive names (ProviderName, DefaultURL)
- **Structs**: Embed anonymous structs for composition (APICallError embeds *AIError)
- **Functions**: Return error as last parameter, use context.Context as first param
- **Testing**: Use testify/assert, table-driven tests, recorder pattern for HTTP mocking
- **Comments**: Godoc format for exported functions, explain complex logic inline
- **JSON**: Use struct tags for marshaling, handle empty values gracefully

## Project Structure
- `/ai` - Core AI abstractions and agent logic
- `/openai`, `/anthropic`, `/google` - Provider implementations
- `/providertests` - Cross-provider integration tests with VCR recordings
- `/examples` - Usage examples for different patterns

## GitLab CLI (glab) Usage

### Installation & Authentication
- **Install**: Follow instructions in the [glab README](https://gitlab.com/gitlab-org/cli)
- **Authenticate**: `glab auth login` (respects `GITLAB_TOKEN` environment variable)
- **Configure Docker helper**: `glab auth configure-docker` after authentication
- **Check version**: `glab version`

### Core Command Structure
`glab <command> <subcommand> [flags]`

### Essential Commands

#### Authentication (`glab auth`)
- `glab auth login` - Authenticate with GitLab
- `glab auth status` - Check authentication status
- `glab auth logout` - Sign out

#### Repository Management (`glab repo`)
- `glab repo clone <repo>` - Clone repository
- `glab repo create` - Create new repository
- `glab repo view` - View repository information
- `glab repo browse` - Open repository in browser

#### Issues (`glab issue`)
- `glab issue list` - List issues (default: open only)
- `glab issue create` - Create new issue interactively
- `glab issue view <id>` - View specific issue
- `glab issue update <id>` - Update issue
- `glab issue close <id>` - Close issue

#### Merge Requests (`glab mr`)
- `glab mr list` - List merge requests
- `glab mr create` - Create merge request (use `--fill` for auto-population)
- `glab mr create <issue_id>` - Create MR from issue
- `glab mr view <id>` - View merge request details
- `glab mr checkout <id>` - Checkout MR branch
- `glab mr approve <id>` - Approve merge request
- `glab mr merge <id>` - Merge merge request
- `glab mr diff <id>` - Show MR diff
- `glab mr note -m "message" <id>` - Add comment to MR

#### CI/CD (`glab ci` & `glab pipeline`)
- `glab ci list` - List CI/CD pipelines
- `glab ci view <pipeline_id>` - View pipeline details
- `glab ci run` - Trigger new pipeline
- `glab ci run --variables-file /path/to/vars.json` - Run with variables
- `glab job list` - List CI jobs
- `glab job trace <job_id>` - View job logs

#### API Access (`glab api`)
- `glab api /projects/:id` - Make authenticated API requests
- `glab api /projects/:id/issues --method POST --field title="Bug" --field description="Fix this"` - POST with data
- `glab api graphql --field query='query { currentUser { name } }'` - GraphQL queries

#### Configuration (`glab config`)
- `glab config set key value` - Set configuration value
- `glab config get key` - Get configuration value
- `glab config list` - List all configuration

### Common Workflows

#### Daily Development
```bash
# Checkout and work on MR
glab mr checkout 123
# Make changes and commit
git push
# View pipeline status
glab ci view
```

#### Creating MR from Issue
```bash
# Create MR from existing issue
glab mr create 123 --fill --label bugfix
# Or create new issue first
glab issue create
glab mr create --fill
```

#### CI/CD Management
```bash
# Trigger pipeline with custom variables
glab ci run --variables-file ci-vars.json
# Monitor running pipeline
glab ci view --watch
# View failed job logs
glab job trace <job_id>
```

#### API Integration
```bash
# Get project info
glab api /projects/:id
# Create issue via API
glab api /projects/:id/issues --method POST \
  --field title="New Issue" \
  --field description="Issue description"
```

### Advanced Features

#### GitLab Duo (Premium/Ultimate)
- `glab duo ask` - Get AI assistance for git commands

#### Variables File Format
```json
[
  { "key": "VAR1", "value": "value1" },
  { "key": "VAR2", "value": "value2" }
]
```

#### Environment Variables (glab 2.0+)
- All glab environment variables are prefixed with `GLAB_`
- `GLAB_TOKEN` - Authentication token
- `GLAB_HOST` - GitLab instance hostname

### Troubleshooting
- **Wrong remote**: Use `git config edit` to fix `glab-resolved = base` settings
- **Completion issues**: Add `setopt completealiases` to ~/.zshrc for 1Password plugin users
- **Multiple remotes**: Set preferred remote with `git config set --append remote.origin.glab-resolved base`

## CI/CD Build Debugging Suite

### Phase 1: Investigation & Analysis
```bash
# Check recent pipeline failures
glab ci list
glab api /projects/:id/pipelines?per_page=10

# Analyze specific failed pipeline
glab api /projects/:id/pipelines/{pipeline_id}/jobs

# Get job logs for failed jobs
glab api /projects/:id/jobs/{job_id}/trace

# Check MR context for pipeline failures
glab mr view {mr_number}
glab mr note --list {mr_number}
```

### Phase 2: Local Reproduction
```bash
# Replicate CI environment locally
docker run --rm -v $(pwd):/app -w /app golang:1.25 bash -c "
  go mod download
  go mod verify
  go mod tidy
  go build -v ./...
  go test -v -race -coverprofile=coverage.txt ./...
"

# Check Go version compatibility
go version
go mod edit -go=1.25  # if needed
```

### Phase 3: Common CI Issues & Solutions

#### Go Module Sync Issues
```bash
# Fix go.mod/go.sum out of sync
go mod download
go mod verify
go mod tidy
git add go.mod go.sum
git commit -m "fix: sync go.mod and go.sum after dependency updates"

# Verify private module access
go env GOPRIVATE GONOSUMDB GONOPROXY
```

#### Dependency Update Failures
```bash
# Handle major version upgrades (like openai-go v2→v3)
go mod tidy
go build ./...  # Check for breaking changes
go test ./...    # Verify tests still pass

# Update imports for major version changes
find . -name "*.go" -exec grep -l "openai-go/v2" {} \;
# Update imports: openai-go/v2 → openai-go/v3
```

#### CI Environment Issues
```bash
# Debug private module access in CI
echo "GOPRIVATE: $GOPRIVATE"
echo "CI_JOB_TOKEN: ${CI_JOB_TOKEN:0:10}..."
cat ~/.netrc

# Test CI commands locally
task lint
task test
go build ./...
```

### Phase 4: Pipeline-Specific Debugging

#### Build Job Failures
```bash
# Check build job specifically
glab api /projects/:id/jobs/{build_job_id}/trace | grep -E "(error|failed|ERROR)"

# Common fixes:
# - Update Go version in .gitlab-ci.yml
# - Fix import paths after major version changes
# - Resolve private module access issues
```

#### Test Job Failures
```bash
# Analyze test failures
glab api /projects/:id/jobs/{test_job_id}/trace | tail -50

# Run specific failing tests locally
go test -v -run TestFailingFunction ./package
go test -v ./... -count=1
```

#### go-mod-verify Job Failures
```bash
# This job fails when go.mod/go.sum changes are detected
# Solution: Commit the changes if they're correct

if [ -n "$(git status --porcelain go.mod go.sum)" ]; then
  echo "go.mod or go.sum changed - reviewing changes:"
  git diff go.mod go.sum
  # If changes are expected (dependency updates):
  git add go.mod go.sum
  git commit -m "chore: update go.sum after dependency changes"
fi
```

### Phase 5: Prevention & Monitoring

#### Pre-commit Checks
```bash
# Add to pre-commit hook or task
task lint
task test
go mod tidy
git diff --exit-code go.mod go.sum || echo "go.mod/go.sum changed - commit them"
```

#### Pipeline Monitoring
```bash
# Monitor running pipelines
watch -n 30 "glab ci list | head -10"

# Set up alerts for critical failures
glab api /projects/:id/pipelines?status=failed
```

### Phase 6: Emergency Procedures

#### Quick Fix for Common Issues
```bash
# 1. Go version mismatch
# Update GO_VERSION in .gitlab-ci.yml and go.mod

# 2. Module sync issues
go mod download && go mod tidy && git add go.mod go.sum && git commit -m "fix: sync modules"

# 3. Import path changes after major version updates
find . -name "*.go" -exec sed -i 's|openai-go/v2|openai-go/v3|g' {} \;

# 4. Private module access
# Ensure GOPRIVATE includes gitlab.com/tinyland/*
# Verify CI_JOB_TOKEN has proper permissions
```

#### Rollback Procedures
```bash
# Revert problematic dependency update
git revert HEAD --no-edit
git push origin main

# Or pin to working version
go get github.com/openai/openai-go/v3@v3.7.0
go mod tidy
```