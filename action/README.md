# AI Change Guard GitHub Action

Runs inside your own CI. Finds retired and retiring AI models and shut-down AI APIs in your code, and can replay your real prompts on the model you use today and the one you plan to switch to. Your API keys stay in your GitHub secrets: the Action calls the providers directly from your runner and sends nothing to AI Change Guard.

## Scan only (no API keys needed)

```yaml
# .github/workflows/ai-change-guard.yml
name: AI Change Guard
on: [pull_request, push]
jobs:
  guard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Navin-kumar0007/ai-change-guard-cli@v1
        with:
          fail-on: retired   # retired | retiring (within 90 days) | none
```

The results appear in the run's **Summary** tab.

## Test a model switch on your own prompts

Save a golden set of real prompts as JSONL (see [`examples/support-bot.jsonl`](../examples/support-bot.jsonl)), add your provider key as a repository secret, then:

```yaml
      - uses: Navin-kumar0007/ai-change-guard-cli@v1
        with:
          prompts: tests/golden.jsonl
          old-model: openai:gpt-4o
          new-model: openai:gpt-6-luna
          runs: 3
          concurrency: 2
          max-calls: 300          # refuses to start above this many calls
          replay-fail-on: broken  # broken | changed | none
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Providers: `openai`, `anthropic`, `google`, `groq`, `openrouter`, `mistral`, `ollama`. Keys are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and so on.

Pull requests from forks do not receive your secrets, so the replay step cannot run on them.

## Nightly watch: is your model still answering the same way?

Providers update models behind the same name and quietly reroute old ones. A watch asks the model you use again every night and compares the answers with a baseline you commit once.

1. Make the baseline on your machine (5 runs per prompt) and commit it next to the golden set:

   ```
   ai-change-guard baseline --set tests/golden.jsonl --model openai:gpt-4o --out tests/golden.baseline.json
   ```

2. Add a nightly workflow:

   ```yaml
   # .github/workflows/ai-watch.yml
   name: AI model watch
   on:
     schedule:
       - cron: "30 2 * * *"
     workflow_dispatch:
   jobs:
     watch:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: Navin-kumar0007/ai-change-guard-cli@v1
           with:
             fail-on: none
             prompts: tests/golden.jsonl
             baseline: tests/golden.baseline.json
             slack-webhook: ${{ secrets.SLACK_WEBHOOK_URL }}   # optional
           env:
             OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
   ```

The step fails, so GitHub emails you, only when something changed: at least 3 of 5 answers to a prompt changed or broke compared with the baseline, the provider served a different model than the one requested (a reroute), or a different snapshot than at baseline time. The Summary lists each changed prompt and why. With `slack-webhook`, the same alert goes to Slack with counts and prompt names only, never prompt or answer text. Prompts edited since the baseline are left out until you make a new baseline. A watch costs prompts × 5 calls a night.

## Prompt Repair

Add `repair: true` to a replay. When the new model breaks a prompt's JSON format, the Action tries a JSON-only instruction in the system prompt, then that plus the provider's JSON mode, re-tests each on the new model, and shows the change that works in the Summary. The step still fails until you apply the fix to your prompt; repair calls count against `max-calls`.

## Verified Prompt Slimming: a shorter system prompt, with proof

```yaml
      - uses: Navin-kumar0007/ai-change-guard-cli@v1
        with:
          prompts: golden.jsonl
          old-model: openai:gpt-5.5
          slim-prompts: true
          calls-per-month: 100000
          open-pr: true
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

A model rewrites each system prompt in the golden set shorter; your prompts are replayed on `old-model` with the old and the new prompt. A rewrite passes only with nothing broken, at most 2% of answers changed and a measured saving. With `open-pr: true` (on the default branch, with `contents: write` and `pull-requests: write`), the Action opens one pull request that puts the passing prompt that saves the most into the code, with the slimming card and the replay report. Only prompts written as one plain string are changed.

## Open a pull request that switches the model (evidence PR)

Add `open-pr: true` to a replay. When every prompt passes (nothing broken, no errors), the Action commits the new model ID to a branch named `ai-change-guard/<old>-to-<new>` and opens a pull request whose description is the replay report: the proof that the switch is safe on your real prompts.

```yaml
# .github/workflows/model-switch.yml
name: Test and propose a model switch
on:
  schedule:
    - cron: "0 3 * * 1"   # every Monday
  workflow_dispatch:
permissions:
  contents: write
  pull-requests: write
  checks: write        # saves the replay report as a check run, for the AI Change Guard dashboard
jobs:
  switch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Navin-kumar0007/ai-change-guard-cli@v1
        with:
          fail-on: none
          prompts: tests/golden.jsonl
          old-model: openai:gpt-4o
          new-model: openai:gpt-6-luna
          open-pr: true
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Also turn on **Settings → Actions → General → Allow GitHub Actions to create and approve pull requests**.

Every replay is also saved as a check run named **AI Change Guard model switch** on the commit (title such as "45 same · 3 changed · 2 broken"), which is where the [AI Change Guard dashboard](https://ai-change-guard.vercel.app/dashboard) reads its migration report. It needs `checks: write`; without it the Action says so in a notice and carries on.

How it behaves:

- It only runs on the default branch (push, schedule or manual run), and only if that branch did not move during the run.
- Only exact uses of the model ID change; fine-tuned models, Bedrock IDs, Azure deployment names and model IDs built at runtime are listed in the PR for you to change by hand. Your code is never sent anywhere: the commit is made through the GitHub API from your own runner.
- Old and new model must be on the same provider (for example `openai:` to `openai:`). A switch between providers changes SDK code, so it is never automated.
- Only files committed in the repository and unchanged by earlier steps of the job are committed. A generated or gitignored file such as `.env` is never added to the PR; it is listed for you instead.
- Through a gateway (for example `openrouter:openai/gpt-4o`), only uses written behind that same prefix change, because that is the route the replay tested. Comments that mention the model are left as they are.
- One pull request per switch. Later runs refresh the report, and rebuild the branch when the default branch moved on, but never push over commits people added. If you close the PR without merging, no new one is opened; after a merge, a later use of the old model gets a new PR.
- A PR opened with the default `github.token` does not start other workflows. To run your CI on it, pass a GitHub App or personal token as `github-token`, or close and reopen the PR.

## Inputs

| Input | Default | Meaning |
| --- | --- | --- |
| `path` | `.` | Folder to scan |
| `fail-on` | `retired` | Fail when a model or API is `retired`, `retiring` (within 90 days), or `none`. A malicious AI package release in a lockfile fails unless `none` |
| `find-savings` | `false` | `true` runs Money-Finder on `old-model`: replays `prompts` on cheaper models of the same provider and writes the savings card to the Summary. With `open-pr: true`, opens the switch pull request to the cheaper model that passed |
| `slim-prompts` | `false` | `true` runs Verified Prompt Slimming on `old-model`: a model rewrites each system prompt in `prompts` shorter, both versions are replayed, and the slimming card goes to the Summary. A rewrite passes only with nothing broken, at most 2% changed and a measured saving. With `open-pr: true`, opens a pull request that puts the shorter prompt (the one that saves most) into the code |
| `writer-model` | | `provider:model` that writes the shorter prompts for `slim-prompts` (default: `old-model`) |
| `candidates` | | Models for `find-savings`, comma-separated `provider:model` (default: the 2 nearest cheaper ones in the catalog) |
| `calls-per-month` | | Your monthly calls to `old-model`, for the monthly saving in the savings or slimming card |
| `mcp-tools` | `false` | `true` starts the MCP servers in the repository's MCP config on this runner and compares their tools with `ai-stack.lock` (needs a lock written with `lock --mcp`). Runs the commands in that config |
| `mcp-env` | | Environment variables `mcp-tools` may fill into the MCP config's `${VAR}` placeholders, comma-separated. Others stay empty, so a config changed in a pull request can't send runner tokens elsewhere |
| `lock-fail-on` | `none` | With an `ai-stack.lock` in the repository: fail when it differs from the code or the catalog (`any`), only from the code (`outdated`), only from the catalog (`catalog`), only a model switched without a new baseline (`unverified`), only MCP tools changed (`mcp`) |
| `online-package-check` | `false` | `true` also checks AI packages online: OSV malicious-code advisories, and npm / PyPI for AI-sounding names that don't exist or are brand new. Sends package names and versions, never code |
| `prompts` | | Golden set (JSONL) for a replay |
| `old-model`, `new-model` | | `provider:model` |
| `runs` | `3` | Runs per prompt on each model |
| `concurrency` | `2` | Calls in flight; use `1` on free-tier keys |
| `max-calls` | `300` | Spend cap for one replay |
| `replay-fail-on` | `broken` | Fail when a prompt is `broken`, `changed`, or `none` |
| `baseline` | | Watch mode: compare the model in use with this baseline file |
| `slack-webhook` | | Watch mode: Slack incoming-webhook URL for alerts (from a secret) |
| `repair` | `false` | When a JSON format breaks, try prompt fixes on the new model and show the one that works |
| `open-pr` | `false` | After a passing replay, open a pull request that switches the model |
| `github-token` | `github.token` | Token for the pull request |

## Development

`npm run build:action` type-checks, bundles `src/main.ts` into `bundle/index.js`, and copies the catalog into `data/`. Commit both: GitHub runs the bundle without installing dependencies. A test fails if `data/` falls behind `packages/catalog/data`.
