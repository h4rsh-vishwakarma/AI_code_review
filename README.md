# AI Code Review Bot

Automated PR reviewer that uses a multi-agent LLM pipeline to find bugs, security vulnerabilities, performance issues, and style problems. Posts inline comments and a summary directly on your pull requests.

## Features

- **Multi-Agent Review Pipeline** — 4 specialized agents (Security, Bugs, Performance, Style) review in parallel, then an aggregator deduplicates and ranks findings
- **RAG Context** — Retrieves relevant codebase context from a vector store so the LLM understands your code beyond just the diff
- **AST Semantic Analysis** — Parses Python code into ASTs to skip cosmetic changes and focus on meaningful modifications
- **Risk Scoring** — Calculates a 0-100 risk score based on files touched, sensitivity, author history, and findings
- **Custom Rule Engine** — Define team-specific rules in `.reviewbot.yml` with regex patterns, path filters, and auto-assignment
- **Feedback Loop** — Captures developer reactions (thumbs up/down, replies) to improve future reviews
- **Auto-Routing** — Labels PRs by risk level and auto-assigns reviewers based on changed code areas
- **Language-Aware** — Language-specific review prompts for Python, JavaScript, TypeScript, Go, Java, Rust, SQL

## Architecture

```
GitHub PR Event
  → GitHub Actions trigger
    → Fetch PR diff & metadata
    → [Optional] AST analysis (filter cosmetic changes)
    → [Optional] RAG context retrieval
    → Custom rule evaluation
    → Multi-agent LLM review (parallel)
      ├── Security Agent
      ├── Bug Detection Agent
      ├── Performance Agent
      └── Style Agent
    → Aggregator (deduplicate + rank)
    → Risk scoring
    → Post inline comments + summary
    → Auto-label + auto-assign reviewers
```

## Quick Start

### 1. Add secrets to your GitHub repo

Go to **Settings → Secrets and variables → Actions** and add:
- `OPENAI_API_KEY` — Your OpenAI API key

`GITHUB_TOKEN` is automatically provided by GitHub Actions.

### 2. Copy workflow and config

Copy these files to your repo:
- `.github/workflows/code_review.yml`
- `.reviewbot.yml` (customize rules for your team)

### 3. Install dependencies (for local development)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 4. Run locally (dry run)

```bash
export GITHUB_TOKEN=ghp_...
export OPENAI_API_KEY=sk-...

python src/main.py --repo owner/repo --pr 123 --dry-run
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_TOKEN` | Yes | — | GitHub API token |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `MODEL_NAME` | No | `gpt-4o` | LLM model (or fine-tuned model ID) |
| `MAX_FILES_TO_REVIEW` | No | `20` | Max files per PR |
| `MAX_DIFF_SIZE` | No | `10000` | Max chars per file diff |
| `TEAM_CONFIG` | No | `{}` | JSON: `{"auth": ["@user1"], "database": ["@user2"]}` |

### Custom Rules (`.reviewbot.yml`)

```yaml
rules:
  - name: no-eval
    pattern: "\\beval\\s*\\("
    severity: critical
    message: "eval() enables code injection"
    languages: ["python", "javascript"]

  - name: migration-review
    paths: ["db/migrations/**"]
    severity: critical
    message: "Migrations require DBA review"
    auto_assign: ["@dba-lead"]
    auto_label: ["database-review"]
```

### CLI Options

```
--repo          GitHub repo (owner/repo)
--pr            PR number
--config        Path to .reviewbot.yml (default: .reviewbot.yml)
--enable-rag    Enable RAG context retrieval
--enable-ast    Enable AST semantic analysis
--dry-run       Print results without posting to GitHub
--min-severity  Minimum severity to post (critical/warning/suggestion)
--team-config   JSON mapping risk areas to reviewers
```

## Advanced: Fine-Tuning

1. Collect feedback over time (reactions + replies on bot comments)
2. Export training data:
   ```python
   from src.feedback import FeedbackCollector
   collector = FeedbackCollector()
   collector.export_training_data("training_data.jsonl")
   ```
3. Fine-tune via OpenAI:
   ```bash
   openai api fine_tuning.jobs.create -t training_data.jsonl -m gpt-4o-mini-2024-07-18
   ```
4. Update `MODEL_NAME` to your fine-tuned model ID

## Testing

```bash
pytest
```

## Stack

- **Python 3.11+**
- **LangChain** — LLM orchestration, prompt templates, output parsing
- **OpenAI API** — GPT-4o for code analysis
- **PyGithub** — GitHub API interactions
- **ChromaDB** — Vector store for RAG
- **GitHub Actions** — CI/CD trigger and automation
