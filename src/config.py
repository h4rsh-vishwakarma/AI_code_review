"""Configuration management for the AI Code Review Bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Required ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# --- LLM Provider ---
# Supported: "groq" (free), "gemini" (free), "openai" (paid)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# --- LLM Settings ---
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# --- Review Settings ---
MAX_FILES_TO_REVIEW = int(os.getenv("MAX_FILES_TO_REVIEW", "20"))
MAX_DIFF_SIZE = int(os.getenv("MAX_DIFF_SIZE", "10000"))

# --- RAG Settings ---
PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Feedback ---
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", str(PROJECT_ROOT / "data" / "feedback.jsonl"))

# --- Risk Scoring ---
RISK_HIGH_THRESHOLD = int(os.getenv("RISK_HIGH_THRESHOLD", "80"))
RISK_MEDIUM_THRESHOLD = int(os.getenv("RISK_MEDIUM_THRESHOLD", "50"))

# --- File filters ---
SKIP_EXTENSIONS = {
    ".lock", ".sum", ".mod",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".zip", ".tar", ".gz",
}

# Filename patterns to skip (checked via substring/endswith)
SKIP_FILENAME_PATTERNS = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Pipfile.lock", "poetry.lock", "composer.lock",
    "Cargo.lock", "Gemfile.lock", "go.sum",
    ".min.js", ".min.css",
}

SKIP_PATHS = {
    "vendor/", "node_modules/", "dist/", "build/",
    ".git/", "__pycache__/", ".next/", "coverage/",
}

# --- Language detection ---
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sql": "sql",
    ".sh": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".tf": "terraform",
    ".dockerfile": "docker",
}


def detect_language(filename: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(filename).suffix.lower()
    if Path(filename).name.lower() == "dockerfile":
        return "docker"
    return LANGUAGE_MAP.get(ext, "unknown")


def should_skip_file(filename: str) -> bool:
    """Check if a file should be skipped during review."""
    ext = Path(filename).suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return True
    for pattern in SKIP_FILENAME_PATTERNS:
        if filename.endswith(pattern) or filename == pattern:
            return True
    for skip_path in SKIP_PATHS:
        if skip_path in filename:
            return True
    return False
