"""Structured output models for review comments."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"


class Category(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BEST_PRACTICE = "best_practice"


class ReviewComment(BaseModel):
    """A single review comment from the LLM."""
    file: str = Field(description="File path")
    line: int = Field(description="Line number in the diff")
    severity: Severity = Field(description="Issue severity")
    category: Category = Field(description="Issue category")
    message: str = Field(description="Description of the issue")
    suggestion: str = Field(description="How to fix it")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score 0-1")


class FileReview(BaseModel):
    """Review results for a single file."""
    filename: str
    language: str
    comments: list[ReviewComment] = Field(default_factory=list)


class ReviewSummary(BaseModel):
    """Overall PR review summary."""
    summary: str = Field(description="1-3 sentence summary")
    verdict: str = Field(description="APPROVE, REQUEST_CHANGES, or COMMENT")
    critical_count: int = 0
    warning_count: int = 0
    suggestion_count: int = 0
    risk_score: int = Field(default=0, ge=0, le=100)


class RiskFactors(BaseModel):
    """Factors contributing to PR risk score."""
    files_changed: int = 0
    lines_changed: int = 0
    touches_auth: bool = False
    touches_payments: bool = False
    touches_infra: bool = False
    touches_database: bool = False
    new_dependencies: bool = False
    critical_findings: int = 0
    author_commit_count: int = 0
    has_tests: bool = False


def format_inline_comment(comment: ReviewComment) -> str:
    """Format a ReviewComment as a markdown GitHub comment."""
    severity_emoji = {
        Severity.CRITICAL: "\U0001f534",   # red circle
        Severity.WARNING: "\U0001f7e1",    # yellow circle
        Severity.SUGGESTION: "\U0001f535", # blue circle
    }
    emoji = severity_emoji.get(comment.severity, "")
    lines = [
        f"{emoji} **{comment.severity.value.upper()}** | {comment.category.value}",
        "",
        comment.message,
        "",
        f"**Suggestion:** {comment.suggestion}",
    ]
    if comment.confidence < 0.6:
        lines.append(f"\n_Confidence: {comment.confidence:.0%} — please verify_")
    return "\n".join(lines)
