"""Tests for parsers module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parsers import ReviewComment, Severity, Category, format_inline_comment, RiskFactors


class TestReviewComment:
    def test_create_valid_comment(self):
        comment = ReviewComment(
            file="src/auth.py",
            line=42,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            message="SQL injection vulnerability",
            suggestion="Use parameterized queries",
            confidence=0.95,
        )
        assert comment.file == "src/auth.py"
        assert comment.line == 42
        assert comment.severity == Severity.CRITICAL
        assert comment.confidence == 0.95

    def test_default_confidence(self):
        comment = ReviewComment(
            file="test.py", line=1, severity=Severity.WARNING,
            category=Category.BUG, message="msg", suggestion="fix",
        )
        assert comment.confidence == 0.8

    def test_confidence_bounds(self):
        import pytest
        with pytest.raises(Exception):
            ReviewComment(
                file="test.py", line=1, severity=Severity.WARNING,
                category=Category.BUG, message="msg", suggestion="fix",
                confidence=1.5,
            )


class TestFormatInlineComment:
    def test_critical_format(self):
        comment = ReviewComment(
            file="auth.py", line=10, severity=Severity.CRITICAL,
            category=Category.SECURITY, message="Hardcoded password",
            suggestion="Use environment variables", confidence=0.9,
        )
        result = format_inline_comment(comment)
        assert "CRITICAL" in result
        assert "security" in result
        assert "Hardcoded password" in result
        assert "Use environment variables" in result

    def test_low_confidence_shows_warning(self):
        comment = ReviewComment(
            file="x.py", line=1, severity=Severity.SUGGESTION,
            category=Category.STYLE, message="msg", suggestion="fix",
            confidence=0.4,
        )
        result = format_inline_comment(comment)
        assert "Confidence" in result
        assert "please verify" in result

    def test_high_confidence_no_warning(self):
        comment = ReviewComment(
            file="x.py", line=1, severity=Severity.WARNING,
            category=Category.BUG, message="msg", suggestion="fix",
            confidence=0.9,
        )
        result = format_inline_comment(comment)
        assert "Confidence" not in result


class TestRiskFactors:
    def test_defaults(self):
        factors = RiskFactors()
        assert factors.files_changed == 0
        assert factors.touches_auth is False
        assert factors.has_tests is False

    def test_custom_values(self):
        factors = RiskFactors(
            files_changed=10,
            lines_changed=500,
            touches_auth=True,
            has_tests=True,
        )
        assert factors.files_changed == 10
        assert factors.touches_auth is True
