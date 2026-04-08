"""Tests for risk scoring module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk_scorer import RiskScorer


class TestRiskScorer:
    def setup_method(self):
        self.scorer = RiskScorer(team_config={
            "auth": ["security-lead"],
            "database": ["dba-team"],
        })

    def test_low_risk_pr(self):
        files = [
            {"filename": "README.md", "additions": 5, "deletions": 2, "patch": "+hello"},
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 100})
        assert result.score < 25
        assert result.level == "low"

    def test_high_risk_auth_changes(self):
        files = [
            {"filename": "src/auth/login.py", "additions": 50, "deletions": 30, "patch": "+code"},
            {"filename": "src/auth/session.py", "additions": 20, "deletions": 10, "patch": "+code"},
        ]
        findings = [{"severity": "critical", "message": "SQL injection"}]
        result = self.scorer.assess(files, findings, author_stats={"total_commits": 5})
        assert result.score >= 25
        assert result.factors.touches_auth is True

    def test_payment_code_flagged(self):
        files = [
            {"filename": "src/billing/payment_handler.py", "additions": 100, "deletions": 0, "patch": "+code"},
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 50})
        assert result.factors.touches_payments is True
        assert "payment-review" in result.suggested_labels

    def test_new_contributor_risk(self):
        files = [
            {"filename": "src/utils.py", "additions": 10, "deletions": 5, "patch": "+code"},
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 2})
        assert "new_contributor" in result.breakdown

    def test_large_pr_risk(self):
        files = [
            {"filename": f"src/file{i}.py", "additions": 100, "deletions": 50, "patch": "+code"}
            for i in range(20)
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 100})
        assert "large_pr_files" in result.breakdown
        assert "large_pr_lines" in result.breakdown

    def test_missing_tests_penalty(self):
        files = [
            {"filename": "src/feature.py", "additions": 200, "deletions": 0, "patch": "+code"},
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 50})
        assert "no_tests" in result.breakdown

    def test_has_tests_no_penalty(self):
        files = [
            {"filename": "src/feature.py", "additions": 200, "deletions": 0, "patch": "+code"},
            {"filename": "tests/test_feature.py", "additions": 100, "deletions": 0, "patch": "+code"},
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 50})
        assert "no_tests" not in result.breakdown

    def test_reviewer_suggestions(self):
        files = [
            {"filename": "src/auth/oauth.py", "additions": 10, "deletions": 5, "patch": "+code"},
            {"filename": "db/migrations/001.sql", "additions": 20, "deletions": 0, "patch": "+code"},
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 50})
        assert "security-lead" in result.suggested_reviewers
        assert "dba-team" in result.suggested_reviewers

    def test_recommendations_generated(self):
        files = [
            {"filename": "src/auth/login.py", "additions": 50, "deletions": 10, "patch": "+code"},
        ]
        findings = [{"severity": "critical", "message": "issue"}]
        result = self.scorer.assess(files, findings, author_stats={"total_commits": 50})
        assert len(result.recommendations) > 0

    def test_dependency_changes_detected(self):
        files = [
            {"filename": "package-lock.json", "additions": 500, "deletions": 200, "patch": "+code"},
        ]
        result = self.scorer.assess(files, findings=[], author_stats={"total_commits": 50})
        assert result.factors.new_dependencies is True
