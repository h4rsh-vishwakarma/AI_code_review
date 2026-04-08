"""PR risk scoring and auto-routing engine.

Calculates a 0-100 risk score based on multiple factors and routes the PR
to appropriate reviewers based on risk level.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from parsers import RiskFactors
from config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD

logger = logging.getLogger(__name__)

# Sensitive path patterns and their risk weights
SENSITIVE_PATHS = {
    "auth": {"patterns": [r"auth", r"login", r"session", r"oauth", r"jwt", r"token"], "weight": 15},
    "payments": {"patterns": [r"payment", r"billing", r"stripe", r"checkout", r"invoice"], "weight": 20},
    "database": {"patterns": [r"migration", r"schema", r"model", r"\.sql$"], "weight": 15},
    "infra": {"patterns": [r"\.ya?ml$", r"\.tf$", r"dockerfile", r"docker-compose", r"\.env"], "weight": 12},
    "security": {"patterns": [r"crypto", r"encrypt", r"secret", r"password", r"credential"], "weight": 18},
    "ci_cd": {"patterns": [r"\.github/workflows", r"\.gitlab-ci", r"jenkinsfile", r"\.circleci"], "weight": 10},
}

# Lockfile patterns (indicate dependency changes)
LOCKFILE_PATTERNS = [
    r"package-lock\.json$", r"yarn\.lock$", r"pnpm-lock\.yaml$",
    r"Gemfile\.lock$", r"poetry\.lock$", r"Pipfile\.lock$",
    r"go\.sum$", r"Cargo\.lock$", r"composer\.lock$",
]

# Test file patterns
TEST_PATTERNS = [
    r"test[_/]", r"_test\.", r"\.test\.", r"\.spec\.",
    r"tests/", r"__tests__/", r"spec/",
]


@dataclass
class RiskAssessment:
    """Complete risk assessment for a PR."""
    score: int  # 0-100
    level: str  # low, medium, high, critical
    factors: RiskFactors
    breakdown: dict[str, int]  # factor_name -> points contributed
    recommendations: list[str]
    suggested_labels: list[str]
    suggested_reviewers: list[str]


class RiskScorer:
    """Calculates PR risk score and provides routing recommendations."""

    def __init__(self, team_config: dict | None = None):
        """
        Args:
            team_config: Optional dict mapping risk areas to reviewer lists.
                Example: {"auth": ["@security-team"], "database": ["@dba-team"]}
        """
        self.team_config = team_config or {}

    def assess(
        self,
        files: list[dict],
        findings: list[dict],
        author_stats: dict,
    ) -> RiskAssessment:
        """Calculate full risk assessment for a PR.

        Args:
            files: Changed files with keys: filename, additions, deletions, patch.
            findings: Review findings from the LLM.
            author_stats: Author contribution stats from GitHub.
        """
        factors = self._extract_factors(files, findings, author_stats)
        breakdown = self._calculate_breakdown(factors)
        score = min(100, sum(breakdown.values()))
        level = self._score_to_level(score)
        recommendations = self._generate_recommendations(factors, level)
        labels = self._suggest_labels(factors, level)
        reviewers = self._suggest_reviewers(factors)

        return RiskAssessment(
            score=score,
            level=level,
            factors=factors,
            breakdown=breakdown,
            recommendations=recommendations,
            suggested_labels=labels,
            suggested_reviewers=reviewers,
        )

    def _extract_factors(
        self,
        files: list[dict],
        findings: list[dict],
        author_stats: dict,
    ) -> RiskFactors:
        """Extract risk factors from PR data."""
        filenames = [f["filename"] for f in files]
        total_additions = sum(f.get("additions", 0) for f in files)
        total_deletions = sum(f.get("deletions", 0) for f in files)

        return RiskFactors(
            files_changed=len(files),
            lines_changed=total_additions + total_deletions,
            touches_auth=self._matches_patterns(filenames, SENSITIVE_PATHS["auth"]["patterns"]),
            touches_payments=self._matches_patterns(filenames, SENSITIVE_PATHS["payments"]["patterns"]),
            touches_infra=self._matches_patterns(filenames, SENSITIVE_PATHS["infra"]["patterns"]),
            touches_database=self._matches_patterns(filenames, SENSITIVE_PATHS["database"]["patterns"]),
            new_dependencies=self._matches_patterns(filenames, LOCKFILE_PATTERNS),
            critical_findings=sum(1 for f in findings if f.get("severity") == "critical"),
            author_commit_count=author_stats.get("total_commits", 0),
            has_tests=self._matches_patterns(filenames, TEST_PATTERNS),
        )

    def _matches_patterns(self, filenames: list[str], patterns: list[str]) -> bool:
        """Check if any filename matches any of the patterns."""
        for filename in filenames:
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    return True
        return False

    def _calculate_breakdown(self, factors: RiskFactors) -> dict[str, int]:
        """Calculate point contribution from each factor."""
        breakdown = {}

        # Size-based risk
        if factors.files_changed > 15:
            breakdown["large_pr_files"] = 15
        elif factors.files_changed > 8:
            breakdown["medium_pr_files"] = 8

        if factors.lines_changed > 1000:
            breakdown["large_pr_lines"] = 15
        elif factors.lines_changed > 500:
            breakdown["medium_pr_lines"] = 8

        # Sensitive area risk
        if factors.touches_auth:
            breakdown["auth_changes"] = SENSITIVE_PATHS["auth"]["weight"]
        if factors.touches_payments:
            breakdown["payment_changes"] = SENSITIVE_PATHS["payments"]["weight"]
        if factors.touches_database:
            breakdown["database_changes"] = SENSITIVE_PATHS["database"]["weight"]
        if factors.touches_infra:
            breakdown["infra_changes"] = SENSITIVE_PATHS["infra"]["weight"]

        # Dependency risk
        if factors.new_dependencies:
            breakdown["dependency_changes"] = 10

        # Findings risk
        if factors.critical_findings > 0:
            breakdown["critical_findings"] = min(30, factors.critical_findings * 15)

        # Author risk (new contributors need more scrutiny)
        if factors.author_commit_count < 5:
            breakdown["new_contributor"] = 10
        elif factors.author_commit_count < 20:
            breakdown["junior_contributor"] = 5

        # Missing tests penalty
        if not factors.has_tests and factors.lines_changed > 100:
            breakdown["no_tests"] = 10

        return breakdown

    def _score_to_level(self, score: int) -> str:
        """Convert numeric score to risk level."""
        if score >= RISK_HIGH_THRESHOLD:
            return "critical"
        elif score >= RISK_MEDIUM_THRESHOLD:
            return "high"
        elif score >= 25:
            return "medium"
        return "low"

    def _generate_recommendations(self, factors: RiskFactors, level: str) -> list[str]:
        """Generate human-readable recommendations based on risk factors."""
        recs = []

        if level in ("critical", "high"):
            recs.append("This PR requires careful manual review before merging.")

        if factors.touches_auth:
            recs.append("Authentication/authorization code modified — verify access control is correct.")
        if factors.touches_payments:
            recs.append("Payment-related code modified — consider extra security review.")
        if factors.touches_database:
            recs.append("Database schema/migration changes — verify backward compatibility and rollback plan.")
        if factors.new_dependencies:
            recs.append("Dependencies changed — audit new packages for known vulnerabilities.")
        if not factors.has_tests and factors.lines_changed > 100:
            recs.append("No test files modified — consider adding tests for the changed code.")
        if factors.critical_findings > 0:
            recs.append(f"{factors.critical_findings} critical issue(s) found — must be addressed before merge.")
        if factors.files_changed > 15:
            recs.append("Large PR — consider splitting into smaller, focused PRs for easier review.")

        return recs

    def _suggest_labels(self, factors: RiskFactors, level: str) -> list[str]:
        """Suggest GitHub labels based on risk assessment."""
        labels = [f"risk:{level}"]

        if factors.touches_auth:
            labels.append("security-review")
        if factors.touches_payments:
            labels.append("payment-review")
        if factors.touches_database:
            labels.append("database-review")
        if factors.critical_findings > 0:
            labels.append("critical-issues")
        if not factors.has_tests and factors.lines_changed > 100:
            labels.append("needs-tests")

        return labels

    def _suggest_reviewers(self, factors: RiskFactors) -> list[str]:
        """Suggest reviewers based on changed areas."""
        reviewers = set()

        if factors.touches_auth and "auth" in self.team_config:
            reviewers.update(self.team_config["auth"])
        if factors.touches_payments and "payments" in self.team_config:
            reviewers.update(self.team_config["payments"])
        if factors.touches_database and "database" in self.team_config:
            reviewers.update(self.team_config["database"])
        if factors.touches_infra and "infra" in self.team_config:
            reviewers.update(self.team_config["infra"])

        return list(reviewers)
