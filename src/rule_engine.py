"""Custom rule engine — loads team-defined rules from .reviewbot.yml."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from fnmatch import fnmatch
from pathlib import PurePath

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A single custom review rule."""
    name: str
    severity: str = "warning"
    message: str = ""
    paths: list[str] = field(default_factory=list)       # glob patterns to include
    exclude: list[str] = field(default_factory=list)      # glob patterns to exclude
    pattern: str = ""                                      # regex to match in diff
    languages: list[str] = field(default_factory=list)    # language filter
    auto_assign: list[str] = field(default_factory=list)  # reviewers to assign
    auto_label: list[str] = field(default_factory=list)   # labels to apply
    condition: str = ""                                    # semantic condition (for LLM)


@dataclass
class RuleMatch:
    """A rule that matched against a file/diff."""
    rule: Rule
    file: str
    line: int = 0
    matched_text: str = ""


class RuleEngine:
    """Loads and evaluates custom review rules from .reviewbot.yml.

    Example .reviewbot.yml:
    ```yaml
    rules:
      - name: no-console-log
        pattern: "console\\.log\\("
        severity: warning
        message: "Remove console.log before merging"
        paths: ["src/**/*.ts", "src/**/*.js"]
        exclude: ["src/debug/**"]

      - name: migration-review
        paths: ["db/migrations/**"]
        severity: critical
        message: "Database migrations require manual review from @db-team"
        auto_assign: ["@db-team-lead"]
        auto_label: ["database-review"]

      - name: no-any-type
        pattern: ":\\s*any[\\s;,)]"
        severity: warning
        message: "Avoid using 'any' type — use a specific type or 'unknown'"
        languages: ["typescript"]
    ```
    """

    def __init__(self, config_path: str | None = None):
        self.rules: list[Rule] = []
        self.config_path = config_path
        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        """Load rules from a YAML config file."""
        path = Path(config_path)
        if not path.exists():
            logger.info("No .reviewbot.yml found at %s", config_path)
            return

        try:
            with open(path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if not config or "rules" not in config:
                return

            for rule_data in config["rules"]:
                rule = Rule(
                    name=rule_data.get("name", "unnamed"),
                    severity=rule_data.get("severity", "warning"),
                    message=rule_data.get("message", ""),
                    paths=rule_data.get("paths", []),
                    exclude=rule_data.get("exclude", []),
                    pattern=rule_data.get("pattern", ""),
                    languages=rule_data.get("languages", []),
                    auto_assign=rule_data.get("auto_assign", []),
                    auto_label=rule_data.get("auto_label", []),
                    condition=rule_data.get("condition", ""),
                )
                self.rules.append(rule)

            logger.info("Loaded %d custom rules from %s", len(self.rules), config_path)
        except Exception as e:
            logger.error("Failed to load rules from %s: %s", config_path, e)

    def load_from_string(self, yaml_content: str) -> None:
        """Load rules from a YAML string (useful for testing)."""
        try:
            config = yaml.safe_load(yaml_content)
            if config and "rules" in config:
                for rule_data in config["rules"]:
                    self.rules.append(Rule(
                        name=rule_data.get("name", "unnamed"),
                        severity=rule_data.get("severity", "warning"),
                        message=rule_data.get("message", ""),
                        paths=rule_data.get("paths", []),
                        exclude=rule_data.get("exclude", []),
                        pattern=rule_data.get("pattern", ""),
                        languages=rule_data.get("languages", []),
                        auto_assign=rule_data.get("auto_assign", []),
                        auto_label=rule_data.get("auto_label", []),
                        condition=rule_data.get("condition", ""),
                    ))
        except Exception as e:
            logger.error("Failed to parse rules YAML: %s", e)

    def get_rules_for_file(self, filename: str, language: str) -> list[dict]:
        """Get all rules applicable to a given file.

        Returns list of dicts with keys: name, message, severity.
        """
        applicable = []
        for rule in self.rules:
            if self._rule_applies(rule, filename, language):
                applicable.append({
                    "name": rule.name,
                    "message": rule.message,
                    "severity": rule.severity,
                })
        return applicable

    def evaluate(self, filename: str, language: str, patch: str) -> list[RuleMatch]:
        """Evaluate all rules against a file's diff.

        Returns matches for pattern-based rules. Condition-based rules are
        returned as applicable rules for the LLM to evaluate.
        """
        matches = []
        for rule in self.rules:
            if not self._rule_applies(rule, filename, language):
                continue

            if rule.pattern:
                # Pattern-based rule — match against diff
                rule_matches = self._match_pattern(rule, filename, patch)
                matches.extend(rule_matches)
            elif rule.condition:
                # Condition-based rule — pass to LLM as context
                matches.append(RuleMatch(rule=rule, file=filename))

        return matches

    def get_auto_assignments(self, files: list[dict]) -> tuple[list[str], list[str]]:
        """Get all auto-assign reviewers and labels for the changed files.

        Returns (reviewers, labels).
        """
        reviewers = set()
        labels = set()

        for f in files:
            filename = f.get("filename", "")
            language = f.get("language", "unknown")
            for rule in self.rules:
                if self._rule_applies(rule, filename, language):
                    reviewers.update(rule.auto_assign)
                    labels.update(rule.auto_label)

        return list(reviewers), list(labels)

    def _rule_applies(self, rule: Rule, filename: str, language: str) -> bool:
        """Check if a rule applies to the given file."""
        # Language filter
        if rule.languages and language not in rule.languages:
            return False

        # Exclusion check
        for pattern in rule.exclude:
            if self._match_glob(filename, pattern):
                return False

        # Path inclusion check (empty = applies to all)
        if rule.paths:
            return any(self._match_glob(filename, p) for p in rule.paths)

        return True

    @staticmethod
    def _match_glob(filename: str, pattern: str) -> bool:
        """Match a filename against a glob pattern, supporting ** for any depth."""
        # fnmatch doesn't handle ** well, so we check multiple ways
        if fnmatch(filename, pattern):
            return True
        # For patterns like "src/**/*.js", also try matching without the ** segment
        # e.g., "src/**/*.js" should match "src/app.js" and "src/sub/app.js"
        if "**" in pattern:
            # Replace **/ with nothing to match files directly under the prefix
            collapsed = pattern.replace("**/", "")
            if fnmatch(filename, collapsed):
                return True
            # Also try PurePath.match for recursive patterns
            try:
                if PurePath(filename).match(pattern):
                    return True
            except ValueError:
                pass
        return False

    def _match_pattern(self, rule: Rule, filename: str, patch: str) -> list[RuleMatch]:
        """Match a regex pattern against added lines in the diff."""
        matches = []
        line_num = 0

        for line in patch.splitlines():
            # Track line numbers from hunk headers
            hunk_match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)", line)
            if hunk_match:
                line_num = int(hunk_match.group(1)) - 1
                continue

            if line.startswith("+") and not line.startswith("+++"):
                line_num += 1
                content = line[1:]  # Strip the leading +
                if re.search(rule.pattern, content):
                    matches.append(RuleMatch(
                        rule=rule,
                        file=filename,
                        line=line_num,
                        matched_text=content.strip(),
                    ))
            elif not line.startswith("-"):
                line_num += 1

        return matches
