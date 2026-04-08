"""Tests for custom rule engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rule_engine import RuleEngine


SAMPLE_CONFIG = """
rules:
  - name: no-console-log
    pattern: "console\\\\.log\\\\("
    severity: warning
    message: "Remove console.log before merging"
    paths: ["src/**/*.js", "src/**/*.ts"]
    exclude: ["src/debug/**"]

  - name: no-eval
    pattern: "eval\\\\("
    severity: critical
    message: "Never use eval() — it enables code injection"
    languages: ["javascript", "typescript", "python"]

  - name: migration-review
    paths: ["db/migrations/**"]
    severity: critical
    message: "Database migrations require manual review"
    auto_assign: ["@dba-lead"]
    auto_label: ["database-review"]

  - name: no-any-type
    pattern: ":\\\\s*any[\\\\s;,)]"
    severity: warning
    message: "Avoid 'any' type"
    languages: ["typescript"]
"""


class TestRuleEngine:
    def setup_method(self):
        self.engine = RuleEngine()
        self.engine.load_from_string(SAMPLE_CONFIG)

    def test_loads_rules(self):
        assert len(self.engine.rules) == 4

    def test_rules_for_js_file(self):
        rules = self.engine.get_rules_for_file("src/app.js", "javascript")
        names = [r["name"] for r in rules]
        assert "no-console-log" in names
        assert "no-eval" in names

    def test_exclude_pattern(self):
        rules = self.engine.get_rules_for_file("src/debug/logger.js", "javascript")
        names = [r["name"] for r in rules]
        assert "no-console-log" not in names

    def test_language_filter(self):
        rules = self.engine.get_rules_for_file("src/main.go", "go")
        names = [r["name"] for r in rules]
        assert "no-any-type" not in names
        assert "no-eval" not in names

    def test_migration_rule_matches(self):
        rules = self.engine.get_rules_for_file("db/migrations/001_add_users.sql", "sql")
        names = [r["name"] for r in rules]
        assert "migration-review" in names

    def test_pattern_evaluation(self):
        patch = """@@ -1,3 +1,5 @@
+console.log("debug info");
+const x = 42;
 const y = 10;"""
        matches = self.engine.evaluate("src/app.js", "javascript", patch)
        matched_rules = [m.rule.name for m in matches]
        assert "no-console-log" in matched_rules

    def test_eval_pattern(self):
        patch = """@@ -1,2 +1,3 @@
+result = eval(user_input)
 x = 1"""
        matches = self.engine.evaluate("src/handler.py", "python", patch)
        matched_rules = [m.rule.name for m in matches]
        assert "no-eval" in matched_rules

    def test_no_match_on_context_lines(self):
        # console.log on a context line (no +) should not match
        patch = """@@ -1,3 +1,4 @@
 console.log("existing");
+const newVar = true;
 const y = 10;"""
        matches = self.engine.evaluate("src/app.js", "javascript", patch)
        matched_rules = [m.rule.name for m in matches]
        assert "no-console-log" not in matched_rules

    def test_auto_assignments(self):
        files = [
            {"filename": "db/migrations/002.sql", "language": "sql"},
            {"filename": "src/app.js", "language": "javascript"},
        ]
        reviewers, labels = self.engine.get_auto_assignments(files)
        assert "@dba-lead" in reviewers
        assert "database-review" in labels
