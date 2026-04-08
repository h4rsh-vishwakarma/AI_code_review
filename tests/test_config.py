"""Tests for config module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import detect_language, should_skip_file


class TestDetectLanguage:
    def test_python(self):
        assert detect_language("src/main.py") == "python"

    def test_javascript(self):
        assert detect_language("app.js") == "javascript"

    def test_typescript(self):
        assert detect_language("component.tsx") == "typescript"

    def test_go(self):
        assert detect_language("main.go") == "go"

    def test_dockerfile(self):
        assert detect_language("Dockerfile") == "docker"

    def test_unknown(self):
        assert detect_language("data.csv") == "unknown"

    def test_sql(self):
        assert detect_language("migration.sql") == "sql"

    def test_terraform(self):
        assert detect_language("main.tf") == "terraform"


class TestShouldSkipFile:
    def test_skip_lockfile(self):
        assert should_skip_file("package-lock.json") is True

    def test_skip_image(self):
        assert should_skip_file("logo.png") is True

    def test_skip_vendor(self):
        assert should_skip_file("vendor/lib/module.js") is True

    def test_skip_node_modules(self):
        assert should_skip_file("node_modules/pkg/index.js") is True

    def test_allow_source(self):
        assert should_skip_file("src/main.py") is False

    def test_allow_test(self):
        assert should_skip_file("tests/test_main.py") is False

    def test_skip_minified(self):
        assert should_skip_file("bundle.min.js") is True
