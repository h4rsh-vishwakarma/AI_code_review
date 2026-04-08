"""Tests for AST analyzer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ast_analyzer import PythonASTAnalyzer, parse_diff_to_old_new


class TestPythonASTAnalyzer:
    def setup_method(self):
        self.analyzer = PythonASTAnalyzer()

    def test_detect_added_function(self):
        old = ""
        new = """
def greet(name: str) -> str:
    return f"Hello, {name}"
"""
        diff = self.analyzer.analyze(old, new, "utils.py")
        assert len(diff.functions) == 1
        assert diff.functions[0].name == "greet"
        assert diff.functions[0].change_type == "added"
        assert diff.has_meaningful_changes is True

    def test_detect_removed_function(self):
        old = """
def old_func():
    pass
"""
        new = ""
        diff = self.analyzer.analyze(old, new, "utils.py")
        assert len(diff.functions) == 1
        assert diff.functions[0].change_type == "removed"

    def test_detect_modified_function(self):
        old = """
def compute(x):
    return x + 1
"""
        new = """
def compute(x):
    return x * 2
"""
        diff = self.analyzer.analyze(old, new, "utils.py")
        modified = [f for f in diff.functions if f.change_type == "modified"]
        assert len(modified) == 1
        assert modified[0].name == "compute"

    def test_detect_signature_change(self):
        old = """
def process(data):
    return data
"""
        new = """
def process(data, verbose=False):
    return data
"""
        diff = self.analyzer.analyze(old, new, "utils.py")
        sig_changes = [f for f in diff.functions if f.change_type == "signature_changed"]
        assert len(sig_changes) == 1

    def test_detect_added_class(self):
        old = ""
        new = """
class UserService:
    def get_user(self, user_id):
        pass
    def create_user(self, data):
        pass
"""
        diff = self.analyzer.analyze(old, new, "service.py")
        assert len(diff.classes) == 1
        assert diff.classes[0].name == "UserService"
        assert diff.classes[0].change_type == "added"
        assert len(diff.classes[0].changed_methods) == 2

    def test_detect_import_changes(self):
        old = """
import os
from pathlib import Path
"""
        new = """
import os
import sys
from pathlib import Path
from typing import Optional
"""
        diff = self.analyzer.analyze(old, new, "main.py")
        assert "sys" in diff.imports.added
        assert "typing.Optional" in diff.imports.added

    def test_cosmetic_only_changes(self):
        # Same AST, just whitespace changes
        old = """
def func():
    x = 1
    return x
"""
        new = """
def func():
    x = 1
    return x
"""
        diff = self.analyzer.analyze(old, new, "test.py")
        assert diff.has_meaningful_changes is False

    def test_syntax_error_graceful(self):
        old = "def broken(:"
        new = "def fixed():\n    pass"
        diff = self.analyzer.analyze(old, new, "bad.py")
        # Should not crash, should mark as meaningful
        assert diff.has_meaningful_changes is True

    def test_summary_generation(self):
        old = ""
        new = """
import json

def parse(data):
    return json.loads(data)

class Parser:
    def run(self):
        pass
"""
        diff = self.analyzer.analyze(old, new, "parser.py")
        assert "parse" in diff.summary
        assert "Parser" in diff.summary


class TestParseDiffToOldNew:
    def test_basic_patch(self):
        patch = """@@ -1,3 +1,4 @@
 existing line
-old line
+new line
+added line
 context"""
        old, new = parse_diff_to_old_new(patch)
        assert "old line" in old
        assert "new line" in new
        assert "added line" in new
        assert "added line" not in old

    def test_empty_patch(self):
        old, new = parse_diff_to_old_new("")
        assert old == ""
        assert new == ""
