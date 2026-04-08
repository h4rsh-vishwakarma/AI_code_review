"""AST-based semantic diff analyzer.

Parses code into Abstract Syntax Trees to understand what actually changed
semantically, filtering out cosmetic changes like whitespace and formatting.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FunctionChange:
    """Represents a changed function/method."""
    name: str
    change_type: str  # added, modified, removed, signature_changed
    old_signature: str = ""
    new_signature: str = ""
    body_changed: bool = False


@dataclass
class ClassChange:
    """Represents a changed class."""
    name: str
    change_type: str  # added, modified, removed
    changed_methods: list[FunctionChange] = field(default_factory=list)
    new_bases: list[str] = field(default_factory=list)


@dataclass
class ImportChange:
    """Represents changed imports."""
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


@dataclass
class SemanticDiff:
    """High-level semantic diff of a file."""
    filename: str
    language: str
    functions: list[FunctionChange] = field(default_factory=list)
    classes: list[ClassChange] = field(default_factory=list)
    imports: ImportChange = field(default_factory=ImportChange)
    has_meaningful_changes: bool = True
    summary: str = ""


class PythonASTAnalyzer:
    """AST analyzer for Python files."""

    def analyze(self, old_content: str, new_content: str, filename: str) -> SemanticDiff:
        """Compare two versions of a Python file semantically."""
        diff = SemanticDiff(filename=filename, language="python")

        try:
            old_tree = ast.parse(old_content) if old_content else ast.Module(body=[], type_ignores=[])
            new_tree = ast.parse(new_content) if new_content else ast.Module(body=[], type_ignores=[])
        except SyntaxError as e:
            logger.warning("Failed to parse %s: %s", filename, e)
            diff.has_meaningful_changes = True
            diff.summary = f"Could not parse AST: {e}"
            return diff

        # Analyze imports
        diff.imports = self._compare_imports(old_tree, new_tree)

        # Analyze functions
        old_funcs = self._extract_functions(old_tree)
        new_funcs = self._extract_functions(new_tree)
        diff.functions = self._compare_functions(old_funcs, new_funcs)

        # Analyze classes
        old_classes = self._extract_classes(old_tree)
        new_classes = self._extract_classes(new_tree)
        diff.classes = self._compare_classes(old_classes, new_classes)

        # Determine if changes are meaningful
        diff.has_meaningful_changes = self._has_meaningful_changes(diff)
        diff.summary = self._generate_summary(diff)

        return diff

    def _extract_functions(self, tree: ast.Module) -> dict[str, ast.FunctionDef]:
        """Extract top-level and method function definitions."""
        funcs = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs[node.name] = node
        return funcs

    def _extract_classes(self, tree: ast.Module) -> dict[str, ast.ClassDef]:
        """Extract class definitions."""
        classes = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
        return classes

    def _get_function_signature(self, func: ast.FunctionDef) -> str:
        """Extract function signature as string."""
        args = func.args
        parts = []

        # Regular args
        for arg in args.args:
            annotation = ast.dump(arg.annotation) if arg.annotation else ""
            parts.append(f"{arg.arg}: {annotation}" if annotation else arg.arg)

        # *args
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")

        # **kwargs
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        returns = ""
        if func.returns:
            returns = f" -> {ast.dump(func.returns)}"

        return f"({', '.join(parts)}){returns}"

    def _compare_functions(
        self,
        old_funcs: dict[str, ast.FunctionDef],
        new_funcs: dict[str, ast.FunctionDef],
    ) -> list[FunctionChange]:
        """Compare function definitions between old and new versions."""
        changes = []

        # Added functions
        for name in new_funcs:
            if name not in old_funcs:
                changes.append(FunctionChange(
                    name=name,
                    change_type="added",
                    new_signature=self._get_function_signature(new_funcs[name]),
                ))

        # Removed functions
        for name in old_funcs:
            if name not in new_funcs:
                changes.append(FunctionChange(
                    name=name,
                    change_type="removed",
                    old_signature=self._get_function_signature(old_funcs[name]),
                ))

        # Modified functions
        for name in old_funcs:
            if name in new_funcs:
                old_sig = self._get_function_signature(old_funcs[name])
                new_sig = self._get_function_signature(new_funcs[name])
                old_body = ast.dump(old_funcs[name])
                new_body = ast.dump(new_funcs[name])

                if old_sig != new_sig:
                    changes.append(FunctionChange(
                        name=name,
                        change_type="signature_changed",
                        old_signature=old_sig,
                        new_signature=new_sig,
                        body_changed=old_body != new_body,
                    ))
                elif old_body != new_body:
                    changes.append(FunctionChange(
                        name=name,
                        change_type="modified",
                        old_signature=old_sig,
                        new_signature=new_sig,
                        body_changed=True,
                    ))

        return changes

    def _compare_classes(
        self,
        old_classes: dict[str, ast.ClassDef],
        new_classes: dict[str, ast.ClassDef],
    ) -> list[ClassChange]:
        """Compare class definitions."""
        changes = []

        for name in new_classes:
            if name not in old_classes:
                methods = [
                    FunctionChange(name=n.name, change_type="added")
                    for n in ast.iter_child_nodes(new_classes[name])
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                changes.append(ClassChange(
                    name=name,
                    change_type="added",
                    changed_methods=methods,
                ))

        for name in old_classes:
            if name not in new_classes:
                changes.append(ClassChange(name=name, change_type="removed"))

        for name in old_classes:
            if name in new_classes:
                old_dump = ast.dump(old_classes[name])
                new_dump = ast.dump(new_classes[name])
                if old_dump != new_dump:
                    old_methods = self._extract_functions(old_classes[name])
                    new_methods = self._extract_functions(new_classes[name])
                    method_changes = self._compare_functions(old_methods, new_methods)
                    if method_changes:
                        changes.append(ClassChange(
                            name=name,
                            change_type="modified",
                            changed_methods=method_changes,
                        ))

        return changes

    def _compare_imports(self, old_tree: ast.Module, new_tree: ast.Module) -> ImportChange:
        """Compare import statements."""
        old_imports = self._extract_imports(old_tree)
        new_imports = self._extract_imports(new_tree)
        return ImportChange(
            added=list(new_imports - old_imports),
            removed=list(old_imports - new_imports),
        )

    def _extract_imports(self, tree: ast.Module) -> set[str]:
        """Extract all import names."""
        imports = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.add(f"{module}.{alias.name}")
        return imports

    def _has_meaningful_changes(self, diff: SemanticDiff) -> bool:
        """Determine if the diff contains meaningful (non-cosmetic) changes."""
        if diff.functions or diff.classes:
            return True
        if diff.imports.added or diff.imports.removed:
            return True
        return False

    def _generate_summary(self, diff: SemanticDiff) -> str:
        """Generate a human-readable summary of semantic changes."""
        parts = []

        added_funcs = [f for f in diff.functions if f.change_type == "added"]
        modified_funcs = [f for f in diff.functions if f.change_type in ("modified", "signature_changed")]
        removed_funcs = [f for f in diff.functions if f.change_type == "removed"]

        if added_funcs:
            names = ", ".join(f.name for f in added_funcs)
            parts.append(f"Added functions: {names}")
        if modified_funcs:
            names = ", ".join(f.name for f in modified_funcs)
            parts.append(f"Modified functions: {names}")
        if removed_funcs:
            names = ", ".join(f.name for f in removed_funcs)
            parts.append(f"Removed functions: {names}")

        sig_changes = [f for f in diff.functions if f.change_type == "signature_changed"]
        if sig_changes:
            for f in sig_changes:
                parts.append(f"Signature change: {f.name}{f.old_signature} → {f.name}{f.new_signature}")

        added_classes = [c for c in diff.classes if c.change_type == "added"]
        if added_classes:
            names = ", ".join(c.name for c in added_classes)
            parts.append(f"Added classes: {names}")

        if diff.imports.added:
            parts.append(f"New imports: {', '.join(diff.imports.added)}")
        if diff.imports.removed:
            parts.append(f"Removed imports: {', '.join(diff.imports.removed)}")

        return "; ".join(parts) if parts else "No meaningful semantic changes detected"


def parse_diff_to_old_new(patch: str) -> tuple[str, str]:
    """Extract old and new file content from a unified diff patch.

    This is a best-effort reconstruction — unified diffs don't contain full files.
    Returns (old_lines_joined, new_lines_joined) from the patch hunks.
    """
    old_lines = []
    new_lines = []

    for line in patch.splitlines():
        if line.startswith("@@"):
            continue
        elif line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        else:
            # Context line (present in both)
            stripped = line[1:] if line.startswith(" ") else line
            old_lines.append(stripped)
            new_lines.append(stripped)

    return "\n".join(old_lines), "\n".join(new_lines)


def get_ast_analyzer(language: str) -> PythonASTAnalyzer | None:
    """Get the appropriate AST analyzer for a language.

    Currently supports Python. Other languages can be added with tree-sitter.
    """
    if language == "python":
        return PythonASTAnalyzer()
    return None
