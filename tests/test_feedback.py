"""Tests for feedback collection system."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feedback import FeedbackCollector, FeedbackEntry


class TestFeedbackCollector:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage_path = str(Path(self.tmpdir) / "feedback.jsonl")
        self.collector = FeedbackCollector(storage_path=self.storage_path)

    def test_record_entry(self):
        entry = FeedbackEntry(
            timestamp="2026-04-08T12:00:00Z",
            repo="user/repo",
            pr_number=42,
            file="src/auth.py",
            line=10,
            severity="critical",
            category="security",
            bot_message="SQL injection found",
            bot_suggestion="Use parameterized queries",
            developer_verdict="helpful",
        )
        self.collector.record(entry)

        with open(self.storage_path) as f:
            data = json.loads(f.readline())
        assert data["developer_verdict"] == "helpful"
        assert data["file"] == "src/auth.py"

    def test_record_from_reaction_positive(self):
        self.collector.record_from_reaction(
            repo="user/repo",
            pr_number=1,
            comment_body="**CRITICAL** | security\n\nSQL injection\n\n**Suggestion:** fix it",
            reaction="+1",
        )
        stats = self.collector.get_stats()
        assert stats["helpful"] == 1

    def test_record_from_reaction_negative(self):
        self.collector.record_from_reaction(
            repo="user/repo",
            pr_number=1,
            comment_body="**WARNING** | bug\n\nmessage\n\n**Suggestion:** fix",
            reaction="-1",
        )
        stats = self.collector.get_stats()
        assert stats["unhelpful"] == 1

    def test_record_from_reply_positive(self):
        self.collector.record_from_reply(
            repo="user/repo",
            pr_number=1,
            comment_body="bot comment body",
            reply_text="Good catch, will fix!",
        )
        stats = self.collector.get_stats()
        assert stats["helpful"] == 1

    def test_record_from_reply_negative(self):
        self.collector.record_from_reply(
            repo="user/repo",
            pr_number=1,
            comment_body="bot comment body",
            reply_text="This is a false positive, the code is intentional",
        )
        stats = self.collector.get_stats()
        assert stats["false_positive"] == 1

    def test_get_stats_empty(self):
        stats = self.collector.get_stats()
        assert stats["total"] == 0
        assert stats["precision"] == 0.0

    def test_precision_calculation(self):
        for _ in range(3):
            self.collector.record(FeedbackEntry(
                timestamp="", repo="", pr_number=0, file="", line=0,
                severity="", category="", bot_message="", bot_suggestion="",
                developer_verdict="helpful",
            ))
        self.collector.record(FeedbackEntry(
            timestamp="", repo="", pr_number=0, file="", line=0,
            severity="", category="", bot_message="", bot_suggestion="",
            developer_verdict="unhelpful",
        ))
        stats = self.collector.get_stats()
        assert stats["precision"] == 0.75

    def test_export_training_data(self):
        self.collector.record(FeedbackEntry(
            timestamp="", repo="", pr_number=0, file="auth.py", line=10,
            severity="critical", category="security",
            bot_message="SQL injection", bot_suggestion="Use params",
            developer_verdict="helpful", diff_snippet="+query = f'SELECT {x}'",
        ))
        self.collector.record(FeedbackEntry(
            timestamp="", repo="", pr_number=0, file="utils.py", line=5,
            severity="warning", category="style",
            bot_message="Bad name", bot_suggestion="Rename",
            developer_verdict="unhelpful", diff_snippet="+x = 1",
        ))

        output_path = str(Path(self.tmpdir) / "training.jsonl")
        count = self.collector.export_training_data(output_path)
        assert count == 2

        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        # Helpful example should have real content
        example1 = json.loads(lines[0])
        assert len(example1["messages"]) == 3
        # Unhelpful example should return empty
        example2 = json.loads(lines[1])
        assert example2["messages"][-1]["content"] == "[]"
