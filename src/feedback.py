"""Feedback loop system — captures developer reactions and builds training data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict

from config import FEEDBACK_FILE

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """A single feedback record from developer reaction."""
    timestamp: str
    repo: str
    pr_number: int
    file: str
    line: int
    severity: str
    category: str
    bot_message: str
    bot_suggestion: str
    developer_verdict: str  # helpful, unhelpful, false_positive
    developer_comment: str = ""
    model_version: str = ""
    diff_snippet: str = ""


class FeedbackCollector:
    """Collects and stores developer feedback for model improvement.

    Feedback sources:
    - Emoji reactions on bot comments (thumbs up/down)
    - Direct replies to bot comments
    - "Resolved" vs "Won't fix" thread closures

    Storage: JSONL file for simplicity. Can be swapped for a database.
    """

    def __init__(self, storage_path: str | None = None):
        self.storage_path = Path(storage_path or FEEDBACK_FILE)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: FeedbackEntry) -> None:
        """Append a feedback entry to storage."""
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
            logger.info(
                "Recorded feedback: %s for %s:%d (%s)",
                entry.developer_verdict, entry.file, entry.line, entry.category,
            )
        except Exception as e:
            logger.error("Failed to record feedback: %s", e)

    def record_from_reaction(
        self,
        repo: str,
        pr_number: int,
        comment_body: str,
        reaction: str,
        diff_snippet: str = "",
    ) -> None:
        """Record feedback from a GitHub reaction on a bot comment.

        Reaction mapping:
            +1, heart, rocket  → helpful
            -1, confused       → unhelpful
        """
        verdict_map = {
            "+1": "helpful",
            "heart": "helpful",
            "rocket": "helpful",
            "-1": "unhelpful",
            "confused": "unhelpful",
        }
        verdict = verdict_map.get(reaction, "unknown")
        if verdict == "unknown":
            return

        # Parse the bot comment to extract structured data
        parsed = self._parse_bot_comment(comment_body)

        entry = FeedbackEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            repo=repo,
            pr_number=pr_number,
            file=parsed.get("file", ""),
            line=parsed.get("line", 0),
            severity=parsed.get("severity", ""),
            category=parsed.get("category", ""),
            bot_message=parsed.get("message", ""),
            bot_suggestion=parsed.get("suggestion", ""),
            developer_verdict=verdict,
            diff_snippet=diff_snippet,
        )
        self.record(entry)

    def record_from_reply(
        self,
        repo: str,
        pr_number: int,
        comment_body: str,
        reply_text: str,
        diff_snippet: str = "",
    ) -> None:
        """Record feedback from a developer's reply to a bot comment."""
        parsed = self._parse_bot_comment(comment_body)

        # Infer verdict from reply text
        verdict = self._infer_verdict(reply_text)

        entry = FeedbackEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            repo=repo,
            pr_number=pr_number,
            file=parsed.get("file", ""),
            line=parsed.get("line", 0),
            severity=parsed.get("severity", ""),
            category=parsed.get("category", ""),
            bot_message=parsed.get("message", ""),
            bot_suggestion=parsed.get("suggestion", ""),
            developer_verdict=verdict,
            developer_comment=reply_text,
            diff_snippet=diff_snippet,
        )
        self.record(entry)

    def get_stats(self) -> dict:
        """Get aggregate feedback statistics."""
        stats = {
            "total": 0,
            "helpful": 0,
            "unhelpful": 0,
            "false_positive": 0,
            "precision": 0.0,
            "by_category": {},
        }

        if not self.storage_path.exists():
            return stats

        try:
            with open(self.storage_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    stats["total"] += 1
                    verdict = entry.get("developer_verdict", "")
                    if verdict in stats:
                        stats[verdict] += 1

                    category = entry.get("category", "other")
                    if category not in stats["by_category"]:
                        stats["by_category"][category] = {"helpful": 0, "unhelpful": 0}
                    if verdict in ("helpful", "unhelpful"):
                        stats["by_category"][category][verdict] += 1

            if stats["helpful"] + stats["unhelpful"] > 0:
                stats["precision"] = stats["helpful"] / (stats["helpful"] + stats["unhelpful"])

        except Exception as e:
            logger.error("Failed to read feedback stats: %s", e)

        return stats

    def export_training_data(self, output_path: str) -> int:
        """Export feedback as OpenAI fine-tuning JSONL format.

        Only exports entries marked as 'helpful' (positive examples)
        and 'unhelpful'/'false_positive' (negative examples for calibration).

        Returns number of training examples exported.
        """
        if not self.storage_path.exists():
            return 0

        count = 0
        with open(self.storage_path, encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                entry = json.loads(line)
                verdict = entry.get("developer_verdict", "")

                if verdict == "helpful":
                    # Positive training example
                    training_example = {
                        "messages": [
                            {"role": "system", "content": "You are an expert code reviewer."},
                            {"role": "user", "content": f"Review this diff:\n```\n{entry.get('diff_snippet', '')}\n```"},
                            {"role": "assistant", "content": json.dumps([{
                                "file": entry["file"],
                                "line": entry["line"],
                                "severity": entry["severity"],
                                "category": entry["category"],
                                "message": entry["bot_message"],
                                "suggestion": entry["bot_suggestion"],
                            }])},
                        ]
                    }
                    f_out.write(json.dumps(training_example) + "\n")
                    count += 1
                elif verdict in ("unhelpful", "false_positive"):
                    # Negative example — teach model to return empty
                    training_example = {
                        "messages": [
                            {"role": "system", "content": "You are an expert code reviewer."},
                            {"role": "user", "content": f"Review this diff:\n```\n{entry.get('diff_snippet', '')}\n```"},
                            {"role": "assistant", "content": "[]"},
                        ]
                    }
                    f_out.write(json.dumps(training_example) + "\n")
                    count += 1

        logger.info("Exported %d training examples to %s", count, output_path)
        return count

    def _parse_bot_comment(self, body: str) -> dict:
        """Parse structured data from a bot review comment."""
        result = {"file": "", "line": 0, "severity": "", "category": "", "message": "", "suggestion": ""}

        lines = body.split("\n")
        for line in lines:
            if "**CRITICAL**" in line:
                result["severity"] = "critical"
            elif "**WARNING**" in line:
                result["severity"] = "warning"
            elif "**SUGGESTION**" in line:
                result["severity"] = "suggestion"

            if "| " in line and any(cat in line.lower() for cat in ["bug", "security", "performance", "style"]):
                for cat in ["bug", "security", "performance", "style", "best_practice"]:
                    if cat in line.lower():
                        result["category"] = cat
                        break

            if line.startswith("**Suggestion:**"):
                result["suggestion"] = line.replace("**Suggestion:**", "").strip()

        # The message is typically the non-header, non-suggestion lines
        content_lines = [
            l for l in lines
            if l.strip()
            and not l.startswith("**")
            and "CRITICAL" not in l
            and "WARNING" not in l
            and "SUGGESTION" not in l
            and "Confidence" not in l
        ]
        result["message"] = " ".join(content_lines).strip()

        return result

    def _infer_verdict(self, reply_text: str) -> str:
        """Infer developer verdict from their reply text."""
        lower = reply_text.lower()

        positive_signals = ["good catch", "thanks", "fixed", "will fix", "agreed", "nice find"]
        negative_signals = ["false positive", "not an issue", "intentional", "by design", "won't fix", "incorrect"]

        for signal in positive_signals:
            if signal in lower:
                return "helpful"
        for signal in negative_signals:
            if signal in lower:
                return "false_positive"

        return "unknown"
