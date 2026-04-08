"""Main orchestrator — wires all modules together and runs the full review pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import MAX_FILES_TO_REVIEW, detect_language
from github_client import GitHubClient
from reviewer import CodeReviewer
from risk_scorer import RiskScorer
from rule_engine import RuleEngine
from feedback import FeedbackCollector
from parsers import Severity, format_inline_comment, ReviewComment
from ast_analyzer import get_ast_analyzer, parse_diff_to_old_new

# Optional: RAG engine (requires chromadb)
try:
    from rag_engine import RAGEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Code Review Bot")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--pr", type=int, default=int(os.environ.get("PR_NUMBER", "0")))
    parser.add_argument("--config", default=".reviewbot.yml", help="Path to custom rules config")
    parser.add_argument("--enable-rag", action="store_true", help="Enable RAG context retrieval")
    parser.add_argument("--enable-ast", action="store_true", help="Enable AST semantic analysis")
    parser.add_argument("--dry-run", action="store_true", help="Print results without posting to GitHub")
    parser.add_argument("--min-severity", default="warning", choices=["critical", "warning", "suggestion"])
    parser.add_argument(
        "--team-config",
        default=os.environ.get("TEAM_CONFIG", ""),
        help="JSON string mapping risk areas to reviewer lists",
    )
    return parser.parse_args()


def run_ast_analysis(files: list[dict], gh: GitHubClient) -> dict[str, dict]:
    """Run AST analysis on supported files and return semantic diffs.

    Returns dict of {filename: {"summary": str, "meaningful": bool}}.
    """
    results = {}
    for f in files:
        language = f.get("language", "unknown")
        analyzer = get_ast_analyzer(language)
        if analyzer is None:
            continue

        old_content, new_content = parse_diff_to_old_new(f["patch"])
        try:
            semantic_diff = analyzer.analyze(old_content, new_content, f["filename"])
            results[f["filename"]] = {
                "summary": semantic_diff.summary,
                "meaningful": semantic_diff.has_meaningful_changes,
            }
            logger.info("AST analysis for %s: %s", f["filename"], semantic_diff.summary)
        except Exception as e:
            logger.warning("AST analysis failed for %s: %s", f["filename"], e)

    return results


def build_summary_markdown(
    summary: dict,
    risk_assessment,
    findings: list[dict],
    rule_matches: list,
    ast_results: dict,
    files_count: int,
    posted_count: int,
) -> str:
    """Build the final markdown summary comment."""
    verdict = summary.get("verdict", "COMMENT")
    verdict_emoji = {
        "APPROVE": "\u2705",
        "REQUEST_CHANGES": "\u274c",
        "COMMENT": "\U0001f4ac",
    }.get(verdict, "\U0001f4ac")

    parts = [
        f"## \U0001f916 AI Code Review — {verdict_emoji} {verdict}",
        "",
        summary.get("summary", "Review complete."),
        "",
    ]

    # Risk score section
    if risk_assessment:
        risk_bar = "\u2588" * (risk_assessment.score // 10) + "\u2591" * (10 - risk_assessment.score // 10)
        parts.extend([
            "### Risk Assessment",
            f"**Score:** {risk_assessment.score}/100 ({risk_assessment.level}) `{risk_bar}`",
            "",
        ])
        if risk_assessment.recommendations:
            for rec in risk_assessment.recommendations:
                parts.append(f"- {rec}")
            parts.append("")

    # Stats
    critical = sum(1 for f in findings if f.get("severity") == "critical")
    warnings = sum(1 for f in findings if f.get("severity") == "warning")
    suggestions = sum(1 for f in findings if f.get("severity") == "suggestion")

    parts.extend([
        "### Summary",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Files reviewed | {files_count} |",
        f"| Issues found | {len(findings)} |",
        f"| \U0001f534 Critical | {critical} |",
        f"| \U0001f7e1 Warnings | {warnings} |",
        f"| \U0001f535 Suggestions | {suggestions} |",
        f"| Inline comments posted | {posted_count} |",
    ])

    if rule_matches:
        parts.append(f"| Custom rule violations | {len(rule_matches)} |")

    parts.append("")

    # AST analysis section
    if ast_results:
        meaningful = [f for f, r in ast_results.items() if r["meaningful"]]
        cosmetic = [f for f, r in ast_results.items() if not r["meaningful"]]
        if cosmetic:
            parts.extend([
                "### AST Analysis",
                f"Files with only cosmetic changes (skipped deep review): {', '.join(f'`{f}`' for f in cosmetic)}",
                "",
            ])

    # Custom rule matches
    if rule_matches:
        parts.extend([
            "### Custom Rule Violations",
            "",
        ])
        for match in rule_matches[:10]:
            parts.append(f"- **{match.rule.name}** in `{match.file}` line {match.line}: {match.rule.message}")
        parts.append("")

    parts.extend([
        "---",
        "*\U0001f916 Powered by AI Code Review Bot | "
        "React with \U0001f44d/\U0001f44e on inline comments to help improve future reviews*",
    ])

    return "\n".join(parts)


def main() -> None:
    args = parse_args()

    if not args.repo or not args.pr:
        logger.error("Missing --repo or --pr (or GITHUB_REPOSITORY / PR_NUMBER env vars)")
        sys.exit(1)

    logger.info("Starting review: %s #%d", args.repo, args.pr)

    # --- Initialize modules ---
    gh = GitHubClient(args.repo, args.pr)
    pr_context = gh.get_pr_context()

    # Rule engine
    rule_engine = RuleEngine(args.config)

    # RAG engine (optional)
    rag_engine = None
    if args.enable_rag and RAG_AVAILABLE:
        rag_engine = RAGEngine()
        logger.info("RAG context retrieval enabled")

    # Reviewer
    reviewer = CodeReviewer(rag_engine=rag_engine, rule_engine=rule_engine)

    # Test API connectivity
    print(f"[DEBUG] Model: {os.environ.get('MODEL_NAME', 'gpt-4o (default)')}")
    print(f"[DEBUG] API key present: {bool(os.environ.get('OPENAI_API_KEY'))}")
    print(f"[DEBUG] API key prefix: {os.environ.get('OPENAI_API_KEY', '')[:8]}...")
    reviewer.test_api_connection()

    # Risk scorer
    team_config = {}
    if args.team_config:
        try:
            team_config = json.loads(args.team_config)
        except json.JSONDecodeError:
            logger.warning("Invalid TEAM_CONFIG JSON, ignoring")
    risk_scorer = RiskScorer(team_config=team_config)

    # Feedback collector
    feedback = FeedbackCollector()

    # --- Prepare files ---
    files = pr_context.files[:MAX_FILES_TO_REVIEW]
    if not files:
        logger.info("No reviewable files changed.")
        gh.post_summary_comment("## \U0001f916 AI Code Review\n\nNo reviewable files changed in this PR.")
        return

    file_dicts = [
        {"filename": f.filename, "patch": f.patch, "language": f.language,
         "additions": f.additions, "deletions": f.deletions}
        for f in files
    ]

    logger.info("Reviewing %d files...", len(files))

    # --- AST Analysis (optional) ---
    ast_results = {}
    if args.enable_ast:
        ast_results = run_ast_analysis(file_dicts, gh)
        # Filter out files with only cosmetic changes
        meaningful_files = [
            f for f in file_dicts
            if f["filename"] not in ast_results
            or ast_results[f["filename"]]["meaningful"]
        ]
        skipped = len(file_dicts) - len(meaningful_files)
        if skipped > 0:
            logger.info("AST analysis: skipping %d files with only cosmetic changes", skipped)
        file_dicts = meaningful_files

    # --- Custom Rule Evaluation ---
    rule_matches = []
    for f in file_dicts:
        matches = rule_engine.evaluate(f["filename"], f["language"], f["patch"])
        rule_matches.extend(matches)

    if rule_matches:
        logger.info("Custom rules: %d violations found", len(rule_matches))

    # --- LLM Review ---
    findings, summary = reviewer.review_all_files(
        files=file_dicts,
        pr_title=pr_context.title,
        pr_description=pr_context.description,
    )

    # --- Risk Assessment ---
    author_stats = gh.get_author_stats()
    risk_assessment = risk_scorer.assess(file_dicts, findings, author_stats)
    logger.info("Risk score: %d/100 (%s)", risk_assessment.score, risk_assessment.level)

    # --- Determine minimum severity to post ---
    severity_order = {"critical": 0, "warning": 1, "suggestion": 2}
    min_severity_val = severity_order.get(args.min_severity, 1)

    # --- Post Results ---
    if args.dry_run:
        print("\n=== DRY RUN — Results ===\n")
        print(f"Risk: {risk_assessment.score}/100 ({risk_assessment.level})")
        print(f"Findings: {len(findings)}")
        print(f"Verdict: {summary.get('verdict')}")
        for f in findings:
            print(f"  [{f.get('severity')}] {f.get('file')}:{f.get('line')} — {f.get('message')}")
        for m in rule_matches:
            print(f"  [rule:{m.rule.name}] {m.file}:{m.line} — {m.rule.message}")
        return

    # Post inline comments for LLM findings
    posted = 0
    review_comments = []
    for finding in findings:
        finding_severity = finding.get("severity", "suggestion")
        if severity_order.get(finding_severity, 2) > min_severity_val:
            continue

        try:
            comment = ReviewComment(
                file=finding["file"],
                line=finding["line"],
                severity=Severity(finding_severity),
                category=finding.get("category", "bug"),
                message=finding["message"],
                suggestion=finding.get("suggestion", ""),
                confidence=finding.get("confidence", 0.8),
            )
            review_comments.append({
                "path": comment.file,
                "line": comment.line,
                "body": format_inline_comment(comment),
            })
            posted += 1
        except Exception as e:
            logger.warning("Failed to format comment: %s", e)

    # Post inline comments for custom rule matches
    for match in rule_matches:
        if match.line > 0:
            body = (
                f"\U0001f7e0 **RULE: {match.rule.name}** | {match.rule.severity}\n\n"
                f"{match.rule.message}\n\n"
                f"Matched: `{match.matched_text}`" if match.matched_text else
                f"\U0001f7e0 **RULE: {match.rule.name}** | {match.rule.severity}\n\n"
                f"{match.rule.message}"
            )
            review_comments.append({
                "path": match.file,
                "line": match.line,
                "body": body,
            })

    # Post as a single review (more efficient than individual comments)
    verdict = summary.get("verdict", "COMMENT")
    event = verdict if verdict in ("APPROVE", "REQUEST_CHANGES") else "COMMENT"

    summary_markdown = build_summary_markdown(
        summary=summary,
        risk_assessment=risk_assessment,
        findings=findings,
        rule_matches=rule_matches,
        ast_results=ast_results,
        files_count=len(files),
        posted_count=posted,
    )

    if review_comments:
        success = gh.post_review(
            body=summary_markdown,
            comments=review_comments,
            event=event,
        )
        if not success:
            # Fallback: post comments individually
            logger.warning("Batch review failed, posting individually")
            for c in review_comments:
                gh.post_inline_comment(c["body"], c["path"], c["line"])
            gh.post_summary_comment(summary_markdown)
    else:
        gh.post_summary_comment(summary_markdown)

    # --- Auto-routing ---
    # Add risk labels
    if risk_assessment.suggested_labels:
        gh.add_labels(risk_assessment.suggested_labels)

    # Add rule-based labels
    _, rule_labels = rule_engine.get_auto_assignments(file_dicts)
    if rule_labels:
        gh.add_labels(rule_labels)

    # Request reviewers
    all_reviewers = set(risk_assessment.suggested_reviewers)
    rule_reviewers, _ = rule_engine.get_auto_assignments(file_dicts)
    all_reviewers.update(rule_reviewers)
    # Remove @ prefix if present
    all_reviewers = [r.lstrip("@") for r in all_reviewers if r.lstrip("@") != pr_context.author]
    if all_reviewers:
        gh.request_reviewers(all_reviewers)

    logger.info(
        "Review complete: %d findings, %d inline comments, verdict=%s, risk=%d",
        len(findings), posted, verdict, risk_assessment.score,
    )

    # Log feedback stats
    stats = feedback.get_stats()
    if stats["total"] > 0:
        logger.info(
            "Feedback stats: %d total, %.0f%% precision",
            stats["total"], stats["precision"] * 100,
        )


if __name__ == "__main__":
    main()
