"""GitHub API client for fetching PR data and posting review comments."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from github import Github, GithubException
from github.PullRequest import PullRequest

from config import GITHUB_TOKEN, should_skip_file

logger = logging.getLogger(__name__)


@dataclass
class ChangedFile:
    """Represents a file changed in a pull request."""
    filename: str
    patch: str
    status: str  # added, modified, removed, renamed
    additions: int
    deletions: int
    language: str = "unknown"


@dataclass
class PRContext:
    """Full context of a pull request."""
    title: str
    description: str
    author: str
    base_branch: str
    head_branch: str
    number: int
    files: list[ChangedFile]
    labels: list[str]


class GitHubClient:
    """Handles all interactions with the GitHub API."""

    def __init__(self, repo_name: str, pr_number: int):
        self.gh = Github(GITHUB_TOKEN)
        self.repo = self.gh.get_repo(repo_name)
        self.pr: PullRequest = self.repo.get_pull(pr_number)
        self.repo_name = repo_name
        self.pr_number = pr_number

    def get_pr_context(self) -> PRContext:
        """Fetch full PR context including metadata and changed files."""
        files = self._get_changed_files()
        return PRContext(
            title=self.pr.title,
            description=self.pr.body or "",
            author=self.pr.user.login,
            base_branch=self.pr.base.ref,
            head_branch=self.pr.head.ref,
            number=self.pr.number,
            files=files,
            labels=[label.name for label in self.pr.get_labels()],
        )

    def _get_changed_files(self) -> list[ChangedFile]:
        """Fetch list of changed files with their diffs."""
        from config import detect_language

        files = []
        for f in self.pr.get_files():
            if not f.patch:
                continue
            if should_skip_file(f.filename):
                logger.info("Skipping file: %s", f.filename)
                continue
            files.append(ChangedFile(
                filename=f.filename,
                patch=f.patch,
                status=f.status,
                additions=f.additions,
                deletions=f.deletions,
                language=detect_language(f.filename),
            ))
        return files

    def get_file_content(self, path: str, ref: str | None = None) -> str | None:
        """Fetch full file content from the repo (for RAG context)."""
        try:
            ref = ref or self.pr.head.ref
            content = self.repo.get_contents(path, ref=ref)
            if isinstance(content, list):
                return None
            return content.decoded_content.decode("utf-8")
        except GithubException:
            return None

    def post_inline_comment(self, body: str, path: str, line: int) -> bool:
        """Post an inline review comment on a specific line."""
        try:
            latest_commit = list(self.pr.get_commits())[-1]
            self.pr.create_review_comment(
                body=body,
                commit=latest_commit,
                path=path,
                line=line,
                side="RIGHT",
            )
            return True
        except GithubException as e:
            logger.warning("Failed to post inline comment on %s:%d — %s", path, line, e)
            return False

    def post_review(self, body: str, comments: list[dict], event: str = "COMMENT") -> bool:
        """Post a full pull request review with inline comments.

        Args:
            body: Summary comment for the review.
            comments: List of dicts with keys: path, line, body.
            event: One of COMMENT, APPROVE, REQUEST_CHANGES.
        """
        try:
            latest_commit = list(self.pr.get_commits())[-1]
            review_comments = []
            for c in comments:
                review_comments.append({
                    "path": c["path"],
                    "line": c["line"],
                    "body": c["body"],
                    "side": "RIGHT",
                })
            self.pr.create_review(
                commit=latest_commit,
                body=body,
                event=event,
                comments=review_comments,
            )
            return True
        except GithubException as e:
            logger.error("Failed to post review: %s", e)
            return False
        except Exception as e:
            logger.error("Failed to post review (unexpected): %s", e)
            return False

    def post_summary_comment(self, body: str) -> bool:
        """Post a general comment on the PR."""
        try:
            self.pr.create_issue_comment(body)
            return True
        except GithubException as e:
            logger.error("Failed to post summary: %s", e)
            return False

    def add_labels(self, labels: list[str]) -> None:
        """Add labels to the PR."""
        try:
            for label in labels:
                self.pr.add_to_labels(label)
        except GithubException as e:
            logger.warning("Failed to add labels: %s", e)

    def request_reviewers(self, reviewers: list[str]) -> None:
        """Request reviews from specific users."""
        try:
            self.pr.create_review_request(reviewers=reviewers)
        except GithubException as e:
            logger.warning("Failed to request reviewers: %s", e)

    def get_existing_bot_comments(self) -> list[dict]:
        """Fetch existing bot comments to avoid duplicates."""
        comments = []
        for comment in self.pr.get_issue_comments():
            if comment.user.login.endswith("[bot]") or "AI Code Review" in (comment.body or ""):
                comments.append({
                    "id": comment.id,
                    "body": comment.body,
                })
        return comments

    def get_author_stats(self) -> dict:
        """Get PR author's contribution stats for risk scoring."""
        author = self.pr.user
        try:
            contributor_stats = self.repo.get_stats_contributors()
            if contributor_stats:
                for stat in contributor_stats:
                    if stat.author.login == author.login:
                        return {
                            "total_commits": stat.total,
                            "login": author.login,
                        }
        except GithubException:
            pass
        return {"total_commits": 0, "login": author.login}
