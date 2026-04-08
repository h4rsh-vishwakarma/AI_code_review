"""AI Code Review Bot — Streamlit Dashboard."""

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

HISTORY_FILE = Path(__file__).parent / "data" / "review_history.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_settings():
    """Apply session_state settings to os.environ so modules pick them up."""
    mapping = {
        "github_token": "GITHUB_TOKEN",
        "groq_api_key": "GROQ_API_KEY",
        "google_api_key": "GOOGLE_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "llm_provider": "LLM_PROVIDER",
        "model_name": "MODEL_NAME",
    }
    for key, env in mapping.items():
        val = st.session_state.get(key, "")
        if val:
            os.environ[env] = val


def save_review(record: dict):
    """Append a review record to history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_history() -> list[dict]:
    """Load all review records."""
    if not HISTORY_FILE.exists():
        return []
    records = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_review(repo: str, pr_number: int, enable_ast: bool, min_severity: str, dry_run: bool):
    """Run the full review pipeline and return structured results."""
    apply_settings()

    # Reload config to pick up new env vars
    import importlib
    import config as cfg
    importlib.reload(cfg)

    from github_client import GitHubClient
    from reviewer import CodeReviewer
    from risk_scorer import RiskScorer
    from rule_engine import RuleEngine
    from parsers import Severity, format_inline_comment, ReviewComment
    from ast_analyzer import get_ast_analyzer, parse_diff_to_old_new

    results = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "pr_number": pr_number,
    }

    # Step 1: Fetch PR
    with st.status("Fetching PR data...", expanded=True) as status:
        gh = GitHubClient(repo, pr_number)
        pr_context = gh.get_pr_context()
        results["pr_title"] = pr_context.title
        results["pr_author"] = pr_context.author
        st.write(f"**{pr_context.title}** by @{pr_context.author}")
        st.write(f"Branch: `{pr_context.head_branch}` -> `{pr_context.base_branch}`")
        st.write(f"Files changed: {len(pr_context.files)}")
        status.update(label=f"Fetched PR #{pr_number}: {pr_context.title}", state="complete")

    files = pr_context.files[:cfg.MAX_FILES_TO_REVIEW]
    file_dicts = [
        {"filename": f.filename, "patch": f.patch, "language": f.language,
         "additions": f.additions, "deletions": f.deletions}
        for f in files
    ]

    # Step 2: AST Analysis
    ast_results = {}
    if enable_ast:
        with st.status("Running AST analysis...", expanded=False) as status:
            for f in file_dicts:
                analyzer = get_ast_analyzer(f.get("language", "unknown"))
                if analyzer:
                    old_content, new_content = parse_diff_to_old_new(f["patch"])
                    try:
                        diff = analyzer.analyze(old_content, new_content, f["filename"])
                        ast_results[f["filename"]] = {
                            "summary": diff.summary,
                            "meaningful": diff.has_meaningful_changes,
                        }
                    except Exception:
                        pass

            meaningful = [f for f in file_dicts
                         if f["filename"] not in ast_results
                         or ast_results[f["filename"]]["meaningful"]]
            skipped = len(file_dicts) - len(meaningful)
            file_dicts = meaningful
            status.update(label=f"AST: {skipped} cosmetic files skipped", state="complete")

    # Step 3: Custom Rules
    with st.status("Evaluating custom rules...", expanded=False) as status:
        rule_engine = RuleEngine(".reviewbot.yml")
        rule_matches = []
        for f in file_dicts:
            matches = rule_engine.evaluate(f["filename"], f["language"], f["patch"])
            rule_matches.extend(matches)
        status.update(label=f"Custom rules: {len(rule_matches)} violations", state="complete")

    # Step 4: LLM Review
    with st.status("Running LLM review agents...", expanded=True) as status:
        reviewer = CodeReviewer(rule_engine=rule_engine)
        findings, summary = reviewer.review_all_files(
            files=file_dicts,
            pr_title=pr_context.title,
            pr_description=pr_context.description,
        )
        st.write(f"Found **{len(findings)} issues** across {len(file_dicts)} files")
        status.update(label=f"LLM review: {len(findings)} findings", state="complete")

    # Step 5: Risk Assessment
    with st.status("Calculating risk score...", expanded=False) as status:
        risk_scorer = RiskScorer()
        author_stats = gh.get_author_stats()
        risk = risk_scorer.assess(file_dicts, findings, author_stats)
        status.update(label=f"Risk: {risk.score}/100 ({risk.level})", state="complete")

    # Build results
    results.update({
        "verdict": summary.get("verdict", "COMMENT"),
        "summary_text": summary.get("summary", ""),
        "risk_score": risk.score,
        "risk_level": risk.level,
        "risk_breakdown": risk.breakdown,
        "recommendations": risk.recommendations,
        "findings": findings,
        "findings_count": len(findings),
        "critical_count": sum(1 for f in findings if f.get("severity") == "critical"),
        "warning_count": sum(1 for f in findings if f.get("severity") == "warning"),
        "suggestion_count": sum(1 for f in findings if f.get("severity") == "suggestion"),
        "files_reviewed": len(file_dicts),
        "rule_matches": [
            {"rule": m.rule.name, "file": m.file, "line": m.line,
             "message": m.rule.message, "severity": m.rule.severity}
            for m in rule_matches
        ],
        "ast_results": ast_results,
        "ast_enabled": enable_ast,
        "dry_run": dry_run,
    })

    # Save to history
    save_review(results)
    st.session_state["last_review"] = results
    return results


# ---------------------------------------------------------------------------
# Page: Review a PR
# ---------------------------------------------------------------------------

def page_review():
    st.title("Review a Pull Request")

    col1, col2 = st.columns([3, 1])
    with col1:
        repo = st.text_input("Repository", placeholder="owner/repo", value=st.session_state.get("repo", ""))
    with col2:
        pr_number = st.number_input("PR Number", min_value=1, value=1, step=1)

    col1, col2, col3 = st.columns(3)
    with col1:
        enable_ast = st.checkbox("Enable AST Analysis", value=True)
    with col2:
        min_severity = st.selectbox("Min Severity", ["suggestion", "warning", "critical"])
    with col3:
        dry_run = st.checkbox("Dry Run (don't post)", value=True)

    # Check API keys
    provider = st.session_state.get("llm_provider", os.environ.get("LLM_PROVIDER", "groq"))
    github_token = st.session_state.get("github_token", os.environ.get("GITHUB_TOKEN", ""))
    api_key_present = False
    if provider == "groq":
        api_key_present = bool(st.session_state.get("groq_api_key", os.environ.get("GROQ_API_KEY", "")))
    elif provider == "gemini":
        api_key_present = bool(st.session_state.get("google_api_key", os.environ.get("GOOGLE_API_KEY", "")))
    elif provider == "openai":
        api_key_present = bool(st.session_state.get("openai_api_key", os.environ.get("OPENAI_API_KEY", "")))

    if not github_token or not api_key_present:
        st.warning("Configure API keys in the **Settings** page first.")

    if st.button("Run Review", type="primary", disabled=not (repo and github_token and api_key_present)):
        st.session_state["repo"] = repo
        try:
            results = run_review(repo, pr_number, enable_ast, min_severity, dry_run)
            display_results(results)
        except Exception as e:
            st.error(f"Review failed: {e}")

    # Show last review if available
    elif "last_review" in st.session_state:
        st.divider()
        st.subheader("Last Review Results")
        display_results(st.session_state["last_review"])


def display_results(r: dict):
    """Display review results."""
    # Verdict banner
    verdict = r.get("verdict", "COMMENT")
    verdict_colors = {"APPROVE": "green", "REQUEST_CHANGES": "red", "COMMENT": "blue"}
    verdict_icons = {"APPROVE": "check-circle-fill", "REQUEST_CHANGES": "x-circle-fill", "COMMENT": "chat-fill"}

    st.markdown(f"### :{verdict_colors.get(verdict, 'blue')}[{verdict}] — {r.get('pr_title', 'Review')}")
    st.caption(f"PR #{r.get('pr_number')} in `{r.get('repo')}` by @{r.get('pr_author', 'unknown')} | {r.get('timestamp', '')[:19]}")

    if r.get("summary_text"):
        st.info(r["summary_text"])

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Risk Score", f"{r.get('risk_score', 0)}/100", r.get("risk_level", ""))
    col2.metric("Files", r.get("files_reviewed", 0))
    col3.metric("Critical", r.get("critical_count", 0))
    col4.metric("Warnings", r.get("warning_count", 0))
    col5.metric("Suggestions", r.get("suggestion_count", 0))

    # Risk breakdown
    breakdown = r.get("risk_breakdown", {})
    if breakdown:
        with st.expander("Risk Breakdown"):
            for factor, points in sorted(breakdown.items(), key=lambda x: -x[1]):
                st.progress(min(points / 30, 1.0), text=f"{factor}: +{points} points")

    # Recommendations
    recommendations = r.get("recommendations", [])
    if recommendations:
        with st.expander("Recommendations"):
            for rec in recommendations:
                st.warning(rec)

    # Findings table
    findings = r.get("findings", [])
    if findings:
        st.subheader(f"Findings ({len(findings)})")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            sev_filter = st.multiselect(
                "Filter by severity",
                ["critical", "warning", "suggestion"],
                default=["critical", "warning", "suggestion"],
            )
        with col2:
            cat_filter = st.multiselect(
                "Filter by category",
                list(set(f.get("category", "") for f in findings)),
                default=list(set(f.get("category", "") for f in findings)),
            )

        filtered = [
            f for f in findings
            if f.get("severity") in sev_filter and f.get("category") in cat_filter
        ]

        for f in filtered:
            sev = f.get("severity", "suggestion")
            icon = {"critical": ":red_circle:", "warning": ":large_yellow_circle:", "suggestion": ":large_blue_circle:"}.get(sev, "")
            with st.expander(f"{icon} **{sev.upper()}** | `{f.get('file')}` line {f.get('line')} — {f.get('category', '')}"):
                st.markdown(f"**Issue:** {f.get('message', '')}")
                st.markdown(f"**Suggestion:** {f.get('suggestion', '')}")
                conf = f.get("confidence", 0.8)
                st.progress(conf, text=f"Confidence: {conf:.0%}")

    # Custom rule violations
    rule_matches = r.get("rule_matches", [])
    if rule_matches:
        st.subheader(f"Custom Rule Violations ({len(rule_matches)})")
        for m in rule_matches:
            st.markdown(f"- **{m.get('rule')}** in `{m.get('file')}` line {m.get('line')}: {m.get('message')}")

    # Download
    st.download_button(
        "Download as JSON",
        data=json.dumps(r, indent=2, default=str),
        file_name=f"review_{r.get('repo', 'unknown').replace('/', '_')}_PR{r.get('pr_number', 0)}.json",
        mime="application/json",
    )


# ---------------------------------------------------------------------------
# Page: History & Stats
# ---------------------------------------------------------------------------

def page_history():
    st.title("Review History & Stats")

    history = load_history()

    if not history:
        st.info("No reviews yet. Run a review from the **Review a PR** page.")
        return

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", len(history))
    avg_risk = sum(r.get("risk_score", 0) for r in history) / len(history)
    col2.metric("Avg Risk Score", f"{avg_risk:.0f}/100")
    total_findings = sum(r.get("findings_count", 0) for r in history)
    col3.metric("Total Findings", total_findings)
    total_critical = sum(r.get("critical_count", 0) for r in history)
    col4.metric("Critical Issues", total_critical)

    # Feedback stats
    try:
        from feedback import FeedbackCollector
        stats = FeedbackCollector().get_stats()
        if stats["total"] > 0:
            st.metric("Feedback Precision", f"{stats['precision']:.0%}")
    except Exception:
        pass

    st.divider()

    # Charts
    if len(history) > 1:
        import plotly.express as px
        import plotly.graph_objects as go

        col1, col2 = st.columns(2)

        with col1:
            # Verdict distribution
            verdicts = [r.get("verdict", "COMMENT") for r in history]
            verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}
            fig = px.pie(
                names=list(verdict_counts.keys()),
                values=list(verdict_counts.values()),
                title="Verdicts",
                color_discrete_sequence=["#28a745", "#dc3545", "#17a2b8"],
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Risk score distribution
            scores = [r.get("risk_score", 0) for r in history]
            fig = px.histogram(x=scores, nbins=10, title="Risk Score Distribution",
                               labels={"x": "Risk Score", "y": "Count"})
            st.plotly_chart(fig, use_container_width=True)

        # Findings by category
        all_findings = []
        for r in history:
            for f in r.get("findings", []):
                all_findings.append(f.get("category", "unknown"))
        if all_findings:
            cat_counts = {c: all_findings.count(c) for c in set(all_findings)}
            fig = px.bar(
                x=list(cat_counts.keys()),
                y=list(cat_counts.values()),
                title="Findings by Category",
                labels={"x": "Category", "y": "Count"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # History table
    st.subheader("All Reviews")
    table_data = []
    for r in reversed(history):
        table_data.append({
            "Time": r.get("timestamp", "")[:16],
            "Repo": r.get("repo", ""),
            "PR": f"#{r.get('pr_number', '')}",
            "Title": r.get("pr_title", "")[:50],
            "Verdict": r.get("verdict", ""),
            "Risk": r.get("risk_score", 0),
            "Findings": r.get("findings_count", 0),
            "Critical": r.get("critical_count", 0),
        })
    st.dataframe(table_data, use_container_width=True)

    # Click to view detail
    if history:
        idx = st.selectbox("View details for review:", range(len(history)),
                           format_func=lambda i: f"{history[i].get('repo')} PR#{history[i].get('pr_number')} — {history[i].get('pr_title', '')[:40]}")
        if st.button("View Details"):
            st.session_state["last_review"] = history[idx]
            st.switch_page(page_results)


# ---------------------------------------------------------------------------
# Page: Results (standalone detail view)
# ---------------------------------------------------------------------------

def page_results():
    st.title("Review Results")

    if "last_review" not in st.session_state:
        st.info("No review to display. Run a review from the **Review a PR** page.")
        return

    display_results(st.session_state["last_review"])


# ---------------------------------------------------------------------------
# Page: Settings
# ---------------------------------------------------------------------------

def page_settings():
    st.title("Settings")

    # API Keys
    st.subheader("API Keys")

    st.text_input(
        "GitHub Token",
        type="password",
        key="github_token",
        value=st.session_state.get("github_token", os.environ.get("GITHUB_TOKEN", "")),
        help="Personal access token with repo scope",
    )

    st.divider()
    st.subheader("LLM Provider")

    provider = st.selectbox(
        "Provider",
        ["groq", "gemini", "openai"],
        index=["groq", "gemini", "openai"].index(
            st.session_state.get("llm_provider", os.environ.get("LLM_PROVIDER", "groq"))
        ),
        key="llm_provider",
    )

    defaults = {
        "groq": ("llama-3.3-70b-versatile", "GROQ_API_KEY", "groq_api_key", "gsk_..."),
        "gemini": ("gemini-2.0-flash", "GOOGLE_API_KEY", "google_api_key", "AIza..."),
        "openai": ("gpt-4o-mini", "OPENAI_API_KEY", "openai_api_key", "sk-..."),
    }
    model_default, env_key, session_key, placeholder = defaults[provider]

    st.text_input(
        f"{provider.title()} API Key",
        type="password",
        key=session_key,
        value=st.session_state.get(session_key, os.environ.get(env_key, "")),
        help=f"Starts with {placeholder}",
    )

    st.text_input(
        "Model Name",
        key="model_name",
        value=st.session_state.get("model_name", os.environ.get("MODEL_NAME", model_default)),
    )

    # Test connection
    if st.button("Test Connection"):
        apply_settings()
        try:
            import importlib
            import config as cfg
            importlib.reload(cfg)
            from reviewer import CodeReviewer
            reviewer = CodeReviewer()
            if reviewer.test_api_connection():
                st.success("Connection successful!")
            else:
                st.error("Connection failed. Check your API key.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.subheader("Review Settings")

    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Max Files to Review", min_value=1, max_value=50, value=20, key="max_files")
        st.slider("Risk High Threshold", 0, 100, 80, key="risk_high")
    with col2:
        st.number_input("Max Diff Size (chars)", min_value=1000, max_value=50000, value=10000, key="max_diff")
        st.slider("Risk Medium Threshold", 0, 100, 50, key="risk_medium")

    st.divider()
    st.subheader("Custom Rules")
    st.caption("Paste your `.reviewbot.yml` content below to preview")

    rules_content = st.text_area("Rules YAML", height=200, placeholder="rules:\n  - name: no-eval\n    pattern: ...")
    if rules_content and st.button("Preview Rules"):
        try:
            from rule_engine import RuleEngine
            engine = RuleEngine()
            engine.load_from_string(rules_content)
            st.success(f"Loaded {len(engine.rules)} rules")
            for rule in engine.rules:
                st.write(f"- **{rule.name}** ({rule.severity}): {rule.message}")
        except Exception as e:
            st.error(f"Failed to parse rules: {e}")


# ---------------------------------------------------------------------------
# App Navigation
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AI Code Review Bot",
        page_icon=":robot_face:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar branding
    with st.sidebar:
        st.markdown("## :robot_face: AI Code Review Bot")
        st.caption("Multi-agent LLM code reviewer")
        st.divider()

        # Quick status
        provider = st.session_state.get("llm_provider", os.environ.get("LLM_PROVIDER", "groq"))
        has_key = bool(st.session_state.get("groq_api_key", os.environ.get("GROQ_API_KEY", "")))
        has_github = bool(st.session_state.get("github_token", os.environ.get("GITHUB_TOKEN", "")))

        st.markdown(f"**Provider:** {provider}")
        st.markdown(f"**LLM Key:** {'Set' if has_key else 'Not set'}")
        st.markdown(f"**GitHub Token:** {'Set' if has_github else 'Not set'}")

    pg = st.navigation([
        st.Page(page_review, title="Review a PR", icon=":material/search:"),
        st.Page(page_results, title="Results", icon=":material/checklist:"),
        st.Page(page_history, title="History & Stats", icon=":material/bar_chart:"),
        st.Page(page_settings, title="Settings", icon=":material/settings:"),
    ])
    pg.run()


if __name__ == "__main__":
    main()
