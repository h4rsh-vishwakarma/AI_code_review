"""Multi-agent code review pipeline using LangChain."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import MODEL_NAME, OPENAI_API_KEY, TEMPERATURE, MAX_TOKENS, MAX_DIFF_SIZE
from parsers import ReviewComment, Severity
from prompts import (
    SECURITY_PROMPT,
    BUG_PROMPT,
    PERFORMANCE_PROMPT,
    STYLE_PROMPT,
    AGGREGATOR_PROMPT,
    SUMMARY_PROMPT,
    get_language_addendum,
)

logger = logging.getLogger(__name__)


class CodeReviewer:
    """Multi-agent LLM code review pipeline.

    Architecture:
        File diff → [Security Agent, Bug Agent, Performance Agent, Style Agent] (parallel)
                  → Aggregator Agent (deduplicate + rank)
                  → Summary Agent (overall verdict)
    """

    def __init__(self, rag_engine=None, rule_engine=None):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        self.json_parser = JsonOutputParser()
        self.rag_engine = rag_engine
        self.rule_engine = rule_engine

        # Build individual agent chains
        self.security_chain = SECURITY_PROMPT | self.llm | self.json_parser
        self.bug_chain = BUG_PROMPT | self.llm | self.json_parser
        self.performance_chain = PERFORMANCE_PROMPT | self.llm | self.json_parser
        self.style_chain = STYLE_PROMPT | self.llm | self.json_parser
        self.aggregator_chain = AGGREGATOR_PROMPT | self.llm | self.json_parser
        self.summary_chain = SUMMARY_PROMPT | self.llm | self.json_parser

    def _get_rag_context(self, patch: str, filename: str) -> str:
        """Retrieve relevant codebase context via RAG."""
        if self.rag_engine is None:
            return "No additional context available."
        try:
            results = self.rag_engine.search(patch, filename, k=3)
            if not results:
                return "No additional context available."
            return "\n\n---\n\n".join(results)
        except Exception as e:
            logger.warning("RAG context retrieval failed: %s", e)
            return "No additional context available."

    def _get_custom_rules(self, filename: str, language: str) -> str:
        """Get custom rules applicable to this file."""
        if self.rule_engine is None:
            return "No custom rules configured."
        try:
            rules = self.rule_engine.get_rules_for_file(filename, language)
            if not rules:
                return "No custom rules for this file."
            return "\n".join(f"- {r['name']}: {r['message']}" for r in rules)
        except Exception as e:
            logger.warning("Rule engine failed: %s", e)
            return "No custom rules configured."

    def _truncate_patch(self, patch: str) -> str:
        """Truncate oversized diffs."""
        if len(patch) > MAX_DIFF_SIZE:
            return patch[:MAX_DIFF_SIZE] + "\n\n... [diff truncated]"
        return patch

    def _run_agent(self, chain, inputs: dict, agent_name: str) -> list[dict]:
        """Run a single review agent with error handling."""
        try:
            result = chain.invoke(inputs)
            if isinstance(result, list):
                return result
            return []
        except Exception as e:
            logger.error("Agent '%s' failed: %s", agent_name, e)
            return []

    def review_file(
        self,
        filename: str,
        patch: str,
        language: str,
        pr_title: str,
        pr_description: str,
    ) -> list[dict]:
        """Review a single file using all specialized agents in parallel."""
        patch = self._truncate_patch(patch)
        rag_context = self._get_rag_context(patch, filename)
        custom_rules = self._get_custom_rules(filename, language)
        lang_addendum = get_language_addendum(language)

        base_inputs = {
            "pr_title": pr_title,
            "pr_description": pr_description,
            "filename": filename,
            "language": language,
            "patch": patch,
            "rag_context": rag_context,
            "language_addendum": lang_addendum,
            "custom_rules": custom_rules,
        }

        # Run all 4 agents in parallel using threads
        all_findings = []
        agents = [
            (self.security_chain, "security"),
            (self.bug_chain, "bugs"),
            (self.performance_chain, "performance"),
            (self.style_chain, "style"),
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._run_agent, chain, base_inputs, name): name
                for chain, name in agents
            }
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    findings = future.result()
                    logger.info("Agent '%s' found %d issues in %s", agent_name, len(findings), filename)
                    all_findings.extend(findings)
                except Exception as e:
                    logger.error("Agent '%s' raised: %s", agent_name, e)

        return all_findings

    def aggregate_findings(
        self,
        all_findings: list[dict],
        pr_title: str,
        file_count: int,
    ) -> list[dict]:
        """Deduplicate and rank findings using the aggregator agent."""
        if not all_findings:
            return []

        try:
            result = self.aggregator_chain.invoke({
                "pr_title": pr_title,
                "file_count": file_count,
                "all_findings": json.dumps(all_findings, indent=2),
            })
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error("Aggregator failed: %s", e)
            # Fallback: basic dedup by (file, line, category)
            seen = set()
            deduped = []
            for f in all_findings:
                key = (f.get("file"), f.get("line"), f.get("category"))
                if key not in seen:
                    seen.add(key)
                    deduped.append(f)
            return deduped

    def generate_summary(
        self,
        findings: list[dict],
        pr_title: str,
        pr_description: str,
        file_count: int,
        risk_score: int = 0,
    ) -> dict:
        """Generate overall review summary and verdict."""
        try:
            result = self.summary_chain.invoke({
                "pr_title": pr_title,
                "pr_description": pr_description,
                "file_count": file_count,
                "risk_score": risk_score,
                "findings": json.dumps(findings, indent=2),
            })
            if isinstance(result, dict):
                return result
            return {"summary": str(result), "verdict": "COMMENT"}
        except Exception as e:
            logger.error("Summary generation failed: %s", e)
            critical = sum(1 for f in findings if f.get("severity") == "critical")
            warnings = sum(1 for f in findings if f.get("severity") == "warning")
            if critical > 0:
                verdict = "REQUEST_CHANGES"
            elif warnings >= 3:
                verdict = "REQUEST_CHANGES"
            elif warnings > 0:
                verdict = "COMMENT"
            else:
                verdict = "APPROVE"
            return {
                "summary": f"Found {len(findings)} issues ({critical} critical, {warnings} warnings).",
                "verdict": verdict,
            }

    def review_all_files(
        self,
        files: list[dict],
        pr_title: str,
        pr_description: str,
    ) -> tuple[list[dict], dict]:
        """Full review pipeline: review all files → aggregate → summarize.

        Args:
            files: List of dicts with keys: filename, patch, language.
            pr_title: PR title.
            pr_description: PR description.

        Returns:
            Tuple of (aggregated_findings, summary_dict).
        """
        all_findings = []
        for f in files:
            findings = self.review_file(
                filename=f["filename"],
                patch=f["patch"],
                language=f.get("language", "unknown"),
                pr_title=pr_title,
                pr_description=pr_description,
            )
            all_findings.extend(findings)

        logger.info("Total raw findings: %d", len(all_findings))

        # Aggregate and deduplicate
        aggregated = self.aggregate_findings(all_findings, pr_title, len(files))
        logger.info("After aggregation: %d findings", len(aggregated))

        # Generate summary
        summary = self.generate_summary(
            findings=aggregated,
            pr_title=pr_title,
            pr_description=pr_description,
            file_count=len(files),
        )

        return aggregated, summary
