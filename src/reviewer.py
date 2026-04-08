"""Multi-agent code review pipeline using LangChain."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import (
    MODEL_NAME, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY,
    TEMPERATURE, MAX_TOKENS, MAX_DIFF_SIZE, LLM_PROVIDER,
)
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


def create_llm():
    """Create the LLM instance based on configured provider."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        print(f"[DEBUG] Using Groq provider, model={MODEL_NAME}")
        return ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        print(f"[DEBUG] Using Gemini provider, model={MODEL_NAME}")
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )
    else:
        from langchain_openai import ChatOpenAI
        print(f"[DEBUG] Using OpenAI provider, model={MODEL_NAME}")
        return ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )


class CodeReviewer:
    """Multi-agent LLM code review pipeline.

    Architecture:
        File diff → [Security Agent, Bug Agent, Performance Agent, Style Agent] (parallel)
                  → Aggregator Agent (deduplicate + rank)
                  → Summary Agent (overall verdict)
    """

    def __init__(self, rag_engine=None, rule_engine=None):
        self.llm = create_llm()
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

    def test_api_connection(self) -> bool:
        """Quick test that the OpenAI API key works."""
        try:
            result = self.llm.invoke("Reply with exactly: OK")
            print(f"[DEBUG] API test successful: {result.content[:50]}")
            return True
        except Exception as e:
            print(f"[DEBUG] API test FAILED: {type(e).__name__}: {e}")
            return False

    def _run_agent(self, chain, inputs: dict, agent_name: str) -> list[dict]:
        """Run a single review agent with error handling."""
        try:
            print(f"[DEBUG] Running agent '{agent_name}'...")
            result = chain.invoke(inputs)
            print(f"[DEBUG] Agent '{agent_name}' returned type={type(result).__name__}, value={str(result)[:300]}")
            if isinstance(result, list):
                print(f"[DEBUG] Agent '{agent_name}' found {len(result)} issues")
                return result
            # If result is a string, try parsing it as JSON
            if isinstance(result, str):
                import json as _json
                try:
                    parsed = _json.loads(result)
                    if isinstance(parsed, list):
                        print(f"[DEBUG] Agent '{agent_name}' string->JSON parsed {len(parsed)} issues")
                        return parsed
                except _json.JSONDecodeError:
                    pass
            print(f"[DEBUG] Agent '{agent_name}' returned unexpected result: {str(result)[:200]}")
            return []
        except Exception as e:
            print(f"[DEBUG] Agent '{agent_name}' FAILED: {type(e).__name__}: {e}")
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

        # Fallback: if all agents returned nothing, try a single combined review
        if not all_findings:
            logger.warning("All agents returned 0 findings for %s — running fallback review", filename)
            all_findings = self._fallback_review(base_inputs)

        return all_findings

    def _fallback_review(self, inputs: dict) -> list[dict]:
        """Single-pass fallback review when specialized agents all return empty."""
        from langchain_core.prompts import ChatPromptTemplate

        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert code reviewer. Analyze the diff and find ALL issues:
bugs, security flaws, performance problems, and style issues.

Only comment on CHANGED lines (lines starting with +).
Be specific — reference exact line numbers.

Output as JSON array:
[{{"file": "path", "line": N, "severity": "critical|warning|suggestion", "category": "bug|security|performance|style", "message": "issue", "suggestion": "fix", "confidence": 0.9}}]

If no issues, return: []"""),
            ("human", """File: {filename} ({language})

Diff:
```
{patch}
```

Find all issues."""),
        ])

        try:
            chain = fallback_prompt | self.llm | self.json_parser
            result = chain.invoke(inputs)
            if isinstance(result, list):
                logger.info("Fallback review found %d issues", len(result))
                return result
            logger.warning("Fallback returned non-list: %s", type(result).__name__)
            return []
        except Exception as e:
            logger.error("Fallback review also failed: %s: %s", type(e).__name__, e)
            return []

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
