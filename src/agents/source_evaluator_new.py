"""
Source evaluation agent using configurable backends.

This is the new implementation that supports both pydantic.ai and baremetal backends
while maintaining the same interface as the original source evaluator.
"""

from typing import List
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import WebSearchResult, SourceEvaluation
from config.settings import settings
from agents.factory import create_source_evaluator_backend
from agents.base import AgentBackend, APIError, ValidationError
from utils.text_utils import extract_domain


class SourceEvaluatorAgent:
    """
    Source evaluation agent with configurable backend support.

    This agent can use either pydantic.ai or baremetal backends based on
    configuration, providing flexibility while maintaining the same interface.
    """

    def __init__(self, backend: AgentBackend = None):
        """
        Initialize the source evaluator agent.

        Args:
            backend: Optional backend instance. If not provided, uses factory default.
        """
        self.backend = backend or create_source_evaluator_backend()

    def evaluate_source(self, result: WebSearchResult, topic: str) -> SourceEvaluation:
        """
        Evaluate a single source for relevance and quality.

        Args:
            result: Search result to evaluate
            topic: Topic being researched

        Returns:
            SourceEvaluation with scores and recommendation
        """
        prompt = f"""Evaluate this web source for the topic: "{topic}"

SOURCE DETAILS:
- URL: {result.url}
- Title: {result.title}
- Snippet: {result.snippet}
- Search Query: {result.search_query}

Evaluate this source based on relevance to "{topic}", domain authority, content type, and expected quality.
Provide scores, classification, and a clear decision on whether to scrape this source."""

        try:
            return self.backend.generate_response(prompt, SourceEvaluation)
        except Exception as e:
            if settings.logging.verbose_output:
                print(f"    ⚠️  Backend evaluation failed for {result.url}: {e}")

            # Fallback to rule-based evaluation
            return self._rule_based_evaluation(result, topic)

    def _rule_based_evaluation(self, result: WebSearchResult, topic: str) -> SourceEvaluation:
        """
        Simple rule-based evaluation as final fallback.
        """
        # Simple relevance scoring based on keyword matching
        topic_words = set(topic.lower().split())
        title_words = set(result.title.lower().split())
        snippet_words = set(result.snippet.lower().split())

        title_matches = len(topic_words.intersection(title_words))
        snippet_matches = len(topic_words.intersection(snippet_words))

        relevance_score = min(1.0, (title_matches * 0.3 + snippet_matches * 0.1))
        relevance_score = max(0.3, relevance_score)  # Minimum relevance

        return SourceEvaluation(
            url=result.url,
            title=result.title,
            relevance_score=relevance_score,
            authority_score=self._estimate_domain_authority(result.url),
            content_type="unknown",
            should_scrape=relevance_score >= settings.source_selection.min_relevance_score,
            reasoning=f"Rule-based evaluation: {title_matches} title matches, {snippet_matches} snippet matches",
            estimated_quality="medium"
        )

    def _estimate_domain_authority(self, url: str) -> float:
        """
        Estimate domain authority based on URL patterns.
        This is a fallback when LLM evaluation fails.
        """
        domain = extract_domain(url).lower()

        # High authority domains
        if any(tld in domain for tld in ['.edu', '.gov', '.org']):
            return 0.9

        # Known high-quality domains
        high_quality_domains = {
            'wikipedia.org', 'github.com', 'stackoverflow.com', 'mozilla.org',
            'w3.org', 'ietf.org', 'ieee.org', 'acm.org', 'arxiv.org'
        }
        if any(hq_domain in domain for hq_domain in high_quality_domains):
            return 0.8

        # News and established platforms
        news_domains = {
            'bbc.com', 'reuters.com', 'npr.org', 'pbs.org', 'cnn.com',
            'techcrunch.com', 'arstechnica.com', 'wired.com'
        }
        if any(news_domain in domain for news_domain in news_domains):
            return 0.7

        # Default medium authority
        return 0.5


def evaluate_sources(search_results: List[WebSearchResult], topic: str) -> List[SourceEvaluation]:
    """
    Evaluate search results and select the best sources for scraping.

    This is the main entry point that maintains compatibility with the existing codebase
    while using the new configurable backend system.
    """
    if not search_results:
        return []

    if settings.logging.verbose_output:
        backend_type = settings.agent.backend_type
        print(f"🔍 Evaluating {len(search_results)} sources using {backend_type} backend")

    # Create agent instance
    evaluator = SourceEvaluatorAgent()
    evaluations = []

    for i, result in enumerate(search_results, 1):
        if settings.logging.verbose_output:
            print(f"  📊 Evaluating source {i}/{len(search_results)}: {result.title[:50]}...")

        evaluation = evaluator.evaluate_source(result, topic)

        if settings.logging.verbose_output:
            status = "✅ SELECTED" if evaluation.should_scrape else "❌ REJECTED"
            print(f"    {status} - Relevance: {evaluation.relevance_score:.2f}, Authority: {evaluation.authority_score:.2f}")
            print(f"    Type: {evaluation.content_type}, Quality: {evaluation.estimated_quality}")
            print(f"    Reason: {evaluation.reasoning[:100]}...")

        evaluations.append(evaluation)

    if settings.logging.verbose_output:
        selected_count = sum(1 for e in evaluations if e.should_scrape)
        print(f"📊 Evaluation complete: {selected_count}/{len(evaluations)} sources selected")

    return evaluations


def select_top_sources(evaluations: List[SourceEvaluation]) -> List[SourceEvaluation]:
    """
    Select the top sources based on evaluation scores.

    This function remains unchanged from the original implementation.
    """
    if not evaluations:
        return []

    if settings.logging.verbose_output:
        print(f"🎯 Selecting top sources from {len(evaluations)} evaluations...")

    # Step 1: Filter by agent's recommendation and minimum relevance
    filtered = [e for e in evaluations if e.should_scrape and
                e.relevance_score >= settings.source_selection.min_relevance_score]

    if settings.logging.verbose_output:
        print(f"  📋 After filtering: {len(filtered)} sources pass minimum criteria")

    # Step 2: Deduplicate similar domains
    deduplicated = _deduplicate_sources(filtered)

    if settings.logging.verbose_output:
        print(f"  🔄 After deduplication: {len(deduplicated)} unique sources")

    # Step 3: Sort by combined score (weighted relevance + authority)
    def calculate_score(evaluation: SourceEvaluation) -> float:
        # Weight relevance more heavily than authority
        relevance_weight = 0.7
        authority_weight = 0.3
        base_score = (evaluation.relevance_score * relevance_weight +
                     evaluation.authority_score * authority_weight)

        # Bonus for high-quality content types
        quality_bonus = {
            "academic": 0.1,
            "documentation": 0.08,
            "reference": 0.06,
            "news": 0.04,
            "blog": 0.02,
            "commercial": 0.0
        }
        bonus = quality_bonus.get(evaluation.content_type, 0.0)

        # Bonus for high estimated quality
        if evaluation.estimated_quality == "high":
            bonus += 0.05
        elif evaluation.estimated_quality == "medium":
            bonus += 0.02

        return base_score + bonus

    sorted_sources = sorted(deduplicated, key=calculate_score, reverse=True)

    # Step 4: Select top N sources with content type diversity
    selected = _select_diverse_sources(sorted_sources, settings.source_selection.max_sources_to_scrape)

    if settings.logging.verbose_output:
        print(f"  ✅ Final selection: {len(selected)} sources chosen")
        for i, source in enumerate(selected, 1):
            score = calculate_score(source)
            print(f"    {i}. {source.title[:60]}... (Score: {score:.3f})")
            print(f"       Type: {source.content_type}, Quality: {source.estimated_quality}")

    return selected


def _deduplicate_sources(evaluations: List[SourceEvaluation]) -> List[SourceEvaluation]:
    """
    Remove duplicate sources from the same domain, keeping the highest scored one.
    """
    domain_map = {}

    for evaluation in evaluations:
        domain = extract_domain(evaluation.url)

        if domain not in domain_map:
            domain_map[domain] = evaluation
        else:
            # Keep the one with higher combined score
            existing = domain_map[domain]
            current_score = evaluation.relevance_score + evaluation.authority_score
            existing_score = existing.relevance_score + existing.authority_score

            if current_score > existing_score:
                domain_map[domain] = evaluation

    return list(domain_map.values())


def _select_diverse_sources(sorted_sources: List[SourceEvaluation], max_count: int) -> List[SourceEvaluation]:
    """
    Select sources ensuring content type diversity.
    """
    selected = []
    content_type_counts = {}

    for source in sorted_sources:
        if len(selected) >= max_count:
            break

        content_type = source.content_type
        current_count = content_type_counts.get(content_type, 0)

        # Limit sources per content type to ensure diversity
        max_per_type = max(1, max_count // 4)  # At most 1/4 of sources from same type

        if current_count < max_per_type:
            selected.append(source)
            content_type_counts[content_type] = current_count + 1

    # If we haven't reached max_count, fill remaining slots
    remaining_slots = max_count - len(selected)
    if remaining_slots > 0:
        for source in sorted_sources:
            if source not in selected and remaining_slots > 0:
                selected.append(source)
                remaining_slots -= 1

    return selected
