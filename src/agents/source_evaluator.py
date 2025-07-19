"""Source evaluation agent for selecting high-quality sources."""

from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.models import WebSearchResult, SourceEvaluation
from config.settings import settings


# Create the source evaluation agent
def create_source_evaluator() -> Agent:
    """
    Create and configure the source evaluation agent.

    This agent evaluates search results for:
    - Relevance to the topic
    - Authority and credibility of the source
    - Content type and expected quality
    - Likelihood of containing substantive information
    """
    model = OpenAIModel(
        model_name=settings.llm.model,
        provider=OpenAIProvider(base_url=settings.llm.base_url)
    )

    system_prompt = """You are an expert source evaluation agent for educational content research.

Your task is to evaluate web sources for their quality, relevance, and value for creating comprehensive learning materials.

EVALUATION CRITERIA:

1. RELEVANCE SCORE (0.0-1.0):
   - 1.0: Directly addresses the topic with comprehensive coverage
   - 0.8: Highly relevant with good topic coverage
   - 0.6: Moderately relevant, covers some aspects
   - 0.4: Tangentially related or limited coverage
   - 0.2: Barely related to the topic
   - 0.0: Not related to the topic

2. AUTHORITY SCORE (0.0-1.0):
   - 1.0: Authoritative sources (gov, edu, established organizations, expert sites)
   - 0.8: Professional publications, established media, technical documentation
   - 0.6: Industry blogs, company websites, established platforms
   - 0.4: Personal blogs by experts, forums with good moderation
   - 0.2: General user-generated content, unverified sources
   - 0.0: Spam, promotional, or unreliable sources

3. CONTENT TYPE Classification:
   - "academic": Research papers, educational institutions, formal studies
   - "documentation": Official docs, technical guides, tutorials
   - "news": News articles, industry reports, current events
   - "blog": Expert blogs, thought leadership, analysis
   - "reference": Wikis, encyclopedias, fact sheets
   - "commercial": Company sites, product pages, marketing content

4. QUALITY ESTIMATION:
   - "high": Authoritative, comprehensive, well-structured content expected
   - "medium": Good information quality but may lack depth or authority
   - "low": Limited value, promotional, or superficial content expected

5. SCRAPING DECISION:
   - should_scrape: true if relevance ≥ {settings.source_selection.min_relevance_score} AND authority ≥ 0.3
   - Prioritize diverse content types and authoritative sources
   - Avoid duplicate domains when possible

6. REASONING:
   - Provide clear justification for scores and decision
   - Mention specific factors that influenced the evaluation
   - Note any red flags or positive indicators

CRITICAL INSTRUCTIONS:
1. Return ONLY a JSON object, no other text
2. Use these EXACT field names and types:
   - "url": string (copy from input)
   - "title": string (copy from input)
   - "relevance_score": number between 0.0 and 1.0
   - "authority_score": number between 0.0 and 1.0
   - "content_type": one of "academic", "documentation", "news", "blog", "reference", "commercial"
   - "should_scrape": boolean true or false (not strings)
   - "reasoning": string explanation
   - "estimated_quality": one of "high", "medium", "low"

EXAMPLE RESPONSE:
{"url": "https://example.com", "title": "Example Title", "relevance_score": 0.8, "authority_score": 0.7, "content_type": "blog", "should_scrape": true, "reasoning": "Highly relevant content", "estimated_quality": "medium"}

Return NOTHING except the JSON object above with your evaluation."""

    return Agent(
        model,
        output_type=SourceEvaluation,
        system_prompt=system_prompt
    )


def evaluate_sources(search_results: List[WebSearchResult], topic: str) -> List[SourceEvaluation]:
    """
    Evaluate search results and select the best sources for scraping.

    Uses an LLM-based agent to intelligently assess each source for:
    - Relevance to the topic
    - Domain authority and credibility
    - Expected content quality
    - Suitability for scraping
    """
    if not search_results:
        return []

    if settings.logging.verbose_output:
        print(f"🔍 Evaluating {len(search_results)} sources for topic: '{topic}'")

    evaluations = []

    for i, result in enumerate(search_results, 1):
        if settings.logging.verbose_output:
            print(f"  📊 Evaluating source {i}/{len(search_results)}: {result.title[:50]}...")

        # Try robust evaluation first, fallback to simple method
        evaluation = _evaluate_source_robust(result, topic)

        if evaluation:
            if settings.logging.verbose_output:
                status = "✅ SELECTED" if evaluation.should_scrape else "❌ REJECTED"
                print(f"    {status} - Relevance: {evaluation.relevance_score:.2f}, Authority: {evaluation.authority_score:.2f}")
                print(f"    Type: {evaluation.content_type}, Quality: {evaluation.estimated_quality}")
                print(f"    Reason: {evaluation.reasoning[:100]}...")

            evaluations.append(evaluation)
        else:
            # Final fallback
            if settings.logging.verbose_output:
                print(f"    ⚠️  Using rule-based fallback evaluation")

            fallback_evaluation = SourceEvaluation(
                url=result.url,
                title=result.title,
                relevance_score=0.5,
                authority_score=_estimate_domain_authority(result.url),
                content_type="unknown",
                should_scrape=True,
                reasoning="LLM evaluation failed completely. Using rule-based fallback.",
                estimated_quality="medium"
            )
            evaluations.append(fallback_evaluation)

    if settings.logging.verbose_output:
        selected_count = sum(1 for e in evaluations if e.should_scrape)
        print(f"📊 Evaluation complete: {selected_count}/{len(evaluations)} sources selected for scraping")

    return evaluations


def _evaluate_source_robust(result: WebSearchResult, topic: str) -> SourceEvaluation:
    """
    Robust source evaluation with multiple fallback strategies.
    """
    # Create evaluation prompt
    prompt = f"""Evaluate this web source for the topic: "{topic}"

SOURCE DETAILS:
- URL: {result.url}
- Title: {result.title}
- Snippet: {result.snippet}
- Search Query: {result.search_query}

Evaluate this source based on relevance to "{topic}", domain authority, content type, and expected quality.
Provide scores, classification, and a clear decision on whether to scrape this source."""

    # Strategy 1: Try with pydantic-ai agent
    try:
        evaluator = create_source_evaluator()
        result_eval = evaluator.run_sync(prompt)
        return result_eval.data
    except Exception as e:
        if settings.logging.verbose_output:
            print(f"    ⚠️  Pydantic-AI failed: {e}")

    # Strategy 2: Try raw LLM call with JSON parsing
    try:
        import requests
        import json

        # Make direct API call
        api_url = settings.llm.base_url.replace('/v1', '') + '/api/generate'

        payload = {
            "model": settings.llm.model,
            "prompt": f"""Evaluate this web source for educational content research.

Topic: {topic}
URL: {result.url}
Title: {result.title}
Snippet: {result.snippet}

Return ONLY this JSON format:
{{"url": "{result.url}", "title": "{result.title}", "relevance_score": 0.8, "authority_score": 0.7, "content_type": "blog", "should_scrape": true, "reasoning": "explanation here", "estimated_quality": "medium"}}

Replace values with your evaluation. Return ONLY the JSON.""",
            "stream": False,
            "options": {"temperature": 0.1}
        }

        response = requests.post(api_url, json=payload, timeout=30)

        if response.status_code == 200:
            llm_response = response.json()['response']

            # Clean and parse JSON
            cleaned_json = _clean_llm_json_response(llm_response)
            evaluation_data = json.loads(cleaned_json)

            # Validate and fix fields
            validated_data = _validate_evaluation_fields(evaluation_data, result)

            # Create SourceEvaluation object
            return SourceEvaluation(**validated_data)

    except Exception as e:
        if settings.logging.verbose_output:
            print(f"    ⚠️  Raw LLM call failed: {e}")

    # Strategy 3: Rule-based evaluation
    try:
        return _rule_based_evaluation(result, topic)
    except Exception as e:
        if settings.logging.verbose_output:
            print(f"    ⚠️  Rule-based evaluation failed: {e}")

    return None


def _rule_based_evaluation(result: WebSearchResult, topic: str) -> SourceEvaluation:
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
        authority_score=_estimate_domain_authority(result.url),
        content_type="unknown",
        should_scrape=relevance_score >= settings.source_selection.min_relevance_score,
        reasoning=f"Rule-based evaluation: {title_matches} title matches, {snippet_matches} snippet matches",
        estimated_quality="medium"
    )


def select_top_sources(evaluations: List[SourceEvaluation]) -> List[SourceEvaluation]:
    """
    Select the top sources based on evaluation scores.

    Applies intelligent filtering and ranking:
    - Filters by minimum relevance score
    - Deduplicates similar domains
    - Sorts by combined relevance and authority scores
    - Limits to maximum number of sources to scrape
    - Ensures content type diversity
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


def _estimate_domain_authority(url: str) -> float:
    """
    Estimate domain authority based on URL patterns.
    This is a fallback when LLM evaluation fails.
    """
    from utils.text_utils import extract_domain

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


def _deduplicate_sources(evaluations: List[SourceEvaluation]) -> List[SourceEvaluation]:
    """
    Remove duplicate sources from the same domain, keeping the highest scored one.
    """
    from utils.text_utils import extract_domain

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


def _clean_llm_json_response(raw_response: str) -> str:
    """
    Clean common LLM JSON response issues.
    """
    # Remove markdown code block formatting
    if "```json" in raw_response:
        raw_response = raw_response.split("```json")[1].split("```")[0]
    elif "```" in raw_response:
        raw_response = raw_response.split("```")[1].split("```")[0]

    # Remove common prefixes
    prefixes_to_remove = [
        "Here's the evaluation:",
        "Here is the evaluation:",
        "Evaluation:",
        "JSON:",
        "Response:",
        "Here's my analysis:",
        "Analysis:",
    ]

    for prefix in prefixes_to_remove:
        if raw_response.strip().lower().startswith(prefix.lower()):
            raw_response = raw_response[len(prefix):].strip()

    # Find JSON object bounds
    start_idx = raw_response.find('{')
    end_idx = raw_response.rfind('}')

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        raw_response = raw_response[start_idx:end_idx+1]

    return raw_response.strip()


def _validate_evaluation_fields(data: dict, original_result: 'WebSearchResult') -> dict:
    """
    Validate and fix common field issues in evaluation data.
    """
    # Ensure required fields are present
    required_fields = {
        'url': original_result.url,
        'title': original_result.title,
        'relevance_score': 0.5,
        'authority_score': 0.5,
        'content_type': 'unknown',
        'should_scrape': True,
        'reasoning': 'No reasoning provided',
        'estimated_quality': 'medium'
    }

    for field, default_value in required_fields.items():
        if field not in data:
            data[field] = default_value

    # Fix data types
    try:
        data['relevance_score'] = float(data['relevance_score'])
        data['authority_score'] = float(data['authority_score'])
        data['relevance_score'] = max(0.0, min(1.0, data['relevance_score']))  # Clamp to 0-1
        data['authority_score'] = max(0.0, min(1.0, data['authority_score']))  # Clamp to 0-1
    except (ValueError, TypeError):
        data['relevance_score'] = 0.5
        data['authority_score'] = 0.5

    # Fix boolean field
    if isinstance(data['should_scrape'], str):
        data['should_scrape'] = data['should_scrape'].lower() in ['true', 'yes', '1']

    # Validate content_type
    valid_types = ['academic', 'documentation', 'news', 'blog', 'reference', 'commercial', 'unknown']
    if data['content_type'] not in valid_types:
        data['content_type'] = 'unknown'

    # Validate quality
    valid_qualities = ['high', 'medium', 'low']
    if data['estimated_quality'] not in valid_qualities:
        data['estimated_quality'] = 'medium'

    return data


def test_source_evaluator():
    """
    Test function to diagnose source evaluator issues.
    """
    print("🧪 Testing source evaluator...")

    # Create test data
    test_result = WebSearchResult(
        url="https://example.com/test",
        title="Test Article About Machine Learning",
        snippet="This is a test snippet about machine learning concepts and applications.",
        search_query="machine learning basics"
    )

    topic = "machine learning"

    try:
        evaluator = create_source_evaluator()

        prompt = f"""Evaluate this web source for the topic: "{topic}"

SOURCE DETAILS:
- URL: {test_result.url}
- Title: {test_result.title}
- Snippet: {test_result.snippet}
- Search Query: {test_result.search_query}

Evaluate this source based on relevance to "{topic}", domain authority, content type, and expected quality.
Provide scores, classification, and a clear decision on whether to scrape this source."""

        print("📤 Sending test prompt to LLM...")
        result = evaluator.run_sync(prompt)
        print(f"✅ LLM Response Type: {type(result)}")
        print(f"✅ LLM Data Type: {type(result.data)}")
        print(f"✅ LLM Data: {result.data}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_source_evaluator()
