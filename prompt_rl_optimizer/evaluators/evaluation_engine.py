"""
Evaluation Engine Module.

Provides robust scoring system with rule-based, LLM-as-judge, and hybrid evaluation.
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import asyncio

from core.models import LLMOutput, EvaluationMetrics, RewardBreakdown
from configs.experiment_config import EvaluationConfig, RewardConfig


class BaseEvaluator(ABC):
    """Abstract base class for evaluators."""
    
    @abstractmethod
    def evaluate(self, output: LLMOutput, context: Dict) -> EvaluationMetrics:
        """Evaluate an LLM output and return metrics."""
        pass


class RuleBasedEvaluator(BaseEvaluator):
    """
    Rule-based evaluation using regex patterns, schema validation, and heuristics.
    
    Evaluates:
    - Format compliance (regex matching)
    - Required/forbidden keywords
    - Output length constraints
    - Repetition detection
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(self, output: LLMOutput, context: Dict) -> EvaluationMetrics:
        """Perform rule-based evaluation."""
        text = output.output_text
        
        # Format compliance check
        format_score = self._check_format_compliance(text, context)
        
        # Keyword checks
        required_score = self._check_required_keywords(text)
        forbidden_score = self._check_forbidden_keywords(text)
        
        # Length assessment
        length_score = self._assess_length(text, context)
        
        # Repetition detection
        repetition_ratio = self._detect_repetition(text)
        
        # Structure analysis
        num_sections = self._count_sections(text)
        
        # Compute individual metric scores
        correctness_score = 0.5  # Placeholder; actual correctness needs semantic analysis
        format_compliance_score = format_score * required_score * forbidden_score
        completeness_score = min(1.0, num_sections / max(1, context.get('expected_sections', 3)))
        conciseness_score = length_score * (1 - repetition_ratio * 0.5)
        
        # Collect errors
        errors = []
        if format_compliance_score < 0.8:
            errors.append("Format compliance below threshold")
        if required_score < 1.0:
            errors.append("Missing required keywords")
        if forbidden_score < 1.0:
            errors.append("Contains forbidden keywords")
        
        return EvaluationMetrics(
            output_id=output.id,
            correctness_score=correctness_score,
            format_compliance_score=format_compliance_score,
            completeness_score=completeness_score,
            conciseness_score=conciseness_score,
            output_length=len(text),
            num_sections=num_sections,
            has_required_format=format_compliance_score >= self.config.format_compliance_threshold,
            repetition_ratio=repetition_ratio,
            rule_based_errors=errors
        )
    
    def _check_format_compliance(self, text: str, context: Dict) -> float:
        """Check if output follows expected format."""
        expected_format = context.get('expected_format', self.config.expected_format)
        
        if not expected_format:
            return 1.0  # No format requirement
        
        # Common format patterns
        format_patterns = {
            'json': r'\\{[^}]*\\}',
            'xml': r'<[^>]+>[^<]*</[^>]+>',
            'numbered_list': r'^\\d+\\.',
            'bullet_points': r'^[•\\-\\*]\\s',
            'code_block': r'```[\\s\\S]*?```',
            'boxed_answer': r'\\\\boxed\\{[^}]+\\}',
            'final_answer': r'[Ff]inal [Aa]nswer[:\\s]+',
        }
        
        # Check for specific format
        if expected_format.lower() in format_patterns:
            pattern = format_patterns[expected_format.lower()]
            if re.search(pattern, text, re.MULTILINE):
                return 1.0
            else:
                return 0.5
        
        # Generic format check - look for structure indicators
        structure_indicators = [
            r'^#+\\s',  # Headers
            r'^\\d+\\.',  # Numbered items
            r'^[•\\-\\*]\\s',  # Bullet points
            r'\\n\\n',  # Paragraph breaks
        ]
        
        matches = sum(1 for pattern in structure_indicators if re.search(pattern, text, re.MULTILINE))
        return min(1.0, matches / len(structure_indicators))
    
    def _check_required_keywords(self, text: str) -> float:
        """Check for required keywords."""
        required = self.config.required_keywords
        
        if not required:
            return 1.0
        
        text_lower = text.lower()
        found = sum(1 for kw in required if kw.lower() in text_lower)
        
        return found / len(required)
    
    def _check_forbidden_keywords(self, text: str) -> float:
        """Check for forbidden keywords."""
        forbidden = self.config.forbidden_keywords
        
        if not forbidden:
            return 1.0
        
        text_lower = text.lower()
        violations = sum(1 for kw in forbidden if kw.lower() in text_lower)
        
        # Penalize based on number of violations
        return max(0.0, 1.0 - (violations * 0.2))
    
    def _assess_length(self, text: str, context: Dict) -> float:
        """Assess if output length is appropriate."""
        length = len(text.split())
        
        min_length = context.get('min_length', 50)
        max_length = context.get('max_length', 1000)
        
        if length < min_length:
            return max(0.0, length / min_length)
        elif length > max_length:
            return max(0.0, 1.0 - ((length - max_length) / max_length) * 0.5)
        else:
            return 1.0
    
    def _detect_repetition(self, text: str) -> float:
        """Detect repetitive content."""
        sentences = re.split(r'[.!?]+', text.lower())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return 0.0
        
        # Check for duplicate sentences
        unique_sentences = set(sentences)
        duplication_ratio = 1.0 - (len(unique_sentences) / len(sentences))
        
        # Check for repeated phrases (3+ words)
        words = text.lower().split()
        phrase_repeats = 0
        
        for i in range(len(words) - 3):
            phrase = ' '.join(words[i:i+3])
            if words.count(words[i]) > 2 and len(words[i]) > 3:
                phrase_repeats += 1
        
        phrase_penalty = min(1.0, phrase_repeats / max(1, len(words) - 3))
        
        return (duplication_ratio + phrase_penalty) / 2
    
    def _count_sections(self, text: str) -> int:
        """Count number of sections in the output."""
        # Count headers
        headers = len(re.findall(r'^#+\\s', text, re.MULTILINE))
        
        # Count paragraph breaks
        paragraphs = len(re.findall(r'\\n\\n+', text))
        
        # Count numbered or bulleted sections
        list_items = len(re.findall(r'^(\\d+\\.|[•\\-\\*])\\s', text, re.MULTILINE))
        
        return max(headers, paragraphs // 2, list_items // 3, 1)


class LLMJudgeEvaluator(BaseEvaluator):
    """
    LLM-as-a-judge evaluation using a separate LLM call to score outputs.
    
    Provides more nuanced evaluation of:
    - Semantic correctness
    - Quality of reasoning
    - Task-specific criteria
    """
    
    def __init__(self, config: EvaluationConfig, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self._cache = {}  # Cache for judge responses
    
    def set_llm_client(self, client):
        """Set the LLM client for judge evaluations."""
        self.llm_client = client
    
    def evaluate(self, output: LLMOutput, context: Dict) -> EvaluationMetrics:
        """Perform LLM-as-judge evaluation."""
        if not self.llm_client:
            raise ValueError("LLM client not configured for judge evaluation")
        
        # Build judge prompt
        judge_prompt = self._build_judge_prompt(output, context)
        
        # Get judge response (with caching)
        cache_key = hash(judge_prompt)
        if cache_key in self._cache:
            judge_response = self._cache[cache_key]
        else:
            judge_response = self._call_judge(judge_prompt)
            self._cache[cache_key] = judge_response
        
        # Parse judge response
        scores = self._parse_judge_response(judge_response)
        
        # Create metrics
        return EvaluationMetrics(
            output_id=output.id,
            correctness_score=scores.get('correctness_score', 0.5),
            format_compliance_score=scores.get('format_compliance_score', 0.5),
            completeness_score=scores.get('completeness_score', 0.5),
            conciseness_score=scores.get('conciseness_score', 0.5),
            output_length=len(output.output_text),
            num_sections=self._count_sections(output.output_text),
            judge_feedback=scores.get('feedback', ''),
            has_required_format=scores.get('format_compliance_score', 0) >= self.config.format_compliance_threshold
        )
    
    def _build_judge_prompt(self, output: LLMOutput, context: Dict) -> str:
        """Build the prompt for the LLM judge."""
        task = context.get('task_description', 'General task')
        expected_format = context.get('expected_format', 'Not specified')
        
        template = self.config.judge_prompt_template
        
        return template.format(
            task=task,
            expected_format=expected_format,
            response=output.output_text
        )
    
    def _call_judge(self, judge_prompt: str) -> str:
        """Call the LLM judge."""
        # This would use the actual LLM client
        # For now, return a placeholder
        return self._generate_mock_judge_response()
    
    def _generate_mock_judge_response(self) -> str:
        """Generate a mock judge response for testing."""
        import random
        
        return json.dumps({
            "correctness_score": round(random.uniform(0.6, 1.0), 2),
            "format_compliance_score": round(random.uniform(0.7, 1.0), 2),
            "completeness_score": round(random.uniform(0.6, 1.0), 2),
            "conciseness_score": round(random.uniform(0.5, 1.0), 2),
            "feedback": "The response demonstrates good understanding with minor areas for improvement."
        })
    
    def _parse_judge_response(self, response: str) -> Dict:
        """Parse the judge's JSON response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\\{[^}]*\\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback: parse as direct JSON
            return json.loads(response)
        
        except (json.JSONDecodeError, AttributeError):
            # Return default scores if parsing fails
            return {
                "correctness_score": 0.5,
                "format_compliance_score": 0.5,
                "completeness_score": 0.5,
                "conciseness_score": 0.5,
                "feedback": "Failed to parse judge response"
            }
    
    def _count_sections(self, text: str) -> int:
        """Count sections in text."""
        headers = len(re.findall(r'^#+\\s', text, re.MULTILINE))
        paragraphs = len(re.findall(r'\\n\\n+', text))
        return max(headers, paragraphs // 2, 1)


class HybridEvaluator(BaseEvaluator):
    """
    Hybrid evaluation combining rule-based and LLM-as-judge approaches.
    
    Uses rule-based for objective metrics and LLM judge for subjective quality.
    """
    
    def __init__(self, config: EvaluationConfig, llm_client=None):
        self.config = config
        self.rule_evaluator = RuleBasedEvaluator(config)
        self.llm_evaluator = LLMJudgeEvaluator(config, llm_client)
        
        # Weight for LLM judge vs rule-based (0.5 = equal weight)
        self.llm_weight = 0.6
        self.rule_weight = 0.4
    
    def set_llm_client(self, client):
        """Set the LLM client for judge evaluations."""
        self.llm_evaluator.set_llm_client(client)
    
    def evaluate(self, output: LLMOutput, context: Dict) -> EvaluationMetrics:
        """Perform hybrid evaluation."""
        # Get rule-based scores
        rule_metrics = self.rule_evaluator.evaluate(output, context)
        
        # Get LLM judge scores (if available)
        try:
            llm_metrics = self.llm_evaluator.evaluate(output, context)
            use_llm = True
        except (ValueError, Exception):
            llm_metrics = None
            use_llm = False
        
        if use_llm and llm_metrics:
            # Combine scores with weighted average
            combined = EvaluationMetrics(
                output_id=output.id,
                correctness_score=(
                    self.rule_weight * rule_metrics.correctness_score +
                    self.llm_weight * llm_metrics.correctness_score
                ),
                format_compliance_score=(
                    self.rule_weight * rule_metrics.format_compliance_score +
                    self.llm_weight * llm_metrics.format_compliance_score
                ),
                completeness_score=(
                    self.rule_weight * rule_metrics.completeness_score +
                    self.llm_weight * llm_metrics.completeness_score
                ),
                conciseness_score=(
                    self.rule_weight * rule_metrics.conciseness_score +
                    self.llm_weight * llm_metrics.conciseness_score
                ),
                output_length=rule_metrics.output_length,
                num_sections=max(rule_metrics.num_sections, llm_metrics.num_sections),
                has_required_format=(
                    rule_metrics.has_required_format or 
                    llm_metrics.has_required_format
                ),
                repetition_ratio=rule_metrics.repetition_ratio,
                judge_feedback=llm_metrics.judge_feedback,
                rule_based_errors=rule_metrics.rule_based_errors
            )
        else:
            # Use only rule-based scores
            combined = rule_metrics
        
        return combined


class RewardCalculator:
    """
    Computes final reward from evaluation metrics.
    
    Formula:
    Reward = Σ(w_i * normalized_metric_i) - penalties
    
    Where:
    - w_i are configurable weights
    - Metrics are normalized using z-score or min-max
    - Penalties include length penalty, repetition penalty, etc.
    """
    
    def __init__(self, reward_config: RewardConfig):
        self.config = reward_config
        self.running_scores = {
            'correctness': [],
            'format_compliance': [],
            'completeness': [],
            'conciseness': []
        }
    
    def compute_reward(
        self,
        metrics: EvaluationMetrics,
        context: Dict = None
    ) -> RewardBreakdown:
        """
        Compute final reward from evaluation metrics.
        
        Returns detailed breakdown for analysis.
        """
        context = context or {}
        
        # Get normalized weights
        weights = self.config.normalize_weights()
        
        # Raw scores
        raw_scores = {
            'correctness': metrics.correctness_score,
            'format_compliance': metrics.format_compliance_score,
            'completeness': metrics.completeness_score,
            'conciseness': metrics.conciseness_score
        }
        
        # Normalize scores (z-score normalization if enabled)
        if self.config.use_z_score_normalization:
            normalized_scores = self._normalize_scores(raw_scores)
        else:
            normalized_scores = raw_scores
        
        # Update running statistics
        for key, score in raw_scores.items():
            self.running_scores[key].append(score)
            if len(self.running_scores[key]) > self.config.running_window_size:
                self.running_scores[key].pop(0)
        
        # Compute weighted sum
        weighted_sum = sum(
            weights[key] * normalized_scores[key]
            for key in weights
        )
        
        # Compute penalties
        penalties = self._compute_penalties(metrics, context)
        total_penalty = sum(penalties.values())
        
        # Final reward
        final_reward = max(0.0, min(1.0, weighted_sum - total_penalty))
        
        return RewardBreakdown(
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
            weights=weights,
            weighted_sum=weighted_sum,
            penalties=penalties,
            final_reward=final_reward
        )
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores using running statistics."""
        normalized = {}
        
        for key, score in scores.items():
            history = self.running_scores.get(key, [])
            
            if len(history) < 2:
                # Not enough data for normalization
                normalized[key] = score
            else:
                mean = sum(history) / len(history)
                std = (sum((x - mean) ** 2 for x in history) / len(history)) ** 0.5
                
                if std > 0.01:  # Avoid division by zero
                    # Z-score normalization, then scale to [0, 1]
                    z_score = (score - mean) / std
                    normalized[key] = 0.5 + 0.5 * z_score  # Map to [0, 1]
                else:
                    normalized[key] = score
        
        # Clamp to [0, 1]
        return {k: max(0.0, min(1.0, v)) for k, v in normalized.items()}
    
    def _compute_penalties(
        self,
        metrics: EvaluationMetrics,
        context: Dict
    ) -> Dict[str, float]:
        """Compute various penalties."""
        penalties = {}
        
        # Length penalty
        if metrics.output_length > self.config.length_penalty_threshold:
            excess = metrics.output_length - self.config.length_penalty_threshold
            penalties['length'] = excess * self.config.length_penalty_factor
        else:
            penalties['length'] = 0.0
        
        # Repetition penalty
        penalties['repetition'] = metrics.repetition_ratio * self.config.repetition_penalty
        
        # Format compliance penalty (if below threshold)
        if not metrics.has_required_format:
            penalties['format_violation'] = 0.2
        
        # Rule-based errors penalty
        if metrics.rule_based_errors:
            penalties['rule_errors'] = len(metrics.rule_based_errors) * 0.1
        
        return penalties
    
    def reset_statistics(self):
        """Reset running statistics for normalization."""
        self.running_scores = {key: [] for key in self.running_scores}


class EvaluationEngine:
    """
    Main evaluation engine that orchestrates all evaluators.
    
    Supports:
    - Single and batch evaluation
    - Multiple evaluation methods
    - Reward computation
    """
    
    def __init__(self, eval_config: EvaluationConfig, reward_config: RewardConfig):
        self.eval_config = eval_config
        self.reward_config = reward_config
        
        # Initialize evaluator based on method
        if eval_config.method == 'rule_based':
            self.evaluator = RuleBasedEvaluator(eval_config)
        elif eval_config.method == 'llm_judge':
            self.evaluator = LLMJudgeEvaluator(eval_config)
        else:  # hybrid
            self.evaluator = HybridEvaluator(eval_config)
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(reward_config)
    
    def set_llm_client(self, client):
        """Set LLM client for judge evaluations."""
        if hasattr(self.evaluator, 'set_llm_client'):
            self.evaluator.set_llm_client(client)
    
    def evaluate(
        self,
        output: LLMOutput,
        context: Dict = None
    ) -> Tuple[EvaluationMetrics, float]:
        """
        Evaluate a single output and compute reward.
        
        Returns:
            Tuple of (metrics, reward)
        """
        context = context or {}
        
        # Get metrics from evaluator
        metrics = self.evaluator.evaluate(output, context)
        
        # Compute reward
        reward_breakdown = self.reward_calculator.compute_reward(metrics, context)
        metrics.final_reward = reward_breakdown.final_reward
        
        return metrics, reward_breakdown.final_reward
    
    def evaluate_batch(
        self,
        outputs: List[LLMOutput],
        contexts: List[Dict] = None
    ) -> List[Tuple[EvaluationMetrics, float]]:
        """
        Evaluate multiple outputs in batch.
        
        Args:
            outputs: List of LLMOutput objects
            contexts: Optional list of context dicts (one per output)
        
        Returns:
            List of (metrics, reward) tuples
        """
        contexts = contexts or [{} for _ in outputs]
        
        results = []
        for output, context in zip(outputs, contexts):
            result = self.evaluate(output, context)
            results.append(result)
        
        return results
    
    def get_evaluation_statistics(self) -> Dict:
        """Get statistics about evaluation history."""
        running_scores = self.reward_calculator.running_scores
        
        stats = {}
        for key, scores in running_scores.items():
            if scores:
                stats[key] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        return stats
