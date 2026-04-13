"""
Core data models for Prompt RL Optimizer.

This module defines all the core data structures used throughout the system.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import uuid


class PromptVariant(BaseModel):
    """Represents a candidate prompt variant."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    base_prompt: str
    variant_prompt: str
    generation_strategy: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Performance tracking
    total_trials: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0)
    average_reward: float = Field(default=0.0)
    
    def update_performance(self, reward: float) -> None:
        """Update performance metrics after a trial."""
        self.total_trials += 1
        self.cumulative_reward += reward
        self.average_reward = self.cumulative_reward / self.total_trials


class LLMOutput(BaseModel):
    """Represents an LLM output with metadata."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt_id: str
    prompt_text: str
    input_data: Dict[str, Any]
    output_text: str
    model_name: str
    tokens_used: int
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)


class EvaluationMetrics(BaseModel):
    """Detailed evaluation metrics for an LLM output."""
    
    output_id: str
    
    # Individual scores (0.0 to 1.0)
    correctness_score: float = Field(ge=0, le=1)
    format_compliance_score: float = Field(ge=0, le=1)
    completeness_score: float = Field(ge=0, le=1)
    conciseness_score: float = Field(ge=0, le=1)
    
    # Raw metrics
    output_length: int
    num_sections: int = Field(default=0)
    has_required_format: bool = Field(default=False)
    repetition_ratio: float = Field(default=0.0)
    
    # Judge feedback
    judge_feedback: Optional[str] = None
    rule_based_errors: List[str] = Field(default_factory=list)
    
    # Computed reward
    final_reward: float = Field(default=0.0)
    
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class Trial(BaseModel):
    """Represents a single optimization trial."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int
    prompt: PromptVariant
    output: LLMOutput
    metrics: EvaluationMetrics
    
    # Context
    task_description: str
    test_input: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OptimizationState(BaseModel):
    """Tracks the state of the optimization process."""
    
    experiment_id: str
    current_iteration: int = Field(default=0)
    total_iterations: int
    
    # Population
    prompts: List[PromptVariant] = Field(default_factory=list)
    elite_prompts: List[str] = Field(default_factory=list)  # IDs of elite prompts
    
    # Best results
    best_prompt_id: Optional[str] = None
    best_reward: float = Field(default=0.0)
    
    # History
    reward_history: List[float] = Field(default_factory=list)
    convergence_metrics: Dict[str, float] = Field(default_factory=dict)
    
    # RL state
    exploration_rate: float = Field(default=0.9)
    action_values: Dict[str, float] = Field(default_factory=dict)
    action_counts: Dict[str, int] = Field(default_factory=dict)
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def is_converged(self, window: int = 10, threshold: float = 0.01) -> bool:
        """Check if optimization has converged."""
        if len(self.reward_history) < window:
            return False
        
        recent_rewards = self.reward_history[-window:]
        variance = sum((r - sum(recent_rewards)/len(recent_rewards))**2 
                      for r in recent_rewards) / len(recent_rewards)
        
        return variance < threshold
    
    def get_best_prompt(self) -> Optional[PromptVariant]:
        """Get the best performing prompt."""
        if not self.best_prompt_id:
            return None
        
        for prompt in self.prompts:
            if prompt.id == self.best_prompt_id:
                return prompt
        
        return None


class ExperimentResult(BaseModel):
    """Results from a complete optimization experiment."""
    
    experiment_id: str
    experiment_name: str
    task_description: str
    
    # Final results
    best_prompt: PromptVariant
    best_reward: float
    improvement_over_baseline: float
    
    # Statistics
    total_iterations: int
    total_trials: int
    convergence_iteration: Optional[int] = None
    
    # Comparison with baselines
    baseline_scores: Dict[str, float] = Field(default_factory=dict)
    statistical_significance: Optional[Dict[str, float]] = None
    
    # Ablation study results
    ablation_results: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class StrategyType(str, Enum):
    """Types of prompt generation strategies."""
    
    INSTRUCTION_CLARITY = "instruction_clarity"
    CONSTRAINT_INJECTION = "constraint_injection"
    OUTPUT_STRUCTURING = "output_structuring"
    TASK_DECOMPOSITION = "task_decomposition"
    FEW_SHOT_EXAMPLES = "few_shot_examples"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_PLAYING = "role_playing"
    NEGATIVE_CONSTRAINTS = "negative_constraints"


class RewardBreakdown(BaseModel):
    """Detailed breakdown of reward computation."""
    
    raw_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    weights: Dict[str, float]
    weighted_sum: float
    penalties: Dict[str, float]
    final_reward: float
    
    def compute(self, config) -> float:
        """Compute final reward from breakdown."""
        return self.final_reward
