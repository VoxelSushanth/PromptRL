"""
Configuration schemas and default settings for Prompt RL Optimizer.

This module defines all configuration options using Pydantic for validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal
from enum import Enum
import yaml
from pathlib import Path


class ExplorationStrategy(str, Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB1 = "ucb1"
    THOMPSON_SAMPLING = "thompson_sampling"
    PPO_LITE = "ppo_lite"


class EvaluationMethod(str, Enum):
    RULE_BASED = "rule_based"
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class RewardConfig(BaseModel):
    """Configuration for reward function weights and penalties."""
    
    # Metric weights (must sum to 1.0)
    correctness_weight: float = Field(default=0.4, ge=0, le=1)
    format_compliance_weight: float = Field(default=0.2, ge=0, le=1)
    completeness_weight: float = Field(default=0.25, ge=0, le=1)
    conciseness_weight: float = Field(default=0.15, ge=0, le=1)
    
    # Penalties
    length_penalty_threshold: int = Field(default=1000, description="Max tokens before penalty")
    length_penalty_factor: float = Field(default=0.001, description="Penalty per token over threshold")
    repetition_penalty: float = Field(default=0.1, description="Penalty for repeated phrases")
    
    # Normalization
    use_z_score_normalization: bool = Field(default=True)
    running_window_size: int = Field(default=50, description="Window for running statistics")
    
    @validator('correctness_weight', 'format_compliance_weight', 
               'completeness_weight', 'conciseness_weight')
    def validate_weights(cls, v):
        if v < 0 or v > 1:
            raise ValueError("Weights must be between 0 and 1")
        return v
    
    def normalize_weights(self) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = (self.correctness_weight + self.format_compliance_weight + 
                self.completeness_weight + self.conciseness_weight)
        if total == 0:
            return {
                'correctness': 0.25,
                'format_compliance': 0.25,
                'completeness': 0.25,
                'conciseness': 0.25
            }
        return {
            'correctness': self.correctness_weight / total,
            'format_compliance': self.format_compliance_weight / total,
            'completeness': self.completeness_weight / total,
            'conciseness': self.conciseness_weight / total
        }


class GeneratorConfig(BaseModel):
    """Configuration for prompt generation strategies."""
    
    # Generation strategies
    use_instruction_clarity: bool = Field(default=True)
    use_constraint_injection: bool = Field(default=True)
    use_output_structuring: bool = Field(default=True)
    use_task_decomposition: bool = Field(default=True)
    
    # Variation parameters
    mutation_rate: float = Field(default=0.3, ge=0, le=1)
    crossover_rate: float = Field(default=0.2, ge=0, le=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    
    # Prompt templates
    clarity_templates: List[str] = Field(default_factory=lambda: [
        "Clearly and step-by-step {task}",
        "Break down {task} into logical steps",
        "Provide a detailed explanation for {task}"
    ])
    
    constraint_templates: List[str] = Field(default_factory=lambda: [
        "Ensure your answer is {constraint}",
        "Your response must follow these rules: {constraint}",
        "Adhere strictly to: {constraint}"
    ])
    
    structure_templates: List[str] = Field(default_factory=lambda: [
        "Format your output as: {format}",
        "Use the following structure: {format}",
        "Organize your response with: {format}"
    ])


class ExecutionConfig(BaseModel):
    """Configuration for LLM execution engine."""
    
    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    model_name: str = Field(default="gpt-4o-mini")
    api_key_env_var: str = Field(default="OPENAI_API_KEY")
    
    # Generation parameters
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    
    # Rate limiting
    requests_per_minute: int = Field(default=60)
    tokens_per_minute: int = Field(default=100000)
    
    # Batch execution
    batch_size: int = Field(default=10)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation engine."""
    
    method: EvaluationMethod = Field(default=EvaluationMethod.HYBRID)
    
    # Rule-based evaluation
    expected_format: Optional[str] = Field(default=None)
    required_keywords: List[str] = Field(default_factory=list)
    forbidden_keywords: List[str] = Field(default_factory=list)
    
    # LLM-as-judge
    judge_model: str = Field(default="gpt-4o-mini")
    judge_prompt_template: str = Field(default="""
Evaluate the following response based on these criteria:

Task: {task}
Expected Format: {expected_format}

Response:
{response}

Score each criterion from 0.0 to 1.0:
1. Correctness: Is the answer factually accurate?
2. Format Compliance: Does it follow the expected format?
3. Completeness: Does it address all aspects of the task?
4. Conciseness: Is it appropriately concise without losing important information?

Provide scores in JSON format:
{{
    "correctness_score": 0.0-1.0,
    "format_compliance_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "conciseness_score": 0.0-1.0,
    "feedback": "Brief explanation of scores"
}}
""")
    
    # Scoring thresholds
    correctness_threshold: float = Field(default=0.7)
    format_compliance_threshold: float = Field(default=0.8)


class RLConfig(BaseModel):
    """Configuration for reinforcement learning agent."""
    
    strategy: ExplorationStrategy = Field(default=ExplorationStrategy.EPSILON_GREEDY)
    
    # Epsilon-greedy parameters
    epsilon_start: float = Field(default=0.9, ge=0, le=1)
    epsilon_end: float = Field(default=0.1, ge=0, le=1)
    epsilon_decay: float = Field(default=0.995, ge=0, le=1)
    
    # UCB1 parameters
    ucb_exploration_constant: float = Field(default=2.0, ge=0)
    
    # Thompson Sampling parameters
    prior_alpha: float = Field(default=1.0, gt=0)
    prior_beta: float = Field(default=1.0, gt=0)
    
    # PPO-lite parameters
    ppo_clip_epsilon: float = Field(default=0.2, ge=0, le=1)
    ppo_learning_rate: float = Field(default=0.001, gt=0)
    ppo_entropy_coefficient: float = Field(default=0.01, ge=0)
    
    # Convergence tracking
    convergence_window: int = Field(default=10)
    convergence_threshold: float = Field(default=0.01)
    early_stopping_patience: int = Field(default=20)
    
    # Population management
    population_size: int = Field(default=20)
    elite_size: int = Field(default=3)
    tournament_size: int = Field(default=5)


class StorageConfig(BaseModel):
    """Configuration for memory and logging system."""
    
    storage_type: Literal["sqlite", "json", "memory"] = Field(default="sqlite")
    
    # SQLite configuration
    database_path: str = Field(default="data/prompt_optimizer.db")
    
    # JSON configuration
    log_directory: str = Field(default="logs")
    
    # Retention policy
    max_trials_to_keep: int = Field(default=10000)
    checkpoint_interval: int = Field(default=10)
    
    # Reproducibility
    random_seed: int = Field(default=42)
    save_all_prompts: bool = Field(default=True)
    save_all_outputs: bool = Field(default=True)


class ExperimentConfig(BaseModel):
    """Top-level configuration for experiments."""
    
    experiment_name: str = Field(default="prompt_optimization_run")
    task_description: str = Field(default="")
    
    # Component configurations
    reward: RewardConfig = Field(default_factory=RewardConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Optimization loop
    num_iterations: int = Field(default=100)
    candidates_per_iteration: int = Field(default=5)
    
    # Baseline comparison
    use_baseline: bool = Field(default=True)
    baseline_prompts: List[str] = Field(default_factory=list)
    
    # Ablation studies
    run_ablation: bool = Field(default=True)
    ablation_variants: List[str] = Field(default_factory=lambda: [
        "no_rl",
        "no_exploration",
        "uniform_weights"
    ])
    
    @classmethod
    def load(cls, config_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()
