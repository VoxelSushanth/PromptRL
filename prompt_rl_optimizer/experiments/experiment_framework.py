"""
Experiment Framework for Prompt RL Optimization.

Provides:
- Pre-configured experiments
- Statistical comparison tools
- Visualization utilities
- Ablation study framework
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from configs.experiment_config import ExperimentConfig, ExplorationStrategy
from core.optimizer import PromptOptimizer
from core.models import ExperimentResult


class ExperimentRunner:
    """
    Runs and manages prompt optimization experiments.
    
    Supports:
    - Multiple task types
    - Baseline comparisons
    - Ablation studies
    - Statistical analysis
    """
    
    def __init__(self, base_config: ExperimentConfig = None):
        self.base_config = base_config or ExperimentConfig()
        self.results: List[ExperimentResult] = []
        self.experiments_dir = Path("experiments/results")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
    
    def run_experiment(
        self,
        task: str,
        base_prompt: str,
        config_overrides: Dict = None,
        num_iterations: int = 50,
        run_name: str = None
    ) -> ExperimentResult:
        """
        Run a single optimization experiment.
        
        Args:
            task: Task description
            base_prompt: Initial prompt
            config_overrides: Configuration overrides
            num_iterations: Number of iterations
            run_name: Optional name for the run
        
        Returns:
            ExperimentResult with optimization results
        """
        # Create config with overrides
        config = self._create_config(config_overrides)
        config.experiment_name = run_name or f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize optimizer
        optimizer = PromptOptimizer(config)
        
        # Run optimization
        result = optimizer.optimize(
            task=task,
            base_prompt=base_prompt,
            num_iterations=num_iterations
        )
        
        self.results.append(result)
        self._save_result(result)
        
        return result
    
    def run_comparison_experiment(
        self,
        task: str,
        base_prompt: str,
        strategies: List[ExplorationStrategy] = None,
        num_iterations: int = 50,
        runs_per_strategy: int = 3
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Compare multiple RL strategies on the same task.
        
        Args:
            task: Task description
            base_prompt: Initial prompt
            strategies: RL strategies to compare
            num_iterations: Iterations per run
            runs_per_strategy: Number of runs per strategy (for statistical significance)
        
        Returns:
            Dictionary mapping strategy names to list of results
        """
        strategies = strategies or [
            ExplorationStrategy.EPSILON_GREEDY,
            ExplorationStrategy.UCB1,
            ExplorationStrategy.THOMPSON_SAMPLING
        ]
        
        comparison_results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"Running experiments with strategy: {strategy.value}")
            print(f"{'='*60}\n")
            
            strategy_results = []
            
            for run in range(runs_per_strategy):
                config_overrides = {
                    'rl': {
                        'strategy': strategy.value
                    }
                }
                
                result = self.run_experiment(
                    task=task,
                    base_prompt=base_prompt,
                    config_overrides=config_overrides,
                    num_iterations=num_iterations,
                    run_name=f"{task}_{strategy.value}_run{run}"
                )
                
                strategy_results.append(result)
            
            comparison_results[strategy.value] = strategy_results
        
        return comparison_results
    
    def run_ablation_study(
        self,
        task: str,
        base_prompt: str,
        num_iterations: int = 50
    ) -> Dict[str, ExperimentResult]:
        """
        Run ablation study to analyze component contributions.
        
        Variants:
        - Full system (with RL)
        - No RL (random selection)
        - No exploration (pure exploitation)
        - Uniform weights
        - Single strategy only
        """
        ablation_results = {}
        
        # Full system
        print("\n[Ablation] Running full system...")
        full_result = self.run_experiment(
            task=task,
            base_prompt=base_prompt,
            num_iterations=num_iterations,
            run_name=f"{task}_full"
        )
        ablation_results['full'] = full_result
        
        # No RL variant
        print("\n[Ablation] Running without RL...")
        no_rl_config = {
            'rl': {
                'strategy': 'epsilon_greedy',
                'epsilon_start': 1.0,
                'epsilon_end': 1.0,  # Always explore
                'epsilon_decay': 1.0
            }
        }
        no_rl_result = self.run_experiment(
            task=task,
            base_prompt=base_prompt,
            config_overrides=no_rl_config,
            num_iterations=num_iterations,
            run_name=f"{task}_no_rl"
        )
        ablation_results['no_rl'] = no_rl_result
        
        # No exploration variant
        print("\n[Ablation] Running without exploration...")
        no_explore_config = {
            'rl': {
                'strategy': 'epsilon_greedy',
                'epsilon_start': 0.0,
                'epsilon_end': 0.0
            }
        }
        no_explore_result = self.run_experiment(
            task=task,
            base_prompt=base_prompt,
            config_overrides=no_explore_config,
            num_iterations=num_iterations,
            run_name=f"{task}_no_exploration"
        )
        ablation_results['no_exploration'] = no_explore_result
        
        return ablation_results
    
    def _create_config(self, overrides: Dict = None) -> ExperimentConfig:
        """Create configuration with optional overrides."""
        if not overrides:
            return ExperimentConfig(**self.base_config.model_dump())
        
        # Deep merge
        config_dict = self.base_config.model_dump()
        self._deep_update(config_dict, overrides)
        
        return ExperimentConfig(**config_dict)
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        """Recursively update dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _save_result(self, result: ExperimentResult) -> None:
        """Save experiment result to file."""
        filename = self.experiments_dir / f"{result.experiment_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
    
    def compute_statistics(
        self,
        results: List[ExperimentResult]
    ) -> Dict[str, float]:
        """
        Compute statistical measures across multiple results.
        
        Returns:
            Dictionary with mean, std, min, max, confidence intervals
        """
        rewards = [r.best_reward for r in results]
        improvements = [r.improvement_over_baseline for r in results]
        
        return {
            'reward': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'ci_95': 1.96 * np.std(rewards) / np.sqrt(len(rewards))
            },
            'improvement': {
                'mean': np.mean(improvements),
                'std': np.std(improvements),
                'min': np.min(improvements),
                'max': np.max(improvements),
                'ci_95': 1.96 * np.std(improvements) / np.sqrt(len(improvements))
            },
            'sample_size': len(results)
        }
    
    def paired_t_test(
        self,
        results_a: List[ExperimentResult],
        results_b: List[ExperimentResult]
    ) -> Dict[str, float]:
        """
        Perform paired t-test between two sets of results.
        
        Returns:
            t-statistic, p-value, and interpretation
        """
        from scipy import stats
        
        rewards_a = [r.best_reward for r in results_a]
        rewards_b = [r.best_reward for r in results_b]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(rewards_a, rewards_b)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(rewards_a) - np.mean(rewards_b)
        pooled_std = np.sqrt((np.std(rewards_a)**2 + np.std(rewards_b)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'effect_size': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small')
        }


class BenchmarkTasks:
    """
    Pre-defined benchmark tasks for evaluation.
    
    Each task includes:
    - Task description
    - Base prompt
    - Test dataset
    - Expected output format
    - Evaluation criteria
    """
    
    MATH_REASONING = {
        'task': 'math_word_problems',
        'base_prompt': 'Solve the following math word problem:',
        'description': 'Multi-step mathematical reasoning problems',
        'test_data': [
            {'problem': 'A store sells apples for $2 each and oranges for $3 each. If you buy 5 apples and 3 oranges, how much do you spend?'},
            {'problem': 'If a car travels at 60 mph for 2 hours and then at 40 mph for 3 hours, what is the total distance traveled?'},
            {'problem': 'What is 25% of 80, plus 15% of 60?'},
            {'problem': 'The sum of three consecutive integers is 72. What is the largest integer?'},
            {'problem': 'If x + 2y = 10 and 2x + y = 11, what is the value of x + y?'}
        ],
        'expected_format': 'Step-by-step solution with final answer clearly marked',
        'evaluation_weights': {
            'correctness': 0.5,
            'format_compliance': 0.2,
            'completeness': 0.2,
            'conciseness': 0.1
        }
    }
    
    CODE_GENERATION = {
        'task': 'code_generation',
        'base_prompt': 'Write Python code to accomplish the following:',
        'description': 'Python code generation tasks',
        'test_data': [
            {'requirement': 'Write a function that takes a list of numbers and returns the second largest unique value'},
            {'requirement': 'Create a class representing a Stack with push, pop, and peek operations'},
            {'requirement': 'Write a function to check if a string is a palindrome (ignoring spaces and case)'},
            {'requirement': 'Implement a decorator that measures and prints the execution time of a function'},
            {'requirement': 'Write a function to flatten a nested list of arbitrary depth'}
        ],
        'expected_format': 'Working Python code with comments explaining the approach',
        'evaluation_weights': {
            'correctness': 0.4,
            'format_compliance': 0.2,
            'completeness': 0.25,
            'conciseness': 0.15
        }
    }
    
    TEXT_CLASSIFICATION = {
        'task': 'text_classification',
        'base_prompt': 'Classify the following text into one of these categories:',
        'description': 'Text classification with explanation',
        'test_data': [
            {'text': 'The new smartphone features an impressive camera system and all-day battery life.', 'categories': ['Technology', 'Sports', 'Politics', 'Entertainment']},
            {'text': 'The team secured a dramatic victory in the final minutes of the championship game.', 'categories': ['Technology', 'Sports', 'Politics', 'Entertainment']},
            {'text': 'Legislators debated the proposed bill for hours before reaching a compromise.', 'categories': ['Technology', 'Sports', 'Politics', 'Entertainment']},
            {'text': 'The highly anticipated sequel broke box office records on its opening weekend.', 'categories': ['Technology', 'Sports', 'Politics', 'Entertainment']},
            {'text': 'Scientists discovered a new species of deep-sea fish in the Pacific Ocean.', 'categories': ['Science', 'Business', 'Health', 'Environment']}
        ],
        'expected_format': 'Category label with brief justification',
        'evaluation_weights': {
            'correctness': 0.45,
            'format_compliance': 0.25,
            'completeness': 0.2,
            'conciseness': 0.1
        }
    }
    
    SUMMARIZATION = {
        'task': 'text_summarization',
        'base_prompt': 'Summarize the following text in 2-3 sentences:',
        'description': 'Abstractive text summarization',
        'test_data': [
            {'text': 'Climate change continues to impact global weather patterns. Recent studies show increasing frequency of extreme weather events including hurricanes, droughts, and floods. Scientists warn that without significant reductions in greenhouse gas emissions, these trends will accelerate. Governments worldwide are implementing various policies to address the crisis, though progress remains slow.'},
            {'text': 'The tech industry has seen remarkable growth in artificial intelligence applications. Major companies are investing billions in AI research and development. New applications range from healthcare diagnostics to autonomous vehicles. However, concerns about job displacement and ethical implications continue to spark debate among experts.'},
            {'text': 'Global supply chains face ongoing challenges from geopolitical tensions and pandemic aftermath. Manufacturing delays and shipping bottlenecks have led to product shortages and price increases. Companies are reevaluating their supply chain strategies, with some considering reshoring production to reduce dependencies.'}
        ],
        'expected_format': 'Concise 2-3 sentence summary capturing main points',
        'evaluation_weights': {
            'correctness': 0.35,
            'format_compliance': 0.25,
            'completeness': 0.25,
            'conciseness': 0.15
        }
    }
    
    @classmethod
    def get_all_tasks(cls) -> Dict[str, Dict]:
        """Get all benchmark tasks."""
        return {
            'math_reasoning': cls.MATH_REASONING,
            'code_generation': cls.CODE_GENERATION,
            'text_classification': cls.TEXT_CLASSIFICATION,
            'summarization': cls.SUMMARIZATION
        }
    
    @classmethod
    def get_task(cls, name: str) -> Dict:
        """Get a specific task by name."""
        tasks = cls.get_all_tasks()
        
        for task_name, task_data in tasks.items():
            if name.lower() in task_name or task_name in name.lower():
                return task_data
        
        raise ValueError(f"Unknown task: {name}. Available tasks: {list(tasks.keys())}")


def run_demo_experiment():
    """Run a demonstration experiment."""
    print("="*60)
    print("PROMPT OPTIMIZATION VIA REINFORCEMENT LEARNING")
    print("Demonstration Experiment")
    print("="*60)
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Get math reasoning task
    task_info = BenchmarkTasks.get_task('math_reasoning')
    
    print(f"\nTask: {task_info['task']}")
    print(f"Description: {task_info['description']}")
    print(f"Base Prompt: {task_info['base_prompt']}")
    
    # Run optimization
    result = runner.run_experiment(
        task=task_info['task'],
        base_prompt=task_info['base_prompt'],
        num_iterations=30,
        run_name="demo_math_optimization"
    )
    
    # Print results
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS")
    print("="*60)
    print(f"\nExperiment ID: {result.experiment_id}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Total Trials: {result.total_trials}")
    print(f"\nBest Reward: {result.best_reward:.4f}")
    print(f"Improvement over Baseline: {result.improvement_over_baseline:.2f}%")
    
    print(f"\nOptimized Prompt:")
    print("-"*40)
    print(result.best_prompt.variant_prompt[:500] + "..." if len(result.best_prompt.variant_prompt) > 500 else result.best_prompt.variant_prompt)
    print("-"*40)
    
    if result.ablation_results:
        print("\nAblation Study Results:")
        for variant, data in result.ablation_results.items():
            print(f"  {variant}: {data['score']:.4f} - {data['description']}")
    
    return result


if __name__ == "__main__":
    run_demo_experiment()
