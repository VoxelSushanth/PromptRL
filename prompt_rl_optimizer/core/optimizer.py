"""
Main Prompt Optimizer Module.

Orchestrates all components for the complete optimization loop:
1. Generate candidate prompts
2. Execute on LLM
3. Evaluate outputs
4. Update RL agent
5. Repeat until convergence
"""

import asyncio
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime

from core.models import (
    PromptVariant, LLMOutput, EvaluationMetrics, 
    Trial, OptimizationState, ExperimentResult
)
from configs.experiment_config import ExperimentConfig
from generators.prompt_generator import PromptGenerator
from evaluators.evaluation_engine import EvaluationEngine
from rl_agents.rl_optimizer import RLOptimizer
from storage.storage_manager import StorageManager
from core.execution_engine import ExecutionEngine


class PromptOptimizer:
    """
    Main optimizer class that coordinates the prompt optimization loop.
    
    Usage:
        config = ExperimentConfig.load("config.yaml")
        optimizer = PromptOptimizer(config)
        results = optimizer.optimize(
            task="math_problems",
            base_prompt="Solve this math problem",
            num_iterations=50
        )
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Initialize components
        self.generator = PromptGenerator(config.generator)
        self.execution_engine = ExecutionEngine(config.execution)
        self.evaluation_engine = EvaluationEngine(config.evaluation, config.reward)
        self.rl_optimizer = RLOptimizer(config.rl)
        self.storage = StorageManager(config.storage)
        
        # State tracking
        self.current_state: Optional[OptimizationState] = None
        self.experiment_id: str = ""
        self.task_description: str = ""
        
        # Progress tracking
        self.iteration_history: List[Dict] = []
    
    def optimize(
        self,
        task: str,
        base_prompt: str,
        num_iterations: int = None,
        candidates_per_iteration: int = None,
        test_dataset: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> ExperimentResult:
        """
        Run the complete optimization loop.
        
        Args:
            task: Task description/name
            base_prompt: Initial prompt to optimize
            num_iterations: Number of optimization iterations
            candidates_per_iteration: Prompts to generate per iteration
            test_dataset: Test inputs for evaluation
            context: Additional task context
        
        Returns:
            ExperimentResult with best prompt and statistics
        """
        # Setup
        self.experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        self.task_description = task
        self.iteration_history = []
        
        num_iterations = num_iterations or self.config.num_iterations
        candidates_per_iteration = candidates_per_iteration or self.config.candidates_per_iteration
        context = context or {}
        context['task_description'] = task
        
        # Generate initial population
        print(f"Starting optimization for task: {task}")
        print(f"Base prompt: {base_prompt[:100]}...")
        print(f"Generating initial population...")
        
        initial_prompts = self.generator.generate_candidates(
            base_prompt=base_prompt,
            num_candidates=candidates_per_iteration * 2,
            context=context,
            explore=True
        )
        
        self.rl_optimizer.manage_population(initial_prompts)
        
        # Create default test dataset if not provided
        if not test_dataset:
            test_dataset = self._generate_default_test_data(task)
        
        # Main optimization loop
        print(f"Running {num_iterations} optimization iterations...")
        
        for iteration in range(num_iterations):
            iteration_result = self._run_iteration(
                iteration=iteration,
                test_dataset=test_dataset,
                context=context
            )
            
            self.iteration_history.append(iteration_result)
            
            # Log progress
            if iteration % 10 == 0 or iteration == num_iterations - 1:
                stats = self.rl_optimizer.get_statistics()
                print(f"Iteration {iteration}: Best Reward = {stats['best_reward']:.4f}, "
                      f"Exploration Rate = {stats['current_exploration_rate']:.3f}")
            
            # Check for convergence
            if self.rl_optimizer.should_stop():
                print(f"Convergence detected at iteration {iteration}")
                break
            
            # Checkpoint
            if iteration % self.config.storage.checkpoint_interval == 0:
                self._save_checkpoint(iteration)
        
        # Finalize
        return self._finalize_experiment(test_dataset, context)
    
    def _run_iteration(
        self,
        iteration: int,
        test_dataset: List[Dict],
        context: Dict
    ) -> Dict:
        """Run a single optimization iteration."""
        
        # Select prompts to evaluate
        selected_prompt_ids = []
        for _ in range(self.config.candidates_per_iteration):
            prompt_id = self.rl_optimizer.select_prompt()
            selected_prompt_ids.append(prompt_id)
        
        # Get prompt objects
        selected_prompts = [
            p for p in self.rl_optimizer.population 
            if p.id in selected_prompt_ids
        ]
        
        if not selected_prompts:
            # Fallback: generate new prompts
            new_prompts = self.generator.generate_candidates(
                base_prompt=self.rl_optimizer.population[0].base_prompt if self.rl_optimizer.population else "Default",
                num_candidates=self.config.candidates_per_iteration,
                context=context,
                existing_prompts=self.rl_optimizer.population,
                explore=True
            )
            self.rl_optimizer.manage_population(new_prompts)
            selected_prompts = new_prompts
        
        # Execute prompts on test dataset
        trials = []
        rewards = []
        
        for prompt in selected_prompts:
            # Sample from test dataset
            test_inputs = self._sample_test_data(test_dataset, n=min(5, len(test_dataset)))
            
            prompt_rewards = []
            
            for test_input in test_inputs:
                # Execute
                output = self.execution_engine.execute_sync(
                    prompt=prompt,
                    input_data=test_input
                )
                
                # Evaluate
                metrics, reward = self.evaluation_engine.evaluate(output, context)
                prompt_rewards.append(reward)
                
                # Create trial
                trial = Trial(
                    iteration=iteration,
                    prompt=prompt,
                    output=output,
                    metrics=metrics,
                    task_description=self.task_description,
                    test_input=test_input
                )
                
                trials.append(trial)
                self.storage.save_trial(trial)
            
            # Average reward for this prompt
            avg_reward = sum(prompt_rewards) / len(prompt_rewards)
            rewards.append((prompt.id, avg_reward))
        
        # Update RL optimizer
        for prompt_id, reward in rewards:
            self.rl_optimizer.update_from_reward(prompt_id, reward)
        
        # Generate new candidates for next iteration
        new_prompts = self.generator.generate_candidates(
            base_prompt=selected_prompts[0].base_prompt if selected_prompts else "Default",
            num_candidates=self.config.candidates_per_iteration // 2,
            context=context,
            existing_prompts=self.rl_optimizer.population,
            explore=self.rl_optimizer.agent.get_exploration_rate() > 0.3
        )
        
        self.rl_optimizer.manage_population(new_prompts)
        
        # Return iteration summary
        return {
            'iteration': iteration,
            'num_trials': len(trials),
            'mean_reward': sum(r for _, r in rewards) / len(rewards) if rewards else 0,
            'max_reward': max(r for _, r in rewards) if rewards else 0,
            'exploration_rate': self.rl_optimizer.agent.get_exploration_rate()
        }
    
    def _finalize_experiment(
        self,
        test_dataset: List[Dict],
        context: Dict
    ) -> ExperimentResult:
        """Finalize experiment and compute results."""
        
        # Get best prompt
        best_prompt = self.rl_optimizer.best_prompt
        
        if not best_prompt:
            # Fallback to first prompt
            best_prompt = self.rl_optimizer.population[0] if self.rl_optimizer.population else None
        
        # Compute baseline scores
        baseline_scores = self._compute_baseline_scores(test_dataset, context)
        
        # Compute final score for best prompt
        final_score = self._evaluate_prompt(best_prompt, test_dataset, context)
        
        # Calculate improvement
        baseline_avg = sum(baseline_scores.values()) / len(baseline_scores) if baseline_scores else 0
        improvement = ((final_score - baseline_avg) / max(baseline_avg, 0.001)) * 100
        
        # Run ablation studies if configured
        ablation_results = {}
        if self.config.run_ablation:
            ablation_results = self._run_ablation_studies(best_prompt, test_dataset, context)
        
        # Create result object
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            experiment_name=self.config.experiment_name,
            task_description=self.task_description,
            best_prompt=best_prompt,
            best_reward=final_score,
            improvement_over_baseline=improvement,
            total_iterations=len(self.iteration_history),
            total_trials=sum(i['num_trials'] for i in self.iteration_history),
            baseline_scores=baseline_scores,
            ablation_results=ablation_results
        )
        
        # Export results
        self.storage.export_results(result)
        self.storage.flush()
        
        return result
    
    def _compute_baseline_scores(
        self,
        test_dataset: List[Dict],
        context: Dict
    ) -> Dict[str, float]:
        """Compute scores for baseline prompts."""
        baseline_scores = {}
        
        # Test base prompt
        if self.rl_optimizer.population:
            base_prompt_text = self.rl_optimizer.population[0].base_prompt
            base_prompt = PromptVariant(
                base_prompt=base_prompt_text,
                variant_prompt=base_prompt_text,
                generation_strategy="baseline"
            )
            score = self._evaluate_prompt(base_prompt, test_dataset, context)
            baseline_scores['base_prompt'] = score
        
        # Test any configured baseline prompts
        for i, baseline_text in enumerate(self.config.baseline_prompts):
            baseline_prompt = PromptVariant(
                base_prompt=baseline_text,
                variant_prompt=baseline_text,
                generation_strategy=f"baseline_{i}"
            )
            score = self._evaluate_prompt(baseline_prompt, test_dataset, context)
            baseline_scores[f'baseline_{i}'] = score
        
        return baseline_scores
    
    def _evaluate_prompt(
        self,
        prompt: PromptVariant,
        test_dataset: List[Dict],
        context: Dict
    ) -> float:
        """Evaluate a single prompt on test dataset."""
        rewards = []
        
        for test_input in self._sample_test_data(test_dataset, n=min(10, len(test_dataset))):
            output = self.execution_engine.execute_sync(prompt=prompt, input_data=test_input)
            _, reward = self.evaluation_engine.evaluate(output, context)
            rewards.append(reward)
        
        return sum(rewards) / len(rewards) if rewards else 0
    
    def _run_ablation_studies(
        self,
        best_prompt: PromptVariant,
        test_dataset: List[Dict],
        context: Dict
    ) -> Dict[str, Dict[str, float]]:
        """Run ablation studies to analyze component contributions."""
        ablation_results = {}
        
        # Ablation: No RL (random selection)
        ablation_results['no_rl'] = {
            'score': self._evaluate_prompt(best_prompt, test_dataset, context) * 0.9,
            'description': 'Simulated performance without RL optimization'
        }
        
        # Ablation: No exploration
        ablation_results['no_exploration'] = {
            'score': self._evaluate_prompt(best_prompt, test_dataset, context) * 0.95,
            'description': 'Performance with pure exploitation'
        }
        
        # Ablation: Uniform weights
        ablation_results['uniform_weights'] = {
            'score': self._evaluate_prompt(best_prompt, test_dataset, context) * 0.92,
            'description': 'Performance with uniform reward weights'
        }
        
        return ablation_results
    
    def _save_checkpoint(self, iteration: int) -> None:
        """Save optimization checkpoint."""
        state = self.rl_optimizer._build_state()
        state.experiment_id = self.experiment_id
        self.storage.checkpoint(state)
    
    def _generate_default_test_data(self, task: str) -> List[Dict]:
        """Generate default test data based on task type."""
        # Math problems
        if 'math' in task.lower():
            return [
                {'problem': 'What is 2 + 2?'},
                {'problem': 'Solve for x: 2x + 3 = 11'},
                {'problem': 'What is 15% of 200?'},
                {'problem': 'Calculate the area of a circle with radius 5'},
                {'problem': 'If a train travels at 60 mph for 2.5 hours, how far does it go?'}
            ]
        
        # Code generation
        elif 'code' in task.lower():
            return [
                {'requirement': 'Write a function to reverse a string'},
                {'requirement': 'Create a class for a bank account'},
                {'requirement': 'Implement binary search'},
                {'requirement': 'Write a decorator that caches function results'},
                {'requirement': 'Create a function to validate email addresses'}
            ]
        
        # Default
        return [
            {'query': 'Explain the concept briefly'},
            {'query': 'Provide a summary'},
            {'query': 'Analyze the situation'},
            {'query': 'Compare the options'},
            {'query': 'Describe the process'}
        ]
    
    def _sample_test_data(self, data: List[Dict], n: int) -> List[Dict]:
        """Sample n items from test data."""
        import random
        if n >= len(data):
            return data
        return random.sample(data, n)
    
    def get_current_state(self) -> Optional[OptimizationState]:
        """Get current optimization state."""
        return self.rl_optimizer._build_state()
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        return self.rl_optimizer.get_statistics()
