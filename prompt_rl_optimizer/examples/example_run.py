#!/usr/bin/env python3
"""
Example Run: Prompt Optimization via Reinforcement Learning

This script demonstrates a complete optimization run showing:
1. Initial baseline prompt
2. Iterative improvement through RL
3. Final optimized prompt with measurable improvement

Usage:
    python examples/example_run.py
"""

import sys
sys.path.insert(0, '/workspace/prompt_rl_optimizer')

from configs.experiment_config import ExperimentConfig
from core.optimizer import PromptOptimizer
from experiments.experiment_framework import BenchmarkTasks, ExperimentRunner


def run_single_iteration_example():
    """
    Demonstrate one full iteration of the optimization loop:
    prompt → output → score → improvement
    """
    print("="*70)
    print("EXAMPLE: Single Optimization Iteration")
    print("="*70)
    
    # Setup configuration
    config = ExperimentConfig(
        experiment_name="example_iteration",
        num_iterations=1,
        candidates_per_iteration=3
    )
    
    # Override to use mock LLM for demonstration
    config.execution.provider = "local"
    
    # Initialize optimizer
    optimizer = PromptOptimizer(config)
    
    # Base prompt
    base_prompt = "Solve this math problem:"
    task = "math_word_problems"
    
    print(f"\n[1] BASE PROMPT:")
    print(f"    {base_prompt}")
    
    # Generate candidate prompts
    print(f"\n[2] GENERATING CANDIDATE PROMPTS...")
    context = {'task_description': task, 'task_type': 'math'}
    
    candidates = optimizer.generator.generate_candidates(
        base_prompt=base_prompt,
        num_candidates=3,
        context=context,
        explore=True
    )
    
    for i, cand in enumerate(candidates, 1):
        print(f"\n    Candidate {i} (Strategy: {cand.generation_strategy}):")
        print(f"    {cand.variant_prompt[:150]}..." if len(cand.variant_prompt) > 150 else f"    {cand.variant_prompt}")
    
    # Execute on test input
    print(f"\n[3] EXECUTING PROMPTS ON TEST INPUT...")
    test_input = {'problem': 'What is 2 + 2?'}
    
    outputs = []
    for cand in candidates:
        output = optimizer.execution_engine.execute_sync(
            prompt=cand,
            input_data=test_input
        )
        outputs.append(output)
        print(f"\n    Output for Candidate {candidates.index(cand)+1}:")
        print(f"    {output.output_text[:100]}...")
    
    # Evaluate outputs
    print(f"\n[4] EVALUATING OUTPUTS AND COMPUTING REWARDS...")
    
    rewards = []
    for i, (cand, output) in enumerate(zip(candidates, outputs), 1):
        metrics, reward = optimizer.evaluation_engine.evaluate(output, context)
        rewards.append((cand.id, reward))
        
        print(f"\n    Candidate {i} Scores:")
        print(f"      - Correctness:       {metrics.correctness_score:.3f}")
        print(f"      - Format Compliance: {metrics.format_compliance_score:.3f}")
        print(f"      - Completeness:      {metrics.completeness_score:.3f}")
        print(f"      - Conciseness:       {metrics.conciseness_score:.3f}")
        print(f"      - FINAL REWARD:      {reward:.4f}")
    
    # Update RL agent
    print(f"\n[5] UPDATING RL AGENT...")
    best_candidate_id = max(rewards, key=lambda x: x[1])[0]
    best_reward = max(r for _, r in rewards)
    
    optimizer.rl_optimizer.manage_population(candidates)
    optimizer.rl_optimizer.update_from_reward(best_candidate_id, best_reward)
    
    stats = optimizer.rl_optimizer.get_statistics()
    print(f"\n    RL Agent Statistics:")
    print(f"      - Best Reward: {stats['best_reward']:.4f}")
    print(f"      - Exploration Rate: {stats['current_exploration_rate']:.3f}")
    print(f"      - Population Size: {stats['population_size']}")
    
    # Show improvement potential
    print(f"\n[6] IMPROVEMENT ANALYSIS:")
    initial_reward = rewards[0][1]
    print(f"      - Initial candidate reward: {initial_reward:.4f}")
    print(f"      - Best candidate reward: {best_reward:.4f}")
    print(f"      - Improvement: {((best_reward - initial_reward) / max(initial_reward, 0.001)) * 100:.1f}%")
    
    print("\n" + "="*70)
    print("Iteration complete! The RL agent has learned which strategies work best.")
    print("Over multiple iterations, this process converges to optimal prompts.")
    print("="*70)
    
    return {
        'base_prompt': base_prompt,
        'candidates': [(c.variant_prompt, r) for c, (_, r) in zip(candidates, rewards)],
        'best_reward': best_reward
    }


def run_full_optimization_demo():
    """Run a complete optimization demonstration."""
    print("\n\n")
    print("#"*70)
    print("# FULL OPTIMIZATION DEMONSTRATION")
    print("#"*70)
    
    # Use benchmark task
    task_info = BenchmarkTasks.get_task('math_reasoning')
    
    runner = ExperimentRunner()
    
    result = runner.run_experiment(
        task=task_info['task'],
        base_prompt=task_info['base_prompt'],
        num_iterations=20,  # Reduced for demo
        run_name="demo_run"
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print(f"\n📊 Experiment Statistics:")
    print(f"   - Total Iterations: {result.total_iterations}")
    print(f"   - Total Trials: {result.total_trials}")
    print(f"   - Best Reward: {result.best_reward:.4f}")
    print(f"   - Improvement over Baseline: {result.improvement_over_baseline:.2f}%")
    
    print(f"\n🎯 Optimized Prompt:")
    print("-"*68)
    prompt_text = result.best_prompt.variant_prompt
    if len(prompt_text) > 500:
        print(f"   {prompt_text[:500]}...")
    else:
        print(f"   {prompt_text}")
    print("-"*68)
    
    if result.ablation_results:
        print(f"\n🔬 Ablation Study Results:")
        for variant, data in result.ablation_results.items():
            print(f"   - {variant}: {data['score']:.4f}")
            print(f"     {data['description']}")
    
    print("\n" + "="*70)
    print("✅ Optimization complete!")
    print("="*70)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Optimization Demo")
    parser.add_argument(
        '--mode',
        choices=['single', 'full'],
        default='single',
        help='Run single iteration example or full optimization'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_iteration_example()
    else:
        run_full_optimization_demo()
