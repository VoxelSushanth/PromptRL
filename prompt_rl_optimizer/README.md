# Prompt Optimization via Reinforcement Learning

A research-grade system for automatically improving prompts using iterative feedback and reinforcement learning signals.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Component Design](#component-design)
5. [Experiment Framework](#experiment-framework)
6. [Research Methodology](#research-methodology)
7. [API Reference](#api-reference)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROMPT RL OPTIMIZER                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Prompt     │───▶│  Execution   │───▶│  Evaluation  │               │
│  │  Generator   │    │    Engine    │    │    Engine    │               │
│  └──────┬───────┘    └──────────────┘    └──────┬───────┘               │
│         │                                       │                        │
│         │              ┌──────────────┐         │                        │
│         │◀─────────────│   RL Agent   │◀────────┤                        │
│         │              │  (Optimizer) │         │ (Rewards)              │
│         │              └──────┬───────┘         │                        │
│         │                     │                 │                        │
│         ▼                     ▼                 ▼                        │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │                    Memory & Logging                      │            │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │            │
│  │  │ Prompts  │  │ Outputs  │  │ Scores   │              │            │
│  │  │   DB     │  │   DB     │  │   DB     │              │            │
│  │  └──────────┘  └──────────┘  └──────────┘              │            │
│  └─────────────────────────────────────────────────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Data Flow:
1. Generator creates candidate prompts from base prompt + history
2. Execution Engine runs prompts on target LLM (batch supported)
3. Evaluation Engine scores outputs (rule-based + LLM-as-judge)
4. RL Agent computes rewards and updates policy
5. Memory stores all trials for reproducibility
6. Loop continues until convergence or max iterations
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from core.optimizer import PromptOptimizer
from configs.experiment_config import ExperimentConfig

# Initialize optimizer
config = ExperimentConfig.load("configs/default_config.yaml")
optimizer = PromptOptimizer(config)

# Run optimization
results = optimizer.optimize(
    task="math_word_problems",
    base_prompt="Solve the following math problem...",
    num_iterations=50,
    batch_size=10
)

# Get best prompt
best_prompt = results.get_best_prompt()
print(f"Best Reward: {results.best_reward}")
print(f"Optimized Prompt: {best_prompt}")
```

## Component Design

### 1. Prompt Generator
- **Input**: Base prompt, historical performance data, exploration flag
- **Output**: Candidate prompts with variations
- **Strategies**: Instruction clarity, constraint injection, output structuring, task decomposition

### 2. Execution Engine
- **Input**: Candidate prompts, test dataset
- **Output**: LLM responses with metadata
- **Features**: Batch execution, async support, rate limiting

### 3. Evaluation Engine
- **Metrics**: Correctness, format compliance, completeness, conciseness
- **Methods**: Rule-based, LLM-as-judge, hybrid scoring
- **Formula**: `Reward = Σ(w_i * metric_i) - penalties`

### 4. RL Agent
- **Algorithms**: ε-greedy Multi-Armed Bandit, UCB, PPO-lite
- **Strategy**: Balance exploration vs exploitation
- **Convergence**: Track score progression, early stopping

## Experiment Framework

The system includes pre-configured experiments to validate:
- **Hypothesis**: RL-optimized prompts outperform manually engineered prompts
- **Tasks**: Math reasoning, code generation, text classification, summarization
- **Baselines**: Human-written prompts, zero-shot, few-shot
- **Ablation Studies**: Without RL, different reward weights, exploration strategies

## Research Methodology

### Hypothesis
> "Iterative prompt optimization using reinforcement learning produces prompts that achieve statistically significant improvement over baseline human-engineered prompts."

### Metrics
- Primary: Average reward score improvement (%)
- Secondary: Convergence speed, sample efficiency, generalization

### Statistical Tests
- Paired t-test between optimized vs baseline
- Effect size (Cohen's d)
- Confidence intervals (95%)

## License

MIT License
