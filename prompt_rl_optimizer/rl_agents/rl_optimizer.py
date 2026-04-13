"""
Reinforcement Learning Agents for Prompt Optimization.

Implements:
- Multi-Armed Bandit (ε-greedy, UCB1, Thompson Sampling)
- PPO-lite style optimization
- Exploration vs Exploitation strategies
"""

import random
import math
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

from core.models import PromptVariant, OptimizationState
from configs.experiment_config import RLConfig, ExplorationStrategy


class BaseRLAgent(ABC):
    """Abstract base class for RL agents."""
    
    @abstractmethod
    def select_action(self, state: OptimizationState) -> str:
        """Select a prompt variant (action) given the current state."""
        pass
    
    @abstractmethod
    def update(self, action: str, reward: float) -> None:
        """Update agent's policy based on observed reward."""
        pass
    
    @abstractmethod
    def get_exploration_rate(self) -> float:
        """Return current exploration rate."""
        pass


class EpsilonGreedyAgent(BaseRLAgent):
    """
    ε-greedy Multi-Armed Bandit agent.
    
    Strategy:
    - With probability ε: explore (random action)
    - With probability 1-ε: exploit (best known action)
    
    ε decays over time from ε_start to ε_end.
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Action-value estimates Q(a)
        self.action_values: Dict[str, float] = {}
        
        # Action counts N(a)
        self.action_counts: Dict[str, int] = {}
        
        # Current epsilon
        self.epsilon = config.epsilon_start
        
        # Total updates
        self.total_updates = 0
    
    def select_action(self, state: OptimizationState) -> str:
        """Select action using ε-greedy strategy."""
        available_actions = [p.id for p in state.prompts]
        
        if not available_actions:
            raise ValueError("No prompts available for selection")
        
        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploit: select best known action
        # Use Upper Confidence Bound for tie-breaking among unexplored actions
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            if action not in self.action_values:
                # Optimistic initialization for unexplored actions
                value = 1.0  # Optimistic prior
            else:
                value = self.action_values[action]
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action or random.choice(available_actions)
    
    def update(self, action: str, reward: float) -> None:
        """Update action-value estimate using incremental mean."""
        self.total_updates += 1
        
        # Update count
        if action not in self.action_counts:
            self.action_counts[action] = 0
        
        self.action_counts[action] += 1
        n = self.action_counts[action]
        
        # Incremental update of action value
        # Q(a) = Q(a) + (1/n) * (R - Q(a))
        if action not in self.action_values:
            self.action_values[action] = reward
        else:
            old_value = self.action_values[action]
            self.action_values[action] = old_value + (reward - old_value) / n
        
        # Decay epsilon
        self._decay_epsilon()
    
    def _decay_epsilon(self) -> None:
        """Decay exploration rate."""
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
    
    def get_exploration_rate(self) -> float:
        """Return current epsilon."""
        return self.epsilon
    
    def get_action_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all actions."""
        stats = {}
        for action in self.action_values:
            stats[action] = {
                'value': self.action_values[action],
                'count': self.action_counts.get(action, 0)
            }
        return stats


class UCB1Agent(BaseRLAgent):
    """
    Upper Confidence Bound (UCB1) agent.
    
    Strategy:
    Select action that maximizes:
    Q(a) + c * sqrt(ln(t) / N(a))
    
    Where:
    - Q(a): estimated value of action a
    - c: exploration constant
    - t: total number of selections
    - N(a): number of times action a was selected
    
    This naturally balances exploration vs exploitation.
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Action-value estimates
        self.action_values: Dict[str, float] = {}
        
        # Action counts
        self.action_counts: Dict[str, int] = {}
        
        # Total selections
        self.total_selections = 0
    
    def select_action(self, state: OptimizationState) -> str:
        """Select action using UCB1 formula."""
        available_actions = [p.id for p in state.prompts]
        
        if not available_actions:
            raise ValueError("No prompts available for selection")
        
        self.total_selections += 1
        
        # First, try all actions at least once
        for action in available_actions:
            if action not in self.action_counts or self.action_counts[action] == 0:
                return action
        
        # Compute UCB1 value for each action
        ucb_values = {}
        
        for action in available_actions:
            q_value = self.action_values.get(action, 0.5)  # Default prior
            n_a = self.action_counts[action]
            
            # UCB1 formula
            exploration_bonus = self.config.ucb_exploration_constant * math.sqrt(
                math.log(self.total_selections) / n_a
            )
            
            ucb_values[action] = q_value + exploration_bonus
        
        # Select action with highest UCB value
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def update(self, action: str, reward: float) -> None:
        """Update action-value estimate."""
        # Update count
        if action not in self.action_counts:
            self.action_counts[action] = 0
        
        self.action_counts[action] += 1
        n = self.action_counts[action]
        
        # Incremental update
        if action not in self.action_values:
            self.action_values[action] = reward
        else:
            old_value = self.action_values[action]
            self.action_values[action] = old_value + (reward - old_value) / n
    
    def get_exploration_rate(self) -> float:
        """
        Return effective exploration rate.
        For UCB1, this is implicit in the exploration bonus.
        """
        if self.total_selections == 0:
            return 1.0
        
        # Estimate exploration rate based on how often we select unexplored actions
        unexplored_ratio = sum(1 for a in self.action_counts if self.action_counts[a] == 1) / max(1, len(self.action_counts))
        return unexplored_ratio * self.config.ucb_exploration_constant


class ThompsonSamplingAgent(BaseRLAgent):
    """
    Thompson Sampling agent using Beta distribution priors.
    
    Strategy:
    - Model reward probability for each action as Beta(α, β)
    - Sample from posterior distribution for each action
    - Select action with highest sampled value
    
    Naturally balances exploration vs exploitation through uncertainty.
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Beta distribution parameters for each action
        # α (alpha): successes + prior
        # β (beta): failures + prior
        self.alpha: Dict[str, float] = {}
        self.beta: Dict[str, float] = {}
        
        # Track total rewards and counts for conversion
        self.total_rewards: Dict[str, float] = {}
        self.action_counts: Dict[str, int] = {}
    
    def select_action(self, state: OptimizationState) -> str:
        """Select action by sampling from posterior distributions."""
        available_actions = [p.id for p in state.prompts]
        
        if not available_actions:
            raise ValueError("No prompts available for selection")
        
        sampled_values = {}
        
        for action in available_actions:
            # Get posterior parameters
            alpha = self.alpha.get(action, self.config.prior_alpha)
            beta_param = self.beta.get(action, self.config.prior_beta)
            
            # Sample from Beta distribution
            sampled_value = np.random.beta(alpha, beta_param)
            sampled_values[action] = sampled_value
        
        # Select action with highest sampled value
        return max(sampled_values.items(), key=lambda x: x[1])[0]
    
    def update(self, action: str, reward: float) -> None:
        """Update posterior distribution based on observed reward."""
        # Initialize if needed
        if action not in self.total_rewards:
            self.total_rewards[action] = 0.0
            self.action_counts[action] = 0
        
        # Update statistics
        self.total_rewards[action] += reward
        self.action_counts[action] += 1
        
        # Update Beta parameters
        # Treat reward as probability of "success"
        # α = successes + prior = Σrewards + prior_alpha
        # β = failures + prior = (n - Σrewards) + prior_beta
        n = self.action_counts[action]
        cumulative_reward = self.total_rewards[action]
        
        self.alpha[action] = cumulative_reward + self.config.prior_alpha
        self.beta[action] = (n - cumulative_reward) + self.config.prior_beta
    
    def get_exploration_rate(self) -> float:
        """Return uncertainty-weighted exploration rate."""
        if not self.alpha:
            return 1.0
        
        # Higher uncertainty = higher exploration
        uncertainties = []
        for action in self.alpha:
            # Variance of Beta distribution
            a = self.alpha[action]
            b = self.beta[action]
            variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
            uncertainties.append(variance)
        
        avg_uncertainty = np.mean(uncertainties) if uncertainties else 0
        return min(1.0, avg_uncertainty * 10)  # Scale to [0, 1]


class PPOLiteAgent(BaseRLAgent):
    """
    PPO-lite agent for prompt optimization.
    
    Simplified Proximal Policy Optimization adapted for discrete prompt selection.
    
    Key ideas:
    - Maintain a policy network (represented as action probabilities)
    - Update policy using clipped surrogate objective
    - Include entropy bonus for exploration
    
    Note: This is a simplified version without neural networks,
    using tabular representation for the policy.
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Policy: probability distribution over actions
        self.policy_logits: Dict[str, float] = {}
        
        # Value function: expected reward for each action
        self.value_function: Dict[str, float] = {}
        
        # Running statistics for normalization
        self.reward_mean = 0.0
        self.reward_var = 0.0
        self.reward_count = 0
        
        # Old policy for clipping
        self.old_policy: Dict[str, float] = {}
        
        # Entropy tracking
        self.entropy_history: List[float] = []
    
    def _get_policy_probs(self, available_actions: List[str]) -> Dict[str, float]:
        """Get softmax probabilities for available actions."""
        if not available_actions:
            return {}
        
        # Get logits for available actions
        logits = []
        for action in available_actions:
            logit = self.policy_logits.get(action, 0.0)
            logits.append(logit)
        
        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        
        probs = {action: exp_logits[i] / sum_exp for i, action in enumerate(available_actions)}
        return probs
    
    def select_action(self, state: OptimizationState) -> str:
        """Select action by sampling from policy."""
        available_actions = [p.id for p in state.prompts]
        
        if not available_actions:
            raise ValueError("No prompts available for selection")
        
        # Get policy probabilities
        probs = self._get_policy_probs(available_actions)
        
        # Sample from policy
        actions = list(probs.keys())
        probabilities = [probs[a] for a in actions]
        
        return np.random.choice(actions, p=probabilities)
    
    def update(self, action: str, reward: float) -> None:
        """Update policy using PPO-style update."""
        # Update running statistics for reward normalization
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_var += delta * (reward - self.reward_mean)
        
        # Normalize reward
        if self.reward_count > 1 and self.reward_var > 0:
            reward_std = math.sqrt(self.reward_var / self.reward_count)
            normalized_reward = (reward - self.reward_mean) / max(reward_std, 1e-6)
        else:
            normalized_reward = reward
        
        # Store old policy
        available_actions = list(self.policy_logits.keys())
        if available_actions:
            self.old_policy = self._get_policy_probs(available_actions).copy()
        
        # Update policy logit for the taken action
        # Simplified PPO update: gradient ascent on clipped objective
        if action in self.policy_logits:
            old_prob = self.old_policy.get(action, 0.5)
        else:
            old_prob = 0.5
        
        # Get current probability
        current_probs = self._get_policy_probs([action] + list(self.policy_logits.keys()))
        current_prob = current_probs.get(action, 0.5)
        
        # Probability ratio
        ratio = current_prob / max(old_prob, 1e-6)
        
        # Clipped surrogate objective
        clip_epsilon = self.config.ppo_clip_epsilon
        clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        
        # Policy gradient update (simplified)
        policy_gradient = min(ratio * normalized_reward, clipped_ratio * normalized_reward)
        
        # Update logit
        if action not in self.policy_logits:
            self.policy_logits[action] = 0.0
        
        self.policy_logits[action] += self.config.ppo_learning_rate * policy_gradient
        
        # Update value function
        if action not in self.value_function:
            self.value_function[action] = 0.0
        
        # TD error
        td_error = normalized_reward - self.value_function[action]
        self.value_function[action] += self.config.ppo_learning_rate * td_error
        
        # Track entropy
        entropy = self._compute_entropy()
        self.entropy_history.append(entropy)
        
        # Add entropy bonus to encourage exploration
        if len(self.entropy_history) > 10:
            recent_entropy = np.mean(self.entropy_history[-10:])
            if recent_entropy < 0.5:  # Low entropy = add bonus
                for a in self.policy_logits:
                    self.policy_logits[a] += self.config.ppo_entropy_coefficient
    
    def _compute_entropy(self) -> float:
        """Compute policy entropy."""
        probs = self._get_policy_probs(list(self.policy_logits.keys()))
        
        if not probs:
            return 0.0
        
        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * math.log(p + 1e-10)
        
        return entropy
    
    def get_exploration_rate(self) -> float:
        """Return policy entropy as exploration rate."""
        if not self.entropy_history:
            return 1.0
        
        # Normalize entropy to [0, 1]
        max_entropy = math.log(max(1, len(self.policy_logits)))
        current_entropy = self.entropy_history[-1]
        
        return current_entropy / max(max_entropy, 1e-6)


class RLOptimizer:
    """
    Main RL optimizer that manages the optimization loop.
    
    Coordinates:
    - RL agent selection
    - Prompt population management
    - Convergence tracking
    - Early stopping
    """
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Initialize RL agent based on strategy
        self.agent = self._create_agent(config.strategy)
        
        # Population management
        self.population: List[PromptVariant] = []
        self.elite_prompts: List[PromptVariant] = []
        
        # Tracking
        self.reward_history: List[float] = []
        self.best_reward = 0.0
        self.best_prompt: Optional[PromptVariant] = None
        
        # Convergence tracking
        self.no_improvement_count = 0
        self.convergence_detected = False
    
    def _create_agent(self, strategy: ExplorationStrategy) -> BaseRLAgent:
        """Create RL agent based on strategy type."""
        if strategy == ExplorationStrategy.EPSILON_GREEDY:
            return EpsilonGreedyAgent(self.config)
        elif strategy == ExplorationStrategy.UCB1:
            return UCB1Agent(self.config)
        elif strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            return ThompsonSamplingAgent(self.config)
        elif strategy == ExplorationStrategy.PPO_LITE:
            return PPOLiteAgent(self.config)
        else:
            return EpsilonGreedyAgent(self.config)
    
    def select_prompt(self) -> str:
        """Select a prompt variant using RL agent."""
        state = self._build_state()
        return self.agent.select_action(state)
    
    def update_from_reward(self, prompt_id: str, reward: float) -> None:
        """Update RL agent and tracking based on observed reward."""
        # Update agent
        self.agent.update(prompt_id, reward)
        
        # Update tracking
        self.reward_history.append(reward)
        
        # Update best
        if reward > self.best_reward:
            self.best_reward = reward
            self.no_improvement_count = 0
            
            # Find and store best prompt
            for prompt in self.population:
                if prompt.id == prompt_id:
                    self.best_prompt = prompt
                    break
        else:
            self.no_improvement_count += 1
        
        # Update prompt performance
        for prompt in self.population:
            if prompt.id == prompt_id:
                prompt.update_performance(reward)
                break
        
        # Check convergence
        self._check_convergence()
    
    def manage_population(
        self,
        new_prompts: List[PromptVariant],
        max_size: int = None
    ) -> None:
        """
        Manage prompt population with elitism.
        
        Args:
            new_prompts: New candidate prompts to add
            max_size: Maximum population size
        """
        max_size = max_size or self.config.population_size
        
        # Add new prompts
        self.population.extend(new_prompts)
        
        # Keep only top performers
        self.population.sort(key=lambda p: p.average_reward, reverse=True)
        self.population = self.population[:max_size]
        
        # Update elite set
        self.elite_prompts = self.population[:self.config.elite_size]
    
    def _build_state(self) -> OptimizationState:
        """Build current optimization state."""
        state = OptimizationState(
            experiment_id="current",
            current_iteration=len(self.reward_history),
            total_iterations=self.config.convergence_window * 10,
            prompts=self.population.copy(),
            elite_prompts=[p.id for p in self.elite_prompts],
            best_prompt_id=self.best_prompt.id if self.best_prompt else None,
            best_reward=self.best_reward,
            reward_history=self.reward_history.copy(),
            exploration_rate=self.agent.get_exploration_rate()
        )
        
        # Copy agent state
        if isinstance(self.agent, EpsilonGreedyAgent):
            state.action_values = self.agent.action_values.copy()
            state.action_counts = self.agent.action_counts.copy()
        
        return state
    
    def _check_convergence(self) -> None:
        """Check if optimization has converged."""
        if len(self.reward_history) < self.config.convergence_window:
            return
        
        # Check reward variance
        recent_rewards = self.reward_history[-self.config.convergence_window:]
        variance = np.var(recent_rewards)
        
        if variance < self.config.convergence_threshold:
            self.convergence_detected = True
        
        # Check early stopping
        if self.no_improvement_count >= self.config.early_stopping_patience:
            self.convergence_detected = True
    
    def should_stop(self) -> bool:
        """Check if optimization should stop."""
        return self.convergence_detected
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        stats = {
            'total_iterations': len(self.reward_history),
            'best_reward': self.best_reward,
            'current_exploration_rate': self.agent.get_exploration_rate(),
            'convergence_detected': self.convergence_detected,
            'population_size': len(self.population),
            'elite_size': len(self.elite_prompts)
        }
        
        if self.reward_history:
            stats['reward_stats'] = {
                'mean': np.mean(self.reward_history),
                'std': np.std(self.reward_history),
                'min': np.min(self.reward_history),
                'max': np.max(self.reward_history),
                'recent_mean': np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else np.mean(self.reward_history)
            }
        
        # Agent-specific stats
        if hasattr(self.agent, 'get_action_statistics'):
            stats['agent_stats'] = self.agent.get_action_statistics()
        
        return stats
    
    def reset(self) -> None:
        """Reset optimizer state."""
        self.population = []
        self.elite_prompts = []
        self.reward_history = []
        self.best_reward = 0.0
        self.best_prompt = None
        self.no_improvement_count = 0
        self.convergence_detected = False
        
        # Recreate agent
        self.agent = self._create_agent(self.config.strategy)
