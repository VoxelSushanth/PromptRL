"""
Prompt Generator Module.

Generates improved prompts from base prompts and historical performance data
using structured optimization strategies.
"""

import random
import re
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from core.models import PromptVariant, StrategyType
from configs.experiment_config import GeneratorConfig


class BaseStrategy(ABC):
    """Abstract base class for prompt generation strategies."""
    
    @abstractmethod
    def apply(self, base_prompt: str, context: Dict) -> str:
        """Apply the strategy to generate a variant prompt."""
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> StrategyType:
        """Return the strategy type."""
        pass


class InstructionClarityStrategy(BaseStrategy):
    """
    Enhances prompt clarity by adding explicit instructions.
    
    Strategies:
    - Add step-by-step reasoning requests
    - Specify output expectations clearly
    - Use action verbs and concrete language
    """
    
    def __init__(self, templates: List[str] = None):
        self.templates = templates or [
            "Clearly and step-by-step {task}",
            "Break down {task} into logical steps",
            "Provide a detailed explanation for {task}",
            "Think through {task} systematically",
            "Approach {task} methodically, showing your work"
        ]
    
    def apply(self, base_prompt: str, context: Dict) -> str:
        task_description = context.get('task_description', 'the task')
        template = random.choice(self.templates)
        
        # Insert clarity instruction at the beginning
        clarity_instruction = template.format(task=task_description)
        
        return f"{clarity_instruction}.\n\n{base_prompt}"
    
    def get_strategy_type(self) -> StrategyType:
        return StrategyType.INSTRUCTION_CLARITY


class ConstraintInjectionStrategy(BaseStrategy):
    """
    Adds explicit constraints to guide the LLM's output.
    
    Strategies:
    - Length constraints
    - Format requirements
    - Content restrictions
    - Style guidelines
    """
    
    def __init__(self, templates: List[str] = None):
        self.templates = templates or [
            "Ensure your answer is {constraint}",
            "Your response must follow these rules: {constraint}",
            "Adhere strictly to: {constraint}",
            "Important constraints: {constraint}",
            "You must: {constraint}"
        ]
    
    def apply(self, base_prompt: str, context: Dict) -> str:
        constraints = context.get('constraints', [])
        
        if not constraints:
            # Default constraints based on task type
            task_type = context.get('task_type', 'general')
            constraints = self._get_default_constraints(task_type)
        
        constraint_text = "; ".join(constraints)
        template = random.choice(self.templates)
        
        constraint_instruction = template.format(constraint=constraint_text)
        
        return f"{base_prompt}\n\n{constraint_instruction}"
    
    def _get_default_constraints(self, task_type: str) -> List[str]:
        """Get default constraints based on task type."""
        constraint_map = {
            'math': ["Show all calculation steps", "Verify your final answer"],
            'code': ["Include comments", "Follow best practices", "Handle edge cases"],
            'writing': ["Use clear language", "Maintain consistent tone"],
            'analysis': ["Support claims with evidence", "Consider multiple perspectives"],
            'general': ["Be precise and accurate", "Stay on topic"]
        }
        return constraint_map.get(task_type, constraint_map['general'])
    
    def get_strategy_type(self) -> StrategyType:
        return StrategyType.CONSTRAINT_INJECTION


class OutputStructuringStrategy(BaseStrategy):
    """
    Specifies the exact format and structure of the expected output.
    
    Strategies:
    - JSON/XML formatting
    - Section headers
    - Bullet points
    - Numbered lists
    """
    
    def __init__(self, templates: List[str] = None):
        self.templates = templates or [
            "Format your output as: {format}",
            "Use the following structure: {format}",
            "Organize your response with: {format}",
            "Your answer should follow this format: {format}",
            "Structure your response as follows: {format}"
        ]
    
    def apply(self, base_prompt: str, context: Dict) -> str:
        output_format = context.get('output_format', None)
        
        if not output_format:
            task_type = context.get('task_type', 'general')
            output_format = self._get_default_format(task_type)
        
        template = random.choice(self.templates)
        format_instruction = template.format(format=output_format)
        
        return f"{base_prompt}\n\n{format_instruction}"
    
    def _get_default_format(self, task_type: str) -> str:
        """Get default output format based on task type."""
        format_map = {
            'math': "Step 1: [analysis], Step 2: [calculation], Final Answer: [boxed result]",
            'code': "```language\\n[code]\\n``` followed by explanation",
            'writing': "Introduction, Body Paragraphs, Conclusion",
            'analysis': "Summary, Key Points (bulleted), Detailed Analysis, Conclusion",
            'classification': "JSON: {{'label': 'category', 'confidence': 0.0-1.0}}",
            'general': "Clear sections with headers, bullet points for key information"
        }
        return format_map.get(task_type, format_map['general'])
    
    def get_strategy_type(self) -> StrategyType:
        return StrategyType.OUTPUT_STRUCTURING


class TaskDecompositionStrategy(BaseStrategy):
    """
    Breaks down complex tasks into smaller, manageable subtasks.
    
    Strategies:
    - Sequential steps
    - Parallel components
    - Hierarchical breakdown
    """
    
    def __init__(self, templates: List[str] = None):
        self.templates = templates or [
            "Complete this task in steps: {steps}",
            "Break this into phases: {steps}",
            "Address each part sequentially: {steps}",
            "First {step1}, then {step2}, finally {step3}"
        ]
    
    def apply(self, base_prompt: str, context: Dict) -> str:
        decomposition = context.get('decomposition', None)
        
        if not decomposition:
            decomposition = self._generate_decomposition(context)
        
        if isinstance(decomposition, list):
            steps_text = " → ".join(decomposition)
        else:
            steps_text = decomposition
        
        return f"Task breakdown: {steps_text}\n\n{base_prompt}"
    
    def _generate_decomposition(self, context: Dict) -> List[str]:
        """Generate a default task decomposition."""
        task_type = context.get('task_type', 'general')
        
        decompositions = {
            'math': ["Understand the problem", "Identify relevant formulas", 
                    "Perform calculations", "Verify the result"],
            'code': ["Analyze requirements", "Design the solution", 
                    "Implement the code", "Test and debug"],
            'writing': ["Plan the structure", "Draft content", 
                       "Revise and refine", "Final polish"],
            'analysis': ["Gather information", "Identify patterns", 
                        "Draw conclusions", "Present findings"],
            'general': ["Understand the task", "Plan approach", 
                       "Execute systematically", "Review results"]
        }
        
        return decompositions.get(task_type, decompositions['general'])
    
    def get_strategy_type(self) -> StrategyType:
        return StrategyType.TASK_DECOMPOSITION


class ChainOfThoughtStrategy(BaseStrategy):
    """
    Encourages explicit reasoning before providing the final answer.
    """
    
    def apply(self, base_prompt: str, context: Dict) -> str:
        cot_prompts = [
            "Let's think through this step by step.",
            "Work through this systematically, explaining your reasoning.",
            "Show your complete thought process before giving the final answer.",
            "Reason through each step carefully."
        ]
        
        cot_instruction = random.choice(cot_prompts)
        return f"{cot_instruction}\\n\\n{base_prompt}\\n\\nRemember to show your reasoning before the final answer."
    
    def get_strategy_type(self) -> StrategyType:
        return StrategyType.CHAIN_OF_THOUGHT


class FewShotExamplesStrategy(BaseStrategy):
    """
    Adds example inputs and outputs to guide the LLM.
    """
    
    def __init__(self):
        self.example_bank = {}
    
    def add_examples(self, task_type: str, examples: List[Dict[str, str]]):
        """Add examples for a specific task type."""
        self.example_bank[task_type] = examples
    
    def apply(self, base_prompt: str, context: Dict) -> str:
        task_type = context.get('task_type', 'general')
        examples = self.example_bank.get(task_type, context.get('examples', []))
        
        if not examples:
            # Generate generic examples
            examples = self._generate_generic_examples(task_type)
        
        examples_text = self._format_examples(examples)
        
        return f"{base_prompt}\\n\\nHere are some examples:\\n{examples_text}"
    
    def _generate_generic_examples(self, task_type: str) -> List[Dict[str, str]]:
        """Generate generic placeholder examples."""
        return [
            {"input": f"Example input 1 for {task_type}", 
             "output": f"Example output 1 demonstrating the expected format"},
            {"input": f"Example input 2 for {task_type}", 
             "output": f"Example output 2 showing correct approach"}
        ]
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format examples for inclusion in the prompt."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"Example {i}:\\nInput: {ex['input']}\\nOutput: {ex['output']}")
        return "\\n\\n".join(formatted)
    
    def get_strategy_type(self) -> StrategyType:
        return StrategyType.FEW_SHOT_EXAMPLES


class PromptGenerator:
    """
    Main prompt generator that combines multiple strategies.
    
    Generates candidate prompts using:
    - Individual strategies
    - Combinations of strategies
    - Evolutionary operations (mutation, crossover)
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        
        # Initialize strategies
        self.strategies: List[BaseStrategy] = []
        
        if config.use_instruction_clarity:
            self.strategies.append(InstructionClarityStrategy(config.clarity_templates))
        
        if config.use_constraint_injection:
            self.strategies.append(ConstraintInjectionStrategy(config.constraint_templates))
        
        if config.use_output_structuring:
            self.strategies.append(OutputStructuringStrategy(config.structure_templates))
        
        if config.use_task_decomposition:
            self.strategies.append(TaskDecompositionStrategy())
        
        # Additional strategies
        self.strategies.append(ChainOfThoughtStrategy())
        self.strategies.append(FewShotExamplesStrategy())
        
        # Track generation history
        self.generation_history: List[PromptVariant] = []
    
    def generate_candidates(
        self,
        base_prompt: str,
        num_candidates: int,
        context: Dict = None,
        existing_prompts: List[PromptVariant] = None,
        explore: bool = True
    ) -> List[PromptVariant]:
        """
        Generate candidate prompt variants.
        
        Args:
            base_prompt: The original prompt to optimize
            num_candidates: Number of candidates to generate
            context: Task-specific context (task_type, constraints, etc.)
            existing_prompts: Previously generated prompts for evolutionary ops
            explore: Whether to use exploration strategies
        
        Returns:
            List of PromptVariant objects
        """
        context = context or {}
        candidates = []
        
        for i in range(num_candidates):
            if explore and random.random() < self.config.mutation_rate:
                # Mutation: Apply random strategy combination
                variant = self._mutate(base_prompt, context)
            elif (existing_prompts and 
                  random.random() < self.config.crossover_rate):
                # Crossover: Combine elements from existing prompts
                variant = self._crossover(base_prompt, existing_prompts, context)
            else:
                # Exploitation: Apply best-known strategies
                variant = self._exploit(base_prompt, existing_prompts, context)
            
            candidates.append(variant)
            self.generation_history.append(variant)
        
        return candidates
    
    def _mutate(self, base_prompt: str, context: Dict) -> PromptVariant:
        """Apply mutation: random strategy combinations."""
        # Select 1-3 random strategies
        num_strategies = random.randint(1, min(3, len(self.strategies)))
        selected_strategies = random.sample(self.strategies, num_strategies)
        
        variant_prompt = base_prompt
        applied_strategies = []
        
        for strategy in selected_strategies:
            variant_prompt = strategy.apply(variant_prompt, context)
            applied_strategies.append(strategy.get_strategy_type().value)
        
        return PromptVariant(
            base_prompt=base_prompt,
            variant_prompt=variant_prompt,
            generation_strategy="mutation_" + "+".join(applied_strategies),
            metadata={
                'strategies_applied': applied_strategies,
                'num_strategies': len(applied_strategies)
            }
        )
    
    def _crossover(
        self,
        base_prompt: str,
        existing_prompts: List[PromptVariant],
        context: Dict
    ) -> PromptVariant:
        """Apply crossover: combine elements from existing high-performing prompts."""
        if not existing_prompts:
            return self._mutate(base_prompt, context)
        
        # Select top performers
        sorted_prompts = sorted(existing_prompts, 
                               key=lambda p: p.average_reward, 
                               reverse=True)
        top_prompts = sorted_prompts[:min(3, len(sorted_prompts))]
        
        # Extract useful components from top prompts
        variant_prompt = base_prompt
        
        # Randomly select components to inherit
        if top_prompts:
            parent = random.choice(top_prompts)
            # Inherit structure from parent
            variant_prompt = self._combine_prompts(variant_prompt, parent.variant_prompt)
        
        # Apply one additional strategy
        strategy = random.choice(self.strategies)
        variant_prompt = strategy.apply(variant_prompt, context)
        
        return PromptVariant(
            base_prompt=base_prompt,
            variant_prompt=variant_prompt,
            generation_strategy="crossover",
            metadata={
                'parent_id': parent.id if top_prompts else None,
                'strategies_applied': [strategy.get_strategy_type().value]
            }
        )
    
    def _exploit(
        self,
        base_prompt: str,
        existing_prompts: List[PromptVariant],
        context: Dict
    ) -> PromptVariant:
        """Exploit: use best-known strategies based on historical performance."""
        if not existing_prompts:
            return self._mutate(base_prompt, context)
        
        # Analyze which strategies performed best
        strategy_performance = {}
        
        for prompt in existing_prompts:
            if prompt.total_trials > 0:
                strategy = prompt.generation_strategy
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = []
                strategy_performance[strategy].append(prompt.average_reward)
        
        # Calculate average performance per strategy
        avg_performance = {
            s: sum(rewards)/len(rewards) 
            for s, rewards in strategy_performance.items()
        }
        
        # Select top strategies
        if avg_performance:
            top_strategy = max(avg_performance.items(), key=lambda x: x[1])[0]
            
            # Parse strategy name and apply
            strategy_types = top_strategy.replace("mutation_", "").split("+")
            variant_prompt = base_prompt
            
            for strat_name in strategy_types:
                for strategy in self.strategies:
                    if strategy.get_strategy_type().value == strat_name:
                        variant_prompt = strategy.apply(variant_prompt, context)
                        break
            
            return PromptVariant(
                base_prompt=base_prompt,
                variant_prompt=variant_prompt,
                generation_strategy=f"exploit_{top_strategy}",
                metadata={'based_on_strategy': top_strategy}
            )
        
        return self._mutate(base_prompt, context)
    
    def _combine_prompts(self, prompt1: str, prompt2: str) -> str:
        """Combine two prompts intelligently."""
        # Simple combination: take introduction from prompt1, structure from prompt2
        lines1 = prompt1.split('\\n')
        lines2 = prompt2.split('\\n')
        
        # Keep first half of prompt1
        split1 = len(lines1) // 2
        
        # Keep second half of prompt2
        split2 = len(lines2) // 2
        
        combined_lines = lines1[:split1] + lines2[split2:]
        
        return '\\n'.join(combined_lines)
    
    def get_generation_statistics(self) -> Dict:
        """Get statistics about prompt generation."""
        if not self.generation_history:
            return {'total_generated': 0}
        
        strategy_counts = {}
        for prompt in self.generation_history:
            strategy = prompt.generation_strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_generated': len(self.generation_history),
            'strategy_distribution': strategy_counts,
            'avg_reward_by_strategy': self._avg_reward_by_strategy()
        }
    
    def _avg_reward_by_strategy(self) -> Dict[str, float]:
        """Calculate average reward per strategy."""
        strategy_rewards = {}
        strategy_counts = {}
        
        for prompt in self.generation_history:
            if prompt.total_trials > 0:
                strategy = prompt.generation_strategy
                strategy_rewards[strategy] = strategy_rewards.get(strategy, 0) + prompt.cumulative_reward
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + prompt.total_trials
        
        return {
            s: strategy_rewards[s] / strategy_counts[s]
            for s in strategy_rewards
        }
