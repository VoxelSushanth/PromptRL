"""
Execution Engine Module.

Handles LLM API interactions for:
- Running prompts on target LLMs
- Batch execution
- Rate limiting and retries
- Output collection
"""

import asyncio
import time
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import random

from core.models import LLMOutput, PromptVariant
from configs.experiment_config import ExecutionConfig, LLMProvider


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        pass


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing without API calls.
    
    Generates realistic-looking responses based on task type.
    """
    
    def __init__(self):
        self.response_templates = {
            'math': self._generate_math_response,
            'code': self._generate_code_response,
            'writing': self._generate_writing_response,
            'analysis': self._generate_analysis_response,
            'default': self._generate_default_response
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response based on prompt content."""
        # Simulate API latency
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Detect task type from prompt
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['calculate', 'solve', 'math', 'equation']):
            return self._generate_math_response(prompt)
        elif any(word in prompt_lower for word in ['code', 'program', 'function', 'write']):
            return self._generate_code_response(prompt)
        elif any(word in prompt_lower for word in ['write', 'essay', 'story', 'describe']):
            return self._generate_writing_response(prompt)
        elif any(word in prompt_lower for word in ['analyze', 'compare', 'evaluate']):
            return self._generate_analysis_response(prompt)
        else:
            return self._generate_default_response(prompt)
    
    def _generate_math_response(self, prompt: str) -> str:
        """Generate math-like response."""
        return """Step 1: Understand the problem
Let's break down what we're asked to solve.

Step 2: Identify relevant information
From the problem statement, we can extract the key variables and constraints.

Step 3: Apply appropriate formulas
Using standard mathematical principles, we calculate:

Calculation:
- Intermediate result 1: 42
- Intermediate result 2: 108
- Final computation: 42 + 108 = 150

Step 4: Verify the result
Checking our work by substituting back into the original equation confirms our answer.

Final Answer: 150"""
    
    def _generate_code_response(self, prompt: str) -> str:
        """Generate code-like response."""
        return '''```python
def solution(input_data):
    """
    Solves the given problem efficiently.
    
    Args:
        input_data: The input to process
    
    Returns:
        The computed result
    """
    # Initialize variables
    result = []
    
    # Process input
    for item in input_data:
        # Apply transformation
        transformed = self._transform(item)
        result.append(transformed)
    
    # Return final result
    return result

def _transform(item):
    """Helper function for transformation."""
    return item * 2
```

This solution has O(n) time complexity and handles edge cases appropriately.'''
    
    def _generate_writing_response(self, prompt: str) -> str:
        """Generate writing-like response."""
        return """## Introduction

This topic presents an interesting perspective that warrants careful consideration. 
In this response, we will explore the key aspects and implications.

## Main Points

### First Point
The primary consideration involves understanding the fundamental principles at play. 
Research has shown that this approach yields significant benefits.

### Second Point
Furthermore, we must examine the secondary effects and how they interact with 
the primary factors. The relationship between these elements is complex but important.

### Third Point
Finally, considering the long-term implications provides valuable insight into 
the broader context and potential applications.

## Conclusion

In summary, the analysis reveals multiple dimensions worth considering. 
The evidence supports a nuanced understanding of the topic."""
    
    def _generate_analysis_response(self, prompt: str) -> str:
        """Generate analysis-like response."""
        return """## Summary

This analysis examines the key components and their interrelationships.

## Key Findings

1. **Primary Observation**: The data indicates a clear pattern that suggests...

2. **Supporting Evidence**: Multiple sources confirm this interpretation:
   - Source A demonstrates...
   - Source B corroborates...
   - Source C extends...

3. **Alternative Perspectives**: While the above is compelling, we should also consider:
   - Counterargument 1: ...
   - Counterargument 2: ...

## Detailed Analysis

### Component 1
Examining the first component reveals...

### Component 2
The second component interacts with the first by...

### Synthesis
Combining these insights, we can conclude that...

## Conclusion

The analysis supports the hypothesis while acknowledging certain limitations. 
Future work should address these gaps."""
    
    def _generate_default_response(self, prompt: str) -> str:
        """Generate default response."""
        return f"""Based on your request, here is a comprehensive response:

## Overview

Your query touches on several important aspects that deserve attention.

## Key Points

1. The primary consideration is understanding the context and requirements.

2. Following established best practices ensures quality results.

3. Attention to detail throughout the process leads to better outcomes.

## Recommendations

- Start by clearly defining objectives
- Break down complex tasks into manageable steps
- Verify results at each stage
- Document your approach for reproducibility

## Conclusion

This structured approach should help achieve the desired outcome effectively."""
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, config: ExecutionConfig):
        self.api_key = api_key
        self.config = config
        
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        
        return response.choices[0].message.content
    
    def get_token_count(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model(self.config.model_name)
            return len(encoder.encode(text))
        except Exception:
            return len(text) // 4


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""
    
    def __init__(self, api_key: str, config: ExecutionConfig):
        self.api_key = api_key
        self.config = config
        
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        
        response = await self.client.messages.create(
            model=self.config.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def get_token_count(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class ExecutionEngine:
    """
    Main execution engine for running prompts on LLMs.
    
    Features:
    - Batch execution
    - Rate limiting
    - Automatic retries
    - Latency tracking
    """
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.client: Optional[BaseLLMClient] = None
        self._rate_limiter = RateLimiter(
            requests_per_minute=config.requests_per_minute,
            tokens_per_minute=config.tokens_per_minute
        )
        
        # Initialize mock client by default
        self.client = MockLLMClient()
    
    def set_client(self, client: BaseLLMClient) -> None:
        """Set the LLM client."""
        self.client = client
    
    def initialize_from_config(self) -> None:
        """Initialize LLM client from configuration."""
        import os
        
        if self.config.provider == LLMProvider.OPENAI:
            api_key = os.getenv(self.config.api_key_env_var)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {self.config.api_key_env_var}")
            self.client = OpenAIClient(api_key, self.config)
        
        elif self.config.provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv(self.config.api_key_env_var)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {self.config.api_key_env_var}")
            self.client = AnthropicClient(api_key, self.config)
        
        else:
            self.client = MockLLMClient()
    
    async def execute_single(
        self,
        prompt: PromptVariant,
        input_data: Dict[str, Any],
        context: Dict = None
    ) -> LLMOutput:
        """
        Execute a single prompt and collect output.
        
        Args:
            prompt: The prompt variant to execute
            input_data: Input data to include in the prompt
            context: Additional context
        
        Returns:
            LLMOutput with the response
        """
        context = context or {}
        
        # Format the full prompt with input data
        full_prompt = self._format_prompt(prompt.variant_prompt, input_data)
        
        # Rate limiting
        await self._rate_limiter.acquire(full_prompt)
        
        # Execute with retries
        start_time = time.time()
        output_text = None
        error_message = None
        retry_count = 0
        
        for attempt in range(self.config.max_retries):
            try:
                output_text = await self.client.generate(
                    full_prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                break
            except Exception as e:
                error_message = str(e)
                retry_count = attempt + 1
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Create output object
        return LLMOutput(
            prompt_id=prompt.id,
            prompt_text=full_prompt,
            input_data=input_data,
            output_text=output_text or "",
            model_name=self.config.model_name,
            tokens_used=self.client.get_token_count(output_text or ""),
            latency_ms=latency_ms,
            error_message=error_message,
            retry_count=retry_count
        )
    
    async def execute_batch(
        self,
        prompts: List[PromptVariant],
        inputs: List[Dict[str, Any]],
        contexts: List[Dict] = None
    ) -> List[LLMOutput]:
        """
        Execute multiple prompts in batch.
        
        Args:
            prompts: List of prompt variants
            inputs: List of input data (one per prompt)
            contexts: Optional list of contexts
        
        Returns:
            List of LLMOutput objects
        """
        contexts = contexts or [{} for _ in prompts]
        
        # Create tasks
        tasks = [
            self.execute_single(prompt, input_data, context)
            for prompt, input_data, context in zip(prompts, inputs, contexts)
        ]
        
        # Execute concurrently with rate limiting
        outputs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        results = []
        for i, output in enumerate(outputs):
            if isinstance(output, Exception):
                # Create error output
                results.append(LLMOutput(
                    prompt_id=prompts[i].id,
                    prompt_text=prompts[i].variant_prompt,
                    input_data=inputs[i],
                    output_text="",
                    model_name=self.config.model_name,
                    tokens_used=0,
                    latency_ms=0,
                    error_message=str(output),
                    retry_count=self.config.max_retries
                ))
            else:
                results.append(output)
        
        return results
    
    def _format_prompt(self, prompt_template: str, input_data: Dict) -> str:
        """Format prompt template with input data."""
        # Simple string formatting
        try:
            return prompt_template.format(**input_data)
        except KeyError:
            # If formatting fails, append input data
            input_str = "\\n".join(f"{k}: {v}" for k, v in input_data.items())
            return f"{prompt_template}\\n\\nInput Data:\\n{input_str}"
    
    def execute_sync(
        self,
        prompt: PromptVariant,
        input_data: Dict[str, Any],
        context: Dict = None
    ) -> LLMOutput:
        """Synchronous wrapper for execute_single."""
        return asyncio.run(self.execute_single(prompt, input_data, context))
    
    def execute_batch_sync(
        self,
        prompts: List[PromptVariant],
        inputs: List[Dict[str, Any]],
        contexts: List[Dict] = None
    ) -> List[LLMOutput]:
        """Synchronous wrapper for execute_batch."""
        return asyncio.run(self.execute_batch(prompts, inputs, contexts))


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Enforces:
    - Requests per minute
    - Tokens per minute
    """
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        
        # Token buckets
        self.request_tokens = float(requests_per_minute)
        self.token_tokens = float(tokens_per_minute)
        
        # Last update time
        self.last_update = time.time()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def acquire(self, text: str) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Replenish tokens
            self.request_tokens = min(self.rpm, self.request_tokens + elapsed * (self.rpm / 60))
            self.token_tokens = min(self.tpm, self.token_tokens + elapsed * (self.tpm / 60))
            
            self.last_update = now
            
            # Estimate tokens needed
            tokens_needed = len(text) // 4  # Rough estimate
            
            # Wait if necessary
            if self.request_tokens < 1:
                wait_time = (1 - self.request_tokens) / (self.rpm / 60)
                await asyncio.sleep(wait_time)
            
            if self.token_tokens < tokens_needed:
                wait_time = (tokens_needed - self.token_tokens) / (self.tpm / 60)
                await asyncio.sleep(wait_time)
            
            # Consume tokens
            self.request_tokens -= 1
            self.token_tokens -= tokens_needed
