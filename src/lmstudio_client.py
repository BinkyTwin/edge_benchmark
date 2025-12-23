"""
LM Studio API Client
====================

REST API client for LM Studio with performance metrics support:
- TTFT (Time To First Token)
- Tokens per second (output/prompt)
- Streaming support
- Structured output (JSON mode)
"""

import time
import json
from dataclasses import dataclass, field
from typing import Optional, Generator, Any
from pathlib import Path

import httpx
import yaml
from openai import OpenAI


@dataclass
class CompletionMetrics:
    """Metrics collected during a completion request."""
    
    # Timing
    ttft_ms: float = 0.0                    # Time to first token (ms)
    total_time_ms: float = 0.0              # Total generation time (ms)
    
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Throughput
    output_tokens_per_sec: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    
    # Response
    finish_reason: str = ""
    model: str = ""
    
    # Raw response stats (from LM Studio)
    raw_stats: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert metrics to a dictionary."""
        return {
            "ttft_ms": self.ttft_ms,
            "total_time_ms": self.total_time_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "output_tokens_per_sec": self.output_tokens_per_sec,
            "prompt_tokens_per_sec": self.prompt_tokens_per_sec,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "raw_stats": self.raw_stats,
        }


@dataclass
class CompletionResult:
    """Completion result with associated metrics."""
    
    content: str
    metrics: CompletionMetrics
    success: bool = True
    error: Optional[str] = None
    
    # For structured output
    json_valid: bool = False
    parsed_json: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert result to a dictionary."""
        return {
            "content": self.content,
            "metrics": self.metrics.to_dict(),
            "success": self.success,
            "error": self.error,
            "json_valid": self.json_valid,
            "parsed_json": self.parsed_json,
        }


class LMStudioClient:
    """
    Client for the LM Studio API.
    
    Compatible with the OpenAI API for chat completions.
    Automatically collects performance metrics.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: float = 120.0,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize the LM Studio client.
        
        Args:
            base_url: Base URL of the LM Studio API
            timeout: Timeout in seconds for requests
            config_path: Path to the models configuration file
        """
        self.base_url = base_url
        self.timeout = timeout
        
        # OpenAI-compatible client
        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LM Studio does not require an API key
            timeout=timeout,
        )
        
        # HTTP client for direct requests
        self.http_client = httpx.Client(
            base_url=base_url.replace("/v1", ""),
            timeout=timeout,
        )
        
        # Load model configuration if provided
        self.models_config = {}
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.models_config = yaml.safe_load(f)
    
    def list_models(self) -> list[dict]:
        """List available models on LM Studio."""
        try:
            response = self.client.models.list()
            return [{"id": m.id, "object": m.object} for m in response.data]
        except Exception as e:
            return [{"error": str(e)}]
    
    def get_model_info(self, model_id: str) -> dict:
        """Retrieve information about a model."""
        try:
            response = self.http_client.get(f"/v1/models/{model_id}")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def load_model(self, model_id: str) -> bool:
        """
        Load a model in LM Studio.
        
        Note: Requires LM Studio to support loading via API.
        """
        try:
            # LM Studio automatically loads the model on the first request
            # We make a minimal request to force loading
            self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
    
    def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0,
        top_p: float = 1,
        max_tokens: int = 512,
        stream: bool = True,
        response_format: Optional[dict] = None,
        stop: Optional[list[str]] = None,
        **kwargs,
    ) -> CompletionResult:
        """
        Perform a completion with metrics collection.
        
        Args:
            model: LM Studio model ID
            messages: List of messages (OpenAI format)
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum number of output tokens
            stream: Use streaming (recommended for TTFT measurement)
            response_format: Response format (e.g., {"type": "json_object"})
            stop: Stop sequences
            
        Returns:
            CompletionResult with content and metrics
        """
        metrics = CompletionMetrics(model=model)
        
        try:
            start_time = time.perf_counter()
            first_token_time = None
            content_chunks = []
            
            # Build base parameters
            # Note: LM Studio does not accept response_format=null, we omit it if None
            base_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            
            # Add optional parameters only if defined
            if response_format is not None:
                base_params["response_format"] = response_format
            if stop is not None:
                base_params["stop"] = stop
            
            # Add kwargs
            base_params.update(kwargs)
            
            if stream:
                # Streaming mode for precise TTFT measurement
                response = self.client.chat.completions.create(
                    stream=True,
                    **base_params,
                )
                
                for chunk in response:
                    if first_token_time is None and chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            first_token_time = time.perf_counter()
                            metrics.ttft_ms = (first_token_time - start_time) * 1000
                    
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunks.append(chunk.choices[0].delta.content)
                    
                    # Get finish_reason from the last chunk
                    if chunk.choices and chunk.choices[0].finish_reason:
                        metrics.finish_reason = chunk.choices[0].finish_reason
                
                content = "".join(content_chunks)
                
            else:
                # Non-streaming mode
                response = self.client.chat.completions.create(
                    stream=False,
                    **base_params,
                )
                
                first_token_time = time.perf_counter()
                metrics.ttft_ms = (first_token_time - start_time) * 1000
                
                content = response.choices[0].message.content or ""
                metrics.finish_reason = response.choices[0].finish_reason or ""
                
                # Usage metrics
                if response.usage:
                    metrics.prompt_tokens = response.usage.prompt_tokens
                    metrics.completion_tokens = response.usage.completion_tokens
                    metrics.total_tokens = response.usage.total_tokens
            
            end_time = time.perf_counter()
            metrics.total_time_ms = (end_time - start_time) * 1000
            
            # Calculate throughput
            if metrics.total_time_ms > 0:
                # Estimate completion tokens if not provided
                if metrics.completion_tokens == 0:
                    # Rough estimation: ~4 characters per token
                    metrics.completion_tokens = len(content) // 4
                
                generation_time_sec = (end_time - (first_token_time or start_time))
                if generation_time_sec > 0:
                    metrics.output_tokens_per_sec = metrics.completion_tokens / generation_time_sec
                
                if first_token_time and metrics.prompt_tokens > 0:
                    prompt_time_sec = first_token_time - start_time
                    if prompt_time_sec > 0:
                        metrics.prompt_tokens_per_sec = metrics.prompt_tokens / prompt_time_sec
            
            # Check if valid JSON (for structured output)
            # Supports json_object (OpenAI) and json_schema (LM Studio)
            json_valid = False
            parsed_json = None
            if response_format and response_format.get("type") in ("json_object", "json_schema"):
                try:
                    parsed_json = json.loads(content)
                    json_valid = True
                except json.JSONDecodeError:
                    json_valid = False
            
            return CompletionResult(
                content=content,
                metrics=metrics,
                success=True,
                json_valid=json_valid,
                parsed_json=parsed_json,
            )
            
        except Exception as e:
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
            return CompletionResult(
                content="",
                metrics=metrics,
                success=False,
                error=str(e),
            )
    
    def complete_with_retry(
        self,
        model: str,
        messages: list[dict],
        max_retries: int = 3,
        **kwargs,
    ) -> CompletionResult:
        """
        Completion with automatic retry on error.
        """
        last_error = None
        for attempt in range(max_retries):
            result = self.complete(model=model, messages=messages, **kwargs)
            if result.success:
                return result
            last_error = result.error
            time.sleep(1 * (attempt + 1))  # Simple exponential backoff
        
        return CompletionResult(
            content="",
            metrics=CompletionMetrics(model=model),
            success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
        )
    
    def warmup(self, model: str, num_requests: int = 3) -> list[CompletionMetrics]:
        """
        Perform warm-up requests (not counted in benchmarks).
        
        Args:
            model: Model ID
            num_requests: Number of warm-up requests
            
        Returns:
            List of metrics from warm-up requests
        """
        warmup_metrics = []
        warmup_message = [{"role": "user", "content": "Hello, this is a warmup request."}]
        
        for _ in range(num_requests):
            result = self.complete(
                model=model,
                messages=warmup_message,
                max_tokens=20,
                stream=True,
            )
            warmup_metrics.append(result.metrics)
            time.sleep(0.5)
        
        return warmup_metrics
    
    def batch_complete(
        self,
        model: str,
        prompts: list[list[dict]],
        cooldown: float = 1.0,
        **kwargs,
    ) -> Generator[CompletionResult, None, None]:
        """
        Perform multiple completions with cooldown between each.
        
        Args:
            model: Model ID
            prompts: List of message lists
            cooldown: Pause time between requests (seconds)
            
        Yields:
            CompletionResult for each prompt
        """
        for i, messages in enumerate(prompts):
            result = self.complete(model=model, messages=messages, **kwargs)
            yield result
            
            if i < len(prompts) - 1:
                time.sleep(cooldown)
    
    def health_check(self) -> dict:
        """Check the health of the LM Studio server."""
        try:
            response = self.http_client.get("/v1/models")
            return {
                "status": "healthy",
                "models_available": response.status_code == 200,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def close(self):
        """Close HTTP connections."""
        self.http_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Utility functions

def format_messages(
    user_content: str,
    system_prompt: Optional[str] = None,
) -> list[dict]:
    """
    Format messages for the chat API.
    
    Args:
        user_content: User message content
        system_prompt: Optional system prompt
        
    Returns:
        List of formatted messages
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def create_classification_prompt(
    text: str,
    labels: list[str],
    instruction: Optional[str] = None,
) -> str:
    """
    Create a prompt for a classification task.
    
    Args:
        text: Text to classify
        labels: Possible labels
        instruction: Custom instruction
        
    Returns:
        Formatted prompt
    """
    if instruction is None:
        instruction = "Classify the following text."
    
    labels_str = ", ".join(labels)
    return f"{instruction}\n\nText: {text}\n\nPossible labels: {labels_str}\n\nLabel:"


