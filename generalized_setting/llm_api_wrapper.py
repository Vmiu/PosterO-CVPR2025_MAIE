"""
LLM API Wrapper to replace vLLM
Mimics the vLLM interface (LLM.generate() and SamplingParams) using API calls
"""

import os
import time
from typing import List, Union, Optional
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    Sampling parameters for text generation.
    Maps to API parameters:
    - temperature: Controls randomness (0=deterministic, higher=more random)
    - max_tokens: Maximum tokens to generate
    - top_p: Nucleus sampling (0-1, filters tokens by cumulative probability)
    - frequency_penalty: Penalizes frequent tokens (-2.0 to 2.0)
    - presence_penalty: Penalizes tokens that have appeared (-2.0 to 2.0)
    - n: Number of completions to generate
    - stop: Stop sequences (list of strings)
    """
    temperature: float = 0.7
    max_tokens: int = 800
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    stop: Optional[List[str]] = None


class Output:
    """Mimics vLLM's output structure"""
    def __init__(self, text: str):
        self.text = text


class RequestOutput:
    """Mimics vLLM's request output structure"""
    def __init__(self, outputs: List[Output]):
        self.outputs = outputs


class LLM:
    """
    LLM wrapper that uses API calls instead of vLLM.
    Supports poe.com API for Claude-Sonnet-4.5.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM API wrapper for poe.com.
        
        Args:
            api_key: Poe.com API key (if None, will try to get from POE_API_KEY environment variable)
        """
        self.model_name = "claude-sonnet-4.5"
        self.api_key = "821sV1Vur7mAM2d-Pr0qqqPUoee-Y6005BIqUxnWIOA"
        self.base_url = "https://api.poe.com/v1"
        
        if not self.api_key:
            raise ValueError(
                "Poe.com API key not provided. Set POE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Poe.com uses OpenAI-compatible API, so we can use the OpenAI client
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError(
                "requests package not installed. Install it with: pip install requests"
            )
    
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        sampling_params: SamplingParams
    ) -> List[RequestOutput]:
        """
        Generate text using poe.com API.
        
        Args:
            prompts: Single prompt string or list of prompts
            sampling_params: SamplingParams object with generation parameters
            
        Returns:
            List[RequestOutput] - Always returns a list to match vLLM interface
        """
        # Handle both single prompt and batch
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else prompts
        
        results = []
        
        for prompt in prompt_list:
            outputs = self._generate_poe(prompt, sampling_params)
            results.append(RequestOutput(outputs))
        
        return results
    
    def _make_request_with_retry(self, url: str, headers: dict, data: dict, max_retries: int = 3, initial_delay: float = 1.0):
        """
        Make an API request with exponential backoff retry for transient errors.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            data: Request payload
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds before first retry
            
        Returns:
            Response object
            
        Raises:
            RuntimeError: If all retries are exhausted or non-retryable error occurs
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            response = None
            try:
                response = self.requests.post(url, headers=headers, json=data, timeout=120)
                response.raise_for_status()
                return response
                
            except self.requests.exceptions.HTTPError as e:
                # Check if it's a retryable error (5xx server errors)
                if response is not None and 500 <= response.status_code < 600:
                    # Retryable server error
                    if attempt < max_retries:
                        delay = initial_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Server error {response.status_code}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})...")
                        time.sleep(delay)
                        last_exception = e
                        continue
                    else:
                        # All retries exhausted
                        error_msg = str(e)
                        try:
                            error_detail = response.json()
                            error_msg += f"\nAPI Error Details: {error_detail}"
                        except:
                            try:
                                error_msg += f"\nResponse Text: {response.text[:500]}"  # Limit text length
                            except:
                                pass
                        raise RuntimeError(f"Poe.com API HTTP error after {max_retries + 1} attempts: {error_msg}")
                else:
                    # Non-retryable error (4xx client errors)
                    error_msg = str(e)
                    if response is not None:
                        try:
                            error_detail = response.json()
                            error_msg += f"\nAPI Error Details: {error_detail}"
                        except:
                            try:
                                error_msg += f"\nResponse Text: {response.text[:500]}"  # Limit text length
                            except:
                                pass
                    raise RuntimeError(f"Poe.com API HTTP error: {error_msg}")
                    
            except self.requests.exceptions.RequestException as e:
                # Network/connection errors - retry these
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)
                    print(f"Request error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})...")
                    time.sleep(delay)
                    last_exception = e
                    continue
                else:
                    raise RuntimeError(f"Poe.com API request error after {max_retries + 1} attempts: {e}")
            except Exception as e:
                # Non-retryable errors
                raise RuntimeError(f"Poe.com API error: {e}")
        
        # Should not reach here, but just in case
        if last_exception:
            raise RuntimeError(f"Poe.com API error after {max_retries + 1} attempts: {last_exception}")
        raise RuntimeError("Unexpected error in retry logic")
    
    def _generate_poe(self, prompt: str, sampling_params: SamplingParams) -> List[Output]:
        """Generate using poe.com API (OpenAI-compatible)"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Map SamplingParams to API parameters
        # Poe.com uses OpenAI-compatible API format
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "top_p": sampling_params.top_p,
            "frequency_penalty": sampling_params.frequency_penalty,
            "presence_penalty": sampling_params.presence_penalty,
            "n": sampling_params.n,
        }
        
        if sampling_params.stop:
            # Convert to list if needed and filter out empty/whitespace-only sequences
            stop_list = sampling_params.stop if isinstance(sampling_params.stop, list) else [sampling_params.stop]
            # Filter out empty strings and whitespace-only strings
            stop_list = [s for s in stop_list if s and s.strip()]
            if stop_list:  # Only add stop if there are valid stop sequences
                data["stop"] = stop_list
        
        outputs = []
        
        # Try to get all n completions in one call first
        try:
            response = self._make_request_with_retry(url, headers, data)
            result = response.json()
            
            # Handle OpenAI-compatible response format
            if "choices" in result:
                for choice in result["choices"]:
                    if "message" in choice and "content" in choice["message"]:
                        outputs.append(Output(choice["message"]["content"]))
                
                # If we got all n results, return them
                if len(outputs) >= sampling_params.n:
                    return outputs[:sampling_params.n]
            else:
                raise RuntimeError(f"Unexpected response format: {result}")
        except Exception as e:
            # If first call fails completely, we'll try individual calls below
            # But if we have no outputs, we need to handle the error
            if len(outputs) == 0:
                # Re-raise if we have no outputs at all
                raise
        
        # If we didn't get enough results, make additional calls
        # (Set n=1 for subsequent calls to avoid getting duplicates)
        data["n"] = 1
        while len(outputs) < sampling_params.n:
            try:
                response = self._make_request_with_retry(url, headers, data)
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        outputs.append(Output(choice["message"]["content"]))
                else:
                    break  # Stop if we can't get more results
                    
            except Exception as e:
                # If we already have some outputs, return what we have
                if len(outputs) > 0:
                    print(f"Warning: Got error while fetching additional outputs, returning {len(outputs)} outputs: {e}")
                    break
                else:
                    # No outputs yet, re-raise the error
                    raise
        
        return outputs[:sampling_params.n]
