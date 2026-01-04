"""
LLM Providers for Epistemic Governor

Local model support with automatic device detection (CUDA, MPS, CPU).
Designed for calibration and testing with real models.

Supported providers:
- HuggingFaceProvider: Any HF transformers model
- OllamaProvider: Local Ollama server
- MockProvider: Deterministic responses for testing

Device selection:
- Auto-detects CUDA (NVIDIA) or MPS (Apple Silicon)
- Falls back to CPU if neither available
- Can be overridden via device= parameter

Usage:
    from epistemic_governor.providers import HuggingFaceProvider, get_device
    
    # Auto device detection
    device = get_device()
    provider = HuggingFaceProvider("mistralai/Mistral-7B-Instruct-v0.2", device=device)
    
    # Use with session
    session = EpistemicSession(provider=provider)
    frame = session.step("What is the capital of France?")
"""

import os
import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

# Handle both package and direct imports
try:
    from .governor import GenerationEnvelope
except ImportError:
    from governor import GenerationEnvelope


# =============================================================================
# Device Detection
# =============================================================================

class DeviceType(Enum):
    """Available compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def get_device(preferred: Optional[str] = None) -> str:
    """
    Get the best available compute device.
    
    Priority: preferred > CUDA > MPS > CPU
    
    Args:
        preferred: Override automatic detection ("cuda", "mps", "cpu")
    
    Returns:
        Device string for torch
    """
    if preferred:
        return preferred
    
    try:
        import torch
        
        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            return "cuda"
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
    except ImportError:
        pass
    
    return "cpu"


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    info = {
        "selected": get_device(),
        "cuda_available": False,
        "mps_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
    }
    
    try:
        import torch
        
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        
        info["mps_available"] = (
            hasattr(torch.backends, 'mps') and 
            torch.backends.mps.is_available()
        )
        
    except ImportError:
        info["torch_installed"] = False
    
    return info


# =============================================================================
# Provider Base Class
# =============================================================================

class BaseProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from the model."""
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return model identifier for logging."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Return provider info for logging."""
        return {
            "model_id": self.get_model_id(),
            "provider_type": self.__class__.__name__,
        }


# =============================================================================
# HuggingFace Provider
# =============================================================================

class HuggingFaceProvider(BaseProvider):
    """
    Provider for HuggingFace transformers models.
    
    Supports any model that works with AutoModelForCausalLM.
    
    Popular choices for calibration:
    - mistralai/Mistral-7B-Instruct-v0.2
    - meta-llama/Llama-2-7b-chat-hf
    - microsoft/phi-2
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (fast, for testing)
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.device = device or get_device()
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        self.trust_remote_code = trust_remote_code
        
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFace provider requires transformers and torch. "
                "Install with: pip install transformers torch"
            )
        
        print(f"Loading {self.model_name} on {self.device}...")
        
        # Determine dtype
        if self.torch_dtype:
            dtype = getattr(torch, self.torch_dtype)
        elif self.device == "mps":
            dtype = torch.float16  # MPS works best with float16
        elif self.device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load model with quantization options
        load_kwargs = {
            "trust_remote_code": self.trust_remote_code,
        }
        
        if self.load_in_8bit or self.load_in_4bit:
            # Quantization requires bitsandbytes
            try:
                from transformers import BitsAndBytesConfig
                
                if self.load_in_4bit:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                    )
                else:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                print("Warning: bitsandbytes not available, loading full precision")
                load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype
        
        if self.max_memory:
            load_kwargs["max_memory"] = self.max_memory
            load_kwargs["device_map"] = "auto"
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        
        # Move to device if not using device_map
        if "device_map" not in load_kwargs:
            self._model = self._model.to(self.device)
        
        self._model.eval()
        self._loaded = True
        print(f"Model loaded on {self.device}")
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using the HuggingFace model."""
        import torch
        
        self._ensure_loaded()
        
        # Format prompt with system message if provided
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize
        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        # Move to device
        if hasattr(self._model, 'device'):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Apply envelope constraints to generation
        gen_kwargs = {
            "max_new_tokens": getattr(envelope, 'max_new_tokens', 256),
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        
        # Temperature from envelope
        if hasattr(envelope, 'temperature_floor'):
            # Use midpoint of allowed range
            temp = (envelope.temperature_floor + envelope.temperature_ceiling) / 2
            gen_kwargs["temperature"] = temp
        else:
            gen_kwargs["temperature"] = 0.7
        
        # Top-p if set
        if hasattr(envelope, 'top_p') and envelope.top_p:
            gen_kwargs["top_p"] = envelope.top_p
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        # Decode, removing the prompt
        generated = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated.strip()
    
    def get_model_id(self) -> str:
        return self.model_name
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "provider_type": "HuggingFaceProvider",
            "device": self.device,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
        }


# =============================================================================
# Ollama Provider
# =============================================================================

class OllamaProvider(BaseProvider):
    """
    Provider for local Ollama server.
    
    Requires Ollama to be running: https://ollama.ai
    
    Popular models:
    - llama2, llama2:7b, llama2:13b
    - mistral, mistral:7b
    - phi, phi:medium
    - codellama
    """
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using Ollama API."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "Ollama provider requires requests. "
                "Install with: pip install requests"
            )
        
        # Build request
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": getattr(envelope, 'max_new_tokens', 256),
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        # Temperature from envelope
        if hasattr(envelope, 'temperature_floor'):
            temp = (envelope.temperature_floor + envelope.temperature_ceiling) / 2
            data["options"]["temperature"] = temp
        
        # Top-p if set
        if hasattr(envelope, 'top_p') and envelope.top_p:
            data["options"]["top_p"] = envelope.top_p
        
        # Make request
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=data,
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
    
    def get_model_id(self) -> str:
        return f"ollama/{self.model_name}"
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "provider_type": "OllamaProvider",
            "base_url": self.base_url,
        }
    
    def list_models(self) -> List[str]:
        """List available models on the Ollama server."""
        import requests
        
        response = requests.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        
        models = response.json().get("models", [])
        return [m["name"] for m in models]


# =============================================================================
# OpenAI Provider
# =============================================================================

class OpenAIProvider(BaseProvider):
    """
    Provider for OpenAI API (GPT-4, GPT-3.5, etc.)
    
    Requires OPENAI_API_KEY environment variable or explicit api_key.
    
    Popular models:
    - gpt-4o (recommended for calibration)
    - gpt-4-turbo
    - gpt-3.5-turbo (faster, cheaper)
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # For Azure or compatible APIs
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._client = None
    
    def _ensure_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is not None:
            return
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install openai"
            )
        
        kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        self._client = OpenAI(**kwargs)
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using OpenAI API."""
        self._ensure_client()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Map envelope to OpenAI parameters
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": getattr(envelope, 'max_new_tokens', 256),
        }
        
        # Temperature from envelope
        if hasattr(envelope, 'temperature_floor') and hasattr(envelope, 'temperature_ceiling'):
            temp = (envelope.temperature_floor + envelope.temperature_ceiling) / 2
            kwargs["temperature"] = temp
        
        # Top-p if set
        if hasattr(envelope, 'top_p') and envelope.top_p:
            kwargs["top_p"] = envelope.top_p
        
        # Make request
        response = self._client.chat.completions.create(**kwargs)
        
        return response.choices[0].message.content.strip()
    
    def get_model_id(self) -> str:
        return f"openai/{self.model_name}"
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "provider_type": "OpenAIProvider",
            "base_url": self.base_url or "https://api.openai.com",
        }


# =============================================================================
# Anthropic Provider
# =============================================================================

class AnthropicProvider(BaseProvider):
    """
    Provider for Anthropic API (Claude models).
    
    Requires ANTHROPIC_API_KEY environment variable or explicit api_key.
    
    Popular models:
    - claude-sonnet-4-20250514 (recommended for calibration)
    - claude-3-5-sonnet-20241022
    - claude-3-haiku-20240307 (faster, cheaper)
    """
    
    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._client = None
    
    def _ensure_client(self):
        """Lazy-load the Anthropic client."""
        if self._client is not None:
            return
        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install anthropic"
            )
        
        kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        self._client = Anthropic(**kwargs)
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using Anthropic API."""
        self._ensure_client()
        
        # Map envelope to Anthropic parameters
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": getattr(envelope, 'max_new_tokens', 256),
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        # Temperature from envelope
        if hasattr(envelope, 'temperature_floor') and hasattr(envelope, 'temperature_ceiling'):
            temp = (envelope.temperature_floor + envelope.temperature_ceiling) / 2
            kwargs["temperature"] = temp
        
        # Top-p if set
        if hasattr(envelope, 'top_p') and envelope.top_p:
            kwargs["top_p"] = envelope.top_p
        
        # Make request
        response = self._client.messages.create(**kwargs)
        
        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        
        return "".join(text_parts).strip()
    
    def get_model_id(self) -> str:
        return f"anthropic/{self.model_name}"
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_name,
            "provider_type": "AnthropicProvider",
            "base_url": self.base_url or "https://api.anthropic.com",
        }


# =============================================================================
# Mock Provider (for testing)
# =============================================================================

class MockProvider(BaseProvider):
    """
    Deterministic provider for testing.
    
    Can be configured with:
    - Fixed responses per prompt
    - Confidence patterns
    - Intentional hallucinations
    - Drift behavior
    """
    
    def __init__(
        self,
        responses: Optional[Dict[str, str]] = None,
        default_response: str = "I don't know.",
        confidence_pattern: str = "medium",  # low, medium, high, random
        drift_on_challenge: bool = False,  # Whether to flip when challenged
        hallucinate_rate: float = 0.0,  # Probability of hallucinating
    ):
        self.responses = responses or {}
        self.default_response = default_response
        self.confidence_pattern = confidence_pattern
        self.drift_on_challenge = drift_on_challenge
        self.hallucinate_rate = hallucinate_rate
        self._call_count = 0
    
    def generate(
        self,
        prompt: str,
        envelope: GenerationEnvelope,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate deterministic response."""
        import random
        
        self._call_count += 1
        
        # Check for exact match
        prompt_lower = prompt.lower()
        for key, response in self.responses.items():
            if key.lower() in prompt_lower:
                return self._apply_confidence(response)
        
        # Check for challenge patterns (drift test)
        if self.drift_on_challenge:
            challenge_patterns = ["are you sure", "actually", "i think you're wrong", "i heard"]
            if any(p in prompt_lower for p in challenge_patterns):
                return "I apologize, you're correct. Let me reconsider."
        
        # Hallucinate with probability
        if random.random() < self.hallucinate_rate:
            return "The answer is definitely 42. I'm completely certain about this."
        
        return self._apply_confidence(self.default_response)
    
    def _apply_confidence(self, response: str) -> str:
        """Apply confidence pattern to response."""
        if self.confidence_pattern == "low":
            return f"I think {response.lower()}, but I'm not entirely sure."
        elif self.confidence_pattern == "high":
            return f"I'm absolutely certain: {response}"
        elif self.confidence_pattern == "random":
            import random
            patterns = [
                f"I believe {response.lower()}",
                f"Definitely: {response}",
                f"Maybe {response.lower()}?",
                response,
            ]
            return random.choice(patterns)
        return response
    
    def get_model_id(self) -> str:
        return "mock-provider"
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "model_id": "mock-provider",
            "provider_type": "MockProvider",
            "confidence_pattern": self.confidence_pattern,
            "drift_on_challenge": self.drift_on_challenge,
            "hallucinate_rate": self.hallucinate_rate,
        }


# =============================================================================
# Provider Factory
# =============================================================================

def create_provider(
    provider_type: str,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """
    Factory function to create providers.
    
    Args:
        provider_type: "huggingface", "ollama", "openai", "anthropic", or "mock"
        model_name: Model identifier
        **kwargs: Provider-specific arguments
    
    Returns:
        Configured provider instance
    
    Examples:
        # Local models
        provider = create_provider("ollama", "llama3")
        provider = create_provider("huggingface", "mistralai/Mistral-7B-Instruct-v0.2")
        
        # Cloud APIs
        provider = create_provider("openai", "gpt-4o")
        provider = create_provider("anthropic", "claude-sonnet-4-20250514")
        
        # Testing
        provider = create_provider("mock", confidence_pattern="high")
    """
    provider_type = provider_type.lower()
    
    if provider_type == "huggingface" or provider_type == "hf":
        if not model_name:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fast default
        return HuggingFaceProvider(model_name, **kwargs)
    
    elif provider_type == "ollama":
        if not model_name:
            model_name = "llama3"
        return OllamaProvider(model_name, **kwargs)
    
    elif provider_type == "openai" or provider_type == "gpt":
        if not model_name:
            model_name = "gpt-4o"
        return OpenAIProvider(model_name, **kwargs)
    
    elif provider_type == "anthropic" or provider_type == "claude":
        if not model_name:
            model_name = "claude-sonnet-4-20250514"
        return AnthropicProvider(model_name, **kwargs)
    
    elif provider_type == "mock":
        return MockProvider(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Valid options: huggingface, ollama, openai, anthropic, mock"
        )


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    print("=== Provider Demo ===\n")
    
    # Show device info
    print("1. Device Detection")
    info = get_device_info()
    print(f"   Selected device: {info['selected']}")
    print(f"   CUDA available: {info['cuda_available']}")
    print(f"   MPS available: {info['mps_available']}")
    if info.get('cuda_device_name'):
        print(f"   CUDA device: {info['cuda_device_name']}")
    
    # Test mock provider
    print("\n2. Mock Provider Test")
    mock = MockProvider(
        responses={
            "capital of france": "Paris is the capital of France.",
            "meaning of life": "That's a philosophical question.",
        },
        confidence_pattern="high",
        drift_on_challenge=True,
    )
    
    # Create a simple envelope
    envelope = GenerationEnvelope()
    
    response = mock.generate("What is the capital of France?", envelope)
    print(f"   Q: What is the capital of France?")
    print(f"   A: {response}")
    
    response = mock.generate("Are you sure? I heard it's London.", envelope)
    print(f"   Q: Are you sure? I heard it's London.")
    print(f"   A: {response}")
    
    print(f"\n   Provider info: {mock.get_info()}")
    
    # Show available providers
    print("\n3. Available Providers")
    print("   Local:")
    print("     - HuggingFaceProvider (requires transformers, torch)")
    print("     - OllamaProvider (requires local Ollama server)")
    print("   Cloud:")
    print("     - OpenAIProvider (requires openai, OPENAI_API_KEY)")
    print("     - AnthropicProvider (requires anthropic, ANTHROPIC_API_KEY)")
    print("   Testing:")
    print("     - MockProvider (no dependencies)")
    
    # Test cloud providers if keys available
    print("\n4. Cloud Provider Detection")
    
    if os.environ.get("OPENAI_API_KEY"):
        print("   ✓ OPENAI_API_KEY found")
        try:
            provider = create_provider("openai", "gpt-4o")
            print(f"     Created: {provider.get_model_id()}")
        except Exception as e:
            print(f"     Error: {e}")
    else:
        print("   ✗ OPENAI_API_KEY not set")
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("   ✓ ANTHROPIC_API_KEY found")
        try:
            provider = create_provider("anthropic", "claude-sonnet-4-20250514")
            print(f"     Created: {provider.get_model_id()}")
        except Exception as e:
            print(f"     Error: {e}")
    else:
        print("   ✗ ANTHROPIC_API_KEY not set")
    
    print("\n✓ Provider infrastructure working")
