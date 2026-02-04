"""
Multi-Model Router for Open Source LLMs
Intelligently routes to best model based on task type and availability
Supports: OpenRouter, Hugging Face Inference API, Local Ollama
"""

import os
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    REASONING = "reasoning"           # Complex planning, logic
    CODING = "coding"                 # Code generation, debugging
    LONG_CONTEXT = "long_context"     # Document analysis, RAG
    MULTIMODAL = "multimodal"         # Vision + text
    FAST = "fast"                     # Quick responses, low latency
    GENERAL = "general"               # Default tasks


@dataclass
class ModelConfig:
    name: str
    provider: str                      # openrouter, huggingface, ollama
    task_types: List[TaskType]
    context_length: int
    cost_per_1k: float                 # USD, 0 for free/local
    api_endpoint: Optional[str] = None
    requires_auth: bool = True
    fallback_models: List[str] = None


class LLMRouter:
    """
    Intelligent router for open-source models
    Auto-selects best model for task, handles fallbacks
    """
    
    # Model registry - prioritized by capability
    MODEL_REGISTRY = {
        # Reasoning & General Intelligence
        "gpt-oss-120b": ModelConfig(
            name="openai/gpt-oss-120b",
            provider="openrouter",
            task_types=[TaskType.REASONING, TaskType.CODING, TaskType.GENERAL],
            context_length=128000,
            cost_per_1k=0.0,
            fallback_models=["gpt-oss-20b", "deepseek-r1"]
        ),
        "gpt-oss-20b": ModelConfig(
            name="openai/gpt-oss-20b",
            provider="openrouter",
            task_types=[TaskType.REASONING, TaskType.GENERAL],
            context_length=128000,
            cost_per_1k=0.0,
            fallback_models=["deepseek-r1"]
        ),
        "deepseek-r1": ModelConfig(
            name="deepseek/deepseek-r1",
            provider="openrouter",
            task_types=[TaskType.REASONING, TaskType.CODING],
            context_length=64000,
            cost_per_1k=0.0,
            fallback_models=["deepseek-v3", "qwen3-next"]
        ),
        "deepseek-v3": ModelConfig(
            name="deepseek/deepseek-chat",
            provider="openrouter",
            task_types=[TaskType.REASONING, TaskType.GENERAL, TaskType.CODING],
            context_length=64000,
            cost_per_1k=0.0,
            fallback_models=["qwen3-next"]
        ),
        "qwen3-next": ModelConfig(
            name="qwen/qwen3-next",
            provider="openrouter",
            task_types=[TaskType.REASONING, TaskType.LONG_CONTEXT, TaskType.GENERAL],
            context_length=128000,
            cost_per_1k=0.0,
            fallback_models=["llama-3-70b"]
        ),
        
        # Coding Specialists
        "qwen2.5-coder-32b": ModelConfig(
            name="qwen/qwen2.5-coder-32b-instruct",
            provider="openrouter",
            task_types=[TaskType.CODING],
            context_length=32000,
            cost_per_1k=0.0,
            fallback_models=["deepseek-coder-v2", "codellama-34b"]
        ),
        "deepseek-coder-v2": ModelConfig(
            name="deepseek/deepseek-coder-v2",
            provider="openrouter",
            task_types=[TaskType.CODING],
            context_length=64000,
            cost_per_1k=0.0,
            fallback_models=["qwen2.5-coder-7b"]
        ),
        "qwen2.5-coder-7b": ModelConfig(
            name="qwen/qwen2.5-coder-7b-instruct",
            provider="openrouter",
            task_types=[TaskType.CODING, TaskType.FAST],
            context_length=32000,
            cost_per_1k=0.0,
            fallback_models=["wavecoder-ultra", "codellama-7b"]
        ),
        "wavecoder-ultra": ModelConfig(
            name="michaelwnc/wavecoder-ultra-6.7b",
            provider="openrouter",
            task_types=[TaskType.CODING, TaskType.FAST],
            context_length=16000,
            cost_per_1k=0.0,
            fallback_models=["codellama-7b"]
        ),
        "codellama-34b": ModelConfig(
            name="codellama/codellama-34b-instruct",
            provider="openrouter",
            task_types=[TaskType.CODING],
            context_length=16000,
            cost_per_1k=0.0,
            fallback_models=["codellama-7b"]
        ),
        "codellama-7b": ModelConfig(
            name="codellama/codellama-7b-instruct",
            provider="openrouter",
            task_types=[TaskType.CODING, TaskType.FAST],
            context_length=16000,
            cost_per_1k=0.0,
            fallback_models=["mistral-7b"]
        ),
        
        # Long Context & Memory
        "llama-3-70b": ModelConfig(
            name="meta-llama/llama-3-70b-instruct",
            provider="openrouter",
            task_types=[TaskType.LONG_CONTEXT, TaskType.REASONING, TaskType.GENERAL],
            context_length=8000,
            cost_per_1k=0.0,
            fallback_models=["gemma-2-27b"]
        ),
        "gemma-2-27b": ModelConfig(
            name="google/gemma-2-27b-it",
            provider="openrouter",
            task_types=[TaskType.LONG_CONTEXT, TaskType.GENERAL],
            context_length=8000,
            cost_per_1k=0.0,
            fallback_models=["gemma-3-4b"]
        ),
        "gemma-3-4b": ModelConfig(
            name="google/gemma-3-4b-it",
            provider="openrouter",
            task_types=[TaskType.LONG_CONTEXT, TaskType.FAST],
            context_length=32000,
            cost_per_1k=0.0,
            fallback_models=["mistral-7b"]
        ),
        
        # Multimodal
        "qwen2.5-vl": ModelConfig(
            name="qwen/qwen2.5-vl-72b-instruct",
            provider="openrouter",
            task_types=[TaskType.MULTIMODAL],
            context_length=32000,
            cost_per_1k=0.0,
            fallback_models=["gemma-vision"]
        ),
        "qwen3-omni": ModelConfig(
            name="qwen/qwen3-omni",
            provider="openrouter",
            task_types=[TaskType.MULTIMODAL],
            context_length=32000,
            cost_per_1k=0.0,
            fallback_models=["qwen2.5-vl"]
        ),
        
        # Lightweight & Fast
        "mistral-7b": ModelConfig(
            name="mistralai/mistral-7b-instruct",
            provider="openrouter",
            task_types=[TaskType.FAST, TaskType.GENERAL],
            context_length=8000,
            cost_per_1k=0.0,
            fallback_models=["mixtral-8x7b", "phi-3.5"]
        ),
        "mixtral-8x7b": ModelConfig(
            name="mistralai/mixtral-8x7b-instruct",
            provider="openrouter",
            task_types=[TaskType.FAST, TaskType.REASONING],
            context_length=32000,
            cost_per_1k=0.0,
            fallback_models=["zephyr-7b"]
        ),
        "phi-3.5": ModelConfig(
            name="microsoft/phi-3.5-mini-instruct",
            provider="openrouter",
            task_types=[TaskType.FAST, TaskType.GENERAL],
            context_length=128000,
            cost_per_1k=0.0,
            fallback_models=["nemotron-1.5b"]
        ),
        "zephyr-7b": ModelConfig(
            name="huggingfaceh4/zephyr-7b-beta",
            provider="openrouter",
            task_types=[TaskType.FAST, TaskType.GENERAL],
            context_length=32000,
            cost_per_1k=0.0,
            fallback_models=["nemotron-1.5b"]
        ),
        "nemotron-1.5b": ModelConfig(
            name="nvidia/llama-3.1-nemotron-1.5b-instruct",
            provider="openrouter",
            task_types=[TaskType.FAST],
            context_length=128000,
            cost_per_1k=0.0,
            fallback_models=[]
        ),
        
        # Local Ollama (always available fallback)
        "ollama-default": ModelConfig(
            name="codellama:34b",
            provider="ollama",
            task_types=[TaskType.CODING, TaskType.GENERAL, TaskType.REASONING],
            context_length=16000,
            cost_per_1k=0.0,
            requires_auth=False,
            fallback_models=[]
        ),
    }
    
    # Task to preferred model mapping
    TASK_MODELS = {
        TaskType.REASONING: ["gpt-oss-120b", "deepseek-r1", "deepseek-v3", "qwen3-next"],
        TaskType.CODING: ["qwen2.5-coder-32b", "deepseek-coder-v2", "deepseek-r1", "gpt-oss-120b"],
        TaskType.LONG_CONTEXT: ["qwen3-next", "phi-3.5", "gemma-3-4b"],
        TaskType.MULTIMODAL: ["qwen3-omni", "qwen2.5-vl"],
        TaskType.FAST: ["mistral-7b", "phi-3.5", "zephyr-7b", "nemotron-1.5b"],
        TaskType.GENERAL: ["deepseek-v3", "gpt-oss-20b", "mistral-7b"]
    }
    
    def __init__(self):
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Track model availability
        self.available_models: Dict[str, bool] = {}
        self.failure_counts: Dict[str, int] = {}
        
        # Session for HTTP requests
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120),
                headers={
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://flexcyz.dev",
                    "X-Title": "Flexcyz IDE"
                }
            )
        return self._session
    
    def detect_task_type(self, messages: List[Dict], json_mode: bool = False) -> TaskType:
        """
        Analyze messages to determine optimal task type
        """
        content = " ".join([m.get("content", "") for m in messages]).lower()
        
        # Coding detection
        code_keywords = ["code", "function", "class", "component", "api", "database", 
                        "react", "python", "javascript", "typescript", "bug", "fix",
                        "implement", "refactor", "debug"]
        if any(kw in content for kw in code_keywords):
            return TaskType.CODING
        
        # Reasoning/Planning detection
        reasoning_keywords = ["plan", "design", "architecture", "analyze", "compare",
                            "strategy", "optimize", "structure", "organize"]
        if any(kw in content for kw in reasoning_keywords):
            return TaskType.REASONING
        
        # Long context detection
        total_length = sum(len(m.get("content", "")) for m in messages)
        if total_length > 8000:
            return TaskType.LONG_CONTEXT
        
        # Fast detection (simple queries)
        if len(content) < 200 and not json_mode:
            return TaskType.FAST
        
        return TaskType.GENERAL
    
    def select_model(self, task_type: TaskType, prefer_cheap: bool = False) -> str:
        """
        Select best available model for task
        """
        candidates = self.TASK_MODELS.get(task_type, self.TASK_MODELS[TaskType.GENERAL])
        
        # Filter by availability and failure count
        available = []
        for model_id in candidates:
            config = self.MODEL_REGISTRY[model_id]
            
            # Skip if too many recent failures
            if self.failure_counts.get(model_id, 0) > 3:
                continue
            
            # Check provider availability
            if config.provider == "openrouter" and not self.openrouter_key:
                continue
            if config.provider == "huggingface" and not self.hf_token:
                continue
            
            available.append(model_id)
        
        if not available:
            # Fall back to Ollama local
            return "ollama-default"
        
        if prefer_cheap:
            # Sort by cost (all 0 for these open models, but keep for future)
            available.sort(key=lambda m: self.MODEL_REGISTRY[m].cost_per_1k)
        
        return available[0]
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        task_type: Optional[TaskType] = None
    ) -> str:
        """
        Main completion method with auto-routing and fallbacks
        """
        # Auto-detect task type if not specified
        if task_type is None:
            task_type = self.detect_task_type(messages, json_mode)
        
        # Select model
        if model is None or model not in self.MODEL_REGISTRY:
            model = self.select_model(task_type)
        
        config = self.MODEL_REGISTRY[model]
        
        # Attempt completion with retries and fallbacks
        attempted = set()
        current_model = model
        
        while current_model and current_model not in attempted:
            attempted.add(current_model)
            config = self.MODEL_REGISTRY[current_model]
            
            try:
                if config.provider == "openrouter":
                    result = await self._openrouter_complete(
                        config.name, messages, temperature, max_tokens, json_mode
                    )
                elif config.provider == "huggingface":
                    result = await self._hf_complete(
                        config.name, messages, temperature, max_tokens
                    )
                elif config.provider == "ollama":
                    result = await self._ollama_complete(
                        config.name, messages, temperature, max_tokens
                    )
                else:
                    raise ValueError(f"Unknown provider: {config.provider}")
                
                # Success - reset failure count
                self.failure_counts[current_model] = 0
                return result
                
            except Exception as e:
                print(f"⚠️ Model {current_model} failed: {e}")
                self.failure_counts[current_model] = self.failure_counts.get(current_model, 0) + 1
                
                # Try fallback
                if config.fallback_models:
                    for fallback in config.fallback_models:
                        if fallback not in attempted:
                            current_model = fallback
                            print(f"🔄 Falling back to {fallback}")
                            break
                    else:
                        current_model = None
                else:
                    current_model = None
        
        # All options exhausted
        raise Exception(f"All models failed for task. Attempted: {attempted}")
    
    async def _openrouter_complete(
        self,
        model_name: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        json_mode: bool
    ) -> str:
        """Call OpenRouter API"""
        session = await self._get_session()
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.openrouter_key}"},
            json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"OpenRouter error {resp.status}: {text}")
            
            data = await resp.json()
            
            if "error" in data:
                raise Exception(f"OpenRouter API error: {data['error']}")
            
            return data["choices"][0]["message"]["content"]
    
    async def _hf_complete(
        self,
        model_name: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call Hugging Face Inference API"""
        session = await self._get_session()
        
        # Convert messages to HF format
        prompt = self._messages_to_prompt(messages)
        
        async with session.post(
            f"https://api-inference.huggingface.co/models/{model_name}",
            headers={"Authorization": f"Bearer {self.hf_token}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "return_full_text": False
                }
            }
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"HF error {resp.status}: {text}")
            
            data = await resp.json()
            return data[0]["generated_text"]
    
    async def _ollama_complete(
        self,
        model_name: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Call local Ollama instance"""
        session = await self._get_session()
        
        async with session.post(
            f"{self.ollama_host}/api/chat",
            json={
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Ollama error {resp.status}: {text}")
            
            data = await resp.json()
            return data["message"]["content"]
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to single prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")
        
        prompt_parts.append("<|assistant|>\n")
        return "\n".join(prompt_parts)
    
    async def close(self):
        """Cleanup"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def get_model_info(self) -> Dict:
        """Get info about available models"""
        return {
            "registered": len(self.MODEL_REGISTRY),
            "available": sum(1 for m in self.MODEL_REGISTRY.values() 
                           if (m.provider == "openrouter" and self.openrouter_key) or
                              (m.provider == "huggingface" and self.hf_token) or
                              m.provider == "ollama"),
            "by_task": {
                task.value: [m for m in models if m in self.MODEL_REGISTRY]
                for task, models in self.TASK_MODELS.items()
            }
        }
