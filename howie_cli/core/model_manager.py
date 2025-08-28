"""
Model management system for multi-model support
Allows switching between different AI models based on task requirements
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import os
import json
from pathlib import Path
import aiohttp
from openai import AsyncOpenAI
import anthropic


class ModelProvider(str, Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    PERPLEXITY = "perplexity"
    LOCAL = "local"  # For local models like Ollama


class ModelTier(str, Enum):
    """Model tier for cost/performance optimization"""
    PREMIUM = "premium"      # Most capable, most expensive (GPT-4o, Claude Opus)
    STANDARD = "standard"    # Good balance (GPT-4, Claude Sonnet)
    FAST = "fast"           # Fast and cheap (GPT-3.5, Claude Haiku)
    RESEARCH = "research"    # Specialized for research (Perplexity)


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_name: str
    tier: ModelTier
    api_key_env: str
    max_tokens: int = 4096
    temperature: float = 0.7
    supports_tools: bool = True
    supports_vision: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    best_for: List[str] = None
    
    def __post_init__(self):
        if self.best_for is None:
            self.best_for = []


class ModelManager:
    """Manages multiple AI models and routes tasks to appropriate models"""
    
    # Default model configurations
    DEFAULT_MODELS = {
        # OpenAI Models
        "gpt-4o": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o",
            tier=ModelTier.PREMIUM,
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=True,
            supports_vision=True,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            best_for=["analysis", "code_generation", "complex_reasoning"]
        ),
        "gpt-4o-mini": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            tier=ModelTier.FAST,
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=True,
            supports_vision=True,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            best_for=["simple_queries", "data_extraction", "classification"]
        ),
        "gpt-4-turbo": ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4-turbo-preview",
            tier=ModelTier.STANDARD,
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=True,
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            best_for=["analysis", "reasoning"]
        ),
        
        # Anthropic Models
        "claude-3-opus": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            tier=ModelTier.PREMIUM,
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=True,
            supports_vision=True,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            best_for=["complex_analysis", "creative_tasks", "long_context"]
        ),
        "claude-3-5-sonnet": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-5-sonnet-20241022",
            tier=ModelTier.STANDARD,
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=8192,
            temperature=0.7,
            supports_tools=True,
            supports_vision=True,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            best_for=["balanced_tasks", "code_generation", "analysis"]
        ),
        "claude-sonnet-4": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-sonnet-4-20250514",
            tier=ModelTier.STANDARD,
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=8192,
            temperature=0.7,
            supports_tools=True,
            supports_vision=True,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            best_for=["balanced_tasks", "code_generation", "analysis"]
        ),
        # NOTE: The "-4" variant is not a valid public model name; using official 3.5 Sonnet
        "claude-3-haiku": ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            tier=ModelTier.FAST,
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=True,
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.00125,
            best_for=["quick_tasks", "classification", "extraction"]
        ),
        
        # Perplexity Models (specialized for research)
        "perplexity-sonar": ModelConfig(
            provider=ModelProvider.PERPLEXITY,
            model_name="sonar",
            tier=ModelTier.RESEARCH,
            api_key_env="PERPLEXITY_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=False,
            cost_per_1k_input=0.0006,
            cost_per_1k_output=0.0018,
            best_for=["research", "current_events", "fact_checking"]
        ),
        "perplexity-sonar-pro": ModelConfig(
            provider=ModelProvider.PERPLEXITY,
            model_name="sonar-pro",
            tier=ModelTier.RESEARCH,
            api_key_env="PERPLEXITY_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=False,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.003,
            best_for=["deep_research", "current_events", "comprehensive_search"]
        ),
        "perplexity-sonar-reasoning": ModelConfig(
            provider=ModelProvider.PERPLEXITY,
            model_name="sonar-reasoning",
            tier=ModelTier.RESEARCH,
            api_key_env="PERPLEXITY_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=False,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.003,
            best_for=["reasoning", "analysis", "complex_queries"]
        ),
        "perplexity-sonar-deep-research": ModelConfig(
            provider=ModelProvider.PERPLEXITY,
            model_name="sonar-deep-research",
            tier=ModelTier.RESEARCH,
            api_key_env="PERPLEXITY_API_KEY",
            max_tokens=4096,
            temperature=0.7,
            supports_tools=False,
            cost_per_1k_input=0.002,
            cost_per_1k_output=0.006,
            best_for=["comprehensive_research", "detailed_reports", "in_depth_analysis"]
        ),
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize model manager with optional custom config"""
        self.models = self.DEFAULT_MODELS.copy()
        self.clients = {}
        self.current_model = "claude-sonnet-4"  # Default model
        self.task_model_mapping = self._default_task_mapping()
        self.total_cost = 0.0
        self.usage_stats = {}
        
        # Load custom configuration if provided
        if config_path and config_path.exists():
            self.load_config(config_path)
        else:
            # Try to load from default location
            default_config = Path.home() / ".howie" / "models.json"
            if default_config.exists():
                self.load_config(default_config)
    
    def _default_task_mapping(self) -> Dict[str, str]:
        """Default mapping of task types to models"""
        return {
            # Task type -> Model name
            "research": "perplexity-sonar",
            "deep_research": "perplexity-sonar-pro",
            "reasoning": "claude-sonnet-4",
            "comprehensive_research": "perplexity-sonar-deep-research",
            "analysis": "gpt-4o",
            "code_generation": "claude-sonnet-4",
            "simple_query": "gpt-4o-mini",
            "data_extraction": "gpt-4o-mini",
            "complex_reasoning": "claude-3-opus",
            "optimization": "gpt-4o",
            "classification": "claude-3-haiku",
            "summarization": "gpt-4o-mini",
            "default": "claude-sonnet-4"
        }
    
    def load_config(self, config_path: Path):
        """Load model configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load custom models
            if "models" in config:
                for model_name, model_config in config["models"].items():
                    self.models[model_name] = ModelConfig(**model_config)
            
            # Load task mappings
            if "task_mappings" in config:
                self.task_model_mapping.update(config["task_mappings"])
            
            # Load default model
            if "default_model" in config:
                self.current_model = config["default_model"]
                
        except Exception as e:
            print(f"Error loading model config: {e}")
    
    def save_config(self, config_path: Optional[Path] = None):
        """Save current configuration to JSON file"""
        if not config_path:
            config_path = Path.home() / ".howie" / "models.json"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "default_model": self.current_model,
            "task_mappings": self.task_model_mapping,
            "models": {
                name: {
                    "provider": model.provider,
                    "model_name": model.model_name,
                    "tier": model.tier,
                    "api_key_env": model.api_key_env,
                    "max_tokens": model.max_tokens,
                    "temperature": model.temperature,
                    "supports_tools": model.supports_tools,
                    "supports_vision": model.supports_vision,
                    "cost_per_1k_input": model.cost_per_1k_input,
                    "cost_per_1k_output": model.cost_per_1k_output,
                    "best_for": model.best_for
                }
                for name, model in self.models.items()
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def get_client(self, model_name: Optional[str] = None):
        """Get or create client for specified model"""
        if not model_name:
            model_name = self.current_model
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.models[model_name]
        
        # Check if we already have a client
        if model_name in self.clients:
            return self.clients[model_name], model_config
        
        # Create new client based on provider
        if model_config.provider == ModelProvider.OPENAI:
            api_key = os.getenv(model_config.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found: {model_config.api_key_env}")
            client = AsyncOpenAI(api_key=api_key)
            
        elif model_config.provider == ModelProvider.ANTHROPIC:
            api_key = os.getenv(model_config.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found: {model_config.api_key_env}")
            client = anthropic.AsyncAnthropic(api_key=api_key)
            
        elif model_config.provider == ModelProvider.PERPLEXITY:
            api_key = os.getenv(model_config.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found: {model_config.api_key_env}")
            # Perplexity uses OpenAI-compatible API
            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
            
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        
        self.clients[model_name] = client
        return client, model_config
    
    async def complete(self, 
                      messages: List[Dict],
                      model: Optional[str] = None,
                      task_type: Optional[str] = None,
                      **kwargs) -> str:
        """Get completion from specified model"""
        
        # Determine which model to use
        if not model:
            if task_type and task_type in self.task_model_mapping:
                model = self.task_model_mapping[task_type]
            else:
                model = self.current_model
        
        client, model_config = await self.get_client(model)
        
        # Add system prompt for Perplexity models to maintain football context
        if model_config.provider == ModelProvider.PERPLEXITY:
            football_context = """You are Howie, an expert fantasy football AI assistant. Always maintain focus on NFL football and fantasy football context. If a query seems ambiguous, assume the user is asking about NFL football unless explicitly stated otherwise.

For example:
- "Who benefits most from the most recent cuts" = NFL roster cuts and fantasy football implications
- "What are the latest updates" = NFL news and fantasy football updates
- "Who is the best option" = Best fantasy football option for the context
- "Recent cuts" = NFL roster cuts and fantasy football implications

When searching for information, prioritize NFL football news, player updates, team changes, and fantasy football implications."""
            
            # Insert system prompt at the beginning
            enhanced_messages = [{"role": "system", "content": football_context}] + messages
        else:
            enhanced_messages = messages
        
        # Track usage
        if model not in self.usage_stats:
            self.usage_stats[model] = {"calls": 0, "tokens": 0}
        self.usage_stats[model]["calls"] += 1
        
        try:
            if model_config.provider == ModelProvider.ANTHROPIC:
                # Anthropic API format - separate system messages from user/assistant messages
                system_message = None
                user_messages = []
                
                for msg in enhanced_messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        user_messages.append(msg)
                
                # Create the API call with proper Anthropic format
                api_params = {
                    "model": model_config.model_name,
                    "messages": user_messages,
                    "max_tokens": kwargs.get("max_tokens", model_config.max_tokens),
                    "temperature": kwargs.get("temperature", model_config.temperature)
                }
                
                # Add system message if present
                if system_message:
                    api_params["system"] = system_message
                
                response = await client.messages.create(**api_params)
                content = response.content[0].text
                
                # Track tokens and cost
                if hasattr(response, 'usage'):
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                    self._track_usage(model, input_tokens, output_tokens)
                
            else:
                # OpenAI-compatible API (OpenAI, Perplexity)
                response = await client.chat.completions.create(
                    model=model_config.model_name,
                    messages=enhanced_messages,
                    max_tokens=kwargs.get("max_tokens", model_config.max_tokens),
                    temperature=kwargs.get("temperature", model_config.temperature),
                    **{k: v for k, v in kwargs.items() if k not in ["max_tokens", "temperature"]}
                )
                content = response.choices[0].message.content
                
                # Track tokens and cost
                if hasattr(response, 'usage'):
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    self._track_usage(model, input_tokens, output_tokens)
            
            return content
            
        except Exception as e:
            # Fallback to gpt-4o if specific model fails
            if model != "gpt-4o":
                print(f"Model {model} failed with error: {str(e)[:200]}, falling back to gpt-4o")
                return await self.complete(messages, model="gpt-4o", **kwargs)
            raise e
    
    def _track_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Track token usage and cost"""
        if model in self.models:
            model_config = self.models[model]
            
            # Calculate cost
            input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input
            output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output
            total_cost = input_cost + output_cost
            
            self.total_cost += total_cost
            
            if model not in self.usage_stats:
                self.usage_stats[model] = {"calls": 0, "tokens": 0, "cost": 0}
            
            self.usage_stats[model]["tokens"] += input_tokens + output_tokens
            self.usage_stats[model]["cost"] = self.usage_stats[model].get("cost", 0) + total_cost
    
    def set_model(self, model_name: str):
        """Set the current default model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        self.current_model = model_name
    
    def set_task_model(self, task_type: str, model_name: str):
        """Set model for specific task type"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        self.task_model_mapping[task_type] = model_name
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get the best model for a specific task type"""
        return self.task_model_mapping.get(task_type, self.current_model)
    
    def list_models(self) -> Dict[str, ModelConfig]:
        """List all available models"""
        return self.models
    
    def get_usage_report(self) -> Dict:
        """Get usage statistics and cost report"""
        return {
            "total_cost": round(self.total_cost, 4),
            "by_model": self.usage_stats,
            "current_model": self.current_model,
            "task_mappings": self.task_model_mapping
        }
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a specific model and token count"""
        if model not in self.models:
            return 0.0
        
        model_config = self.models[model]
        input_cost = (input_tokens / 1000) * model_config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_config.cost_per_1k_output
        
        return round(input_cost + output_cost, 4)
    
    def recommend_model(self, task_description: str) -> str:
        """Recommend the best model for a task based on description"""
        task_lower = task_description.lower()
        
        # Check for specific keywords
        if any(word in task_lower for word in ["research", "search", "find information", "current"]):
            return "perplexity-sonar" if "deep" not in task_lower else "perplexity-sonar-pro"
        
        elif any(word in task_lower for word in ["code", "script", "function", "program"]):
            return "claude-sonnet-4"
        
        elif any(word in task_lower for word in ["complex", "detailed", "comprehensive"]):
            return "claude-3-opus" if "analysis" in task_lower else "gpt-4o"
        
        elif any(word in task_lower for word in ["reasoning", "logic", "think", "analyze"]):
            return "claude-sonnet-4"
        
        elif any(word in task_lower for word in ["simple", "quick", "basic", "list"]):
            return "gpt-4o-mini"
        
        elif any(word in task_lower for word in ["classify", "categorize", "extract"]):
            return "claude-3-haiku"
        
        else:
            return self.current_model