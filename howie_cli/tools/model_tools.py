"""
Tools for managing and switching between AI models
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.model_manager import ModelManager


class SwitchModelTool(BaseTool):
    """Switch to a different AI model"""
    
    def __init__(self):
        super().__init__()
        self.name = "switch_model"
        self.category = "models"
        self.description = "Switch to a different AI model"
        self.parameters = [
            ToolParameter(
                name="model_name",
                type="string",
                description="Name of the model to switch to",
                required=True,
                choices=[
                    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
                    "claude-3-opus", "claude-3-5-sonnet", "claude-3-haiku",
                    "perplexity-sonar", "perplexity-sonar-pro"
                ]
            )
        ]
        self.model_manager = None
    
    def _get_manager(self) -> ModelManager:
        if not self.model_manager:
            self.model_manager = ModelManager()
        return self.model_manager
    
    async def execute(self, model_name: str, **kwargs) -> ToolResult:
        """Switch to specified model"""
        try:
            manager = self._get_manager()
            
            # Check if model is available
            if model_name not in manager.models:
                available = list(manager.models.keys())
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Unknown model: {model_name}. Available: {available}"
                )
            
            # Check if API key is configured
            model_config = manager.models[model_name]
            import os
            if not os.getenv(model_config.api_key_env):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"API key not configured: {model_config.api_key_env}"
                )
            
            # Switch model
            previous_model = manager.current_model
            manager.set_model(model_name)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "previous_model": previous_model,
                    "current_model": model_name,
                    "provider": model_config.provider.value,
                    "tier": model_config.tier.value
                },
                metadata={
                    "supports_tools": model_config.supports_tools,
                    "supports_vision": model_config.supports_vision,
                    "best_for": model_config.best_for
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to switch model: {str(e)}"
            )


class ConfigureModelTool(BaseTool):
    """Configure model settings and task mappings"""
    
    def __init__(self):
        super().__init__()
        self.name = "configure_model"
        self.category = "models"
        self.description = "Configure model settings and task mappings"
        self.parameters = [
            ToolParameter(
                name="task_type",
                type="string",
                description="Type of task to configure",
                required=True,
                choices=[
                    "research", "analysis", "code_generation",
                    "optimization", "simple_query", "classification"
                ]
            ),
            ToolParameter(
                name="model_name",
                type="string",
                description="Model to use for this task type",
                required=True
            )
        ]
        self.model_manager = None
    
    def _get_manager(self) -> ModelManager:
        if not self.model_manager:
            self.model_manager = ModelManager()
        return self.model_manager
    
    async def execute(self, task_type: str, model_name: str, **kwargs) -> ToolResult:
        """Configure model for task type"""
        try:
            manager = self._get_manager()
            
            # Validate model
            if model_name not in manager.models:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Unknown model: {model_name}"
                )
            
            # Set task mapping
            manager.set_task_model(task_type, model_name)
            
            # Save configuration
            manager.save_config()
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "task_type": task_type,
                    "model": model_name,
                    "saved": True
                },
                metadata={
                    "all_mappings": manager.task_model_mapping
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to configure model: {str(e)}"
            )


class ModelInfoTool(BaseTool):
    """Get information about available models and usage"""
    
    def __init__(self):
        super().__init__()
        self.name = "model_info"
        self.category = "models"
        self.description = "Get information about available models and usage"
        self.parameters = [
            ToolParameter(
                name="info_type",
                type="string",
                description="Type of information to retrieve",
                required=False,
                default="summary",
                choices=["summary", "models", "usage", "costs", "mappings"]
            )
        ]
        self.model_manager = None
    
    def _get_manager(self) -> ModelManager:
        if not self.model_manager:
            self.model_manager = ModelManager()
        return self.model_manager
    
    async def execute(self, info_type: str = "summary", **kwargs) -> ToolResult:
        """Get model information"""
        try:
            manager = self._get_manager()
            
            if info_type == "models":
                # List all models with details
                models_info = {}
                for name, config in manager.models.items():
                    models_info[name] = {
                        "provider": config.provider.value,
                        "tier": config.tier.value,
                        "supports_tools": config.supports_tools,
                        "supports_vision": config.supports_vision,
                        "cost_input_1k": config.cost_per_1k_input,
                        "cost_output_1k": config.cost_per_1k_output,
                        "best_for": config.best_for
                    }
                data = models_info
                
            elif info_type == "usage":
                # Get usage statistics
                data = manager.get_usage_report()
                
            elif info_type == "costs":
                # Calculate costs
                usage = manager.get_usage_report()
                cost_breakdown = {}
                for model, stats in usage.get("by_model", {}).items():
                    if "cost" in stats:
                        cost_breakdown[model] = {
                            "cost": stats["cost"],
                            "calls": stats["calls"],
                            "tokens": stats["tokens"]
                        }
                data = {
                    "total_cost": usage.get("total_cost", 0),
                    "by_model": cost_breakdown
                }
                
            elif info_type == "mappings":
                # Get task mappings
                data = manager.task_model_mapping
                
            else:  # summary
                data = {
                    "current_model": manager.current_model,
                    "available_models": list(manager.models.keys()),
                    "total_cost": manager.total_cost,
                    "task_mappings": manager.task_model_mapping
                }
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=data,
                metadata={"info_type": info_type}
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to get model info: {str(e)}"
            )


class EstimateCostTool(BaseTool):
    """Estimate cost for a specific model and token count"""
    
    def __init__(self):
        super().__init__()
        self.name = "estimate_cost"
        self.category = "models"
        self.description = "Estimate cost for model usage"
        self.parameters = [
            ToolParameter(
                name="model_name",
                type="string",
                description="Model to estimate cost for",
                required=True
            ),
            ToolParameter(
                name="input_tokens",
                type="int",
                description="Number of input tokens",
                required=True
            ),
            ToolParameter(
                name="output_tokens",
                type="int",
                description="Number of output tokens",
                required=True
            )
        ]
        self.model_manager = None
    
    def _get_manager(self) -> ModelManager:
        if not self.model_manager:
            self.model_manager = ModelManager()
        return self.model_manager
    
    async def execute(self, model_name: str, input_tokens: int, 
                     output_tokens: int, **kwargs) -> ToolResult:
        """Estimate cost"""
        try:
            manager = self._get_manager()
            
            if model_name not in manager.models:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Unknown model: {model_name}"
                )
            
            cost = manager.estimate_cost(model_name, input_tokens, output_tokens)
            model_config = manager.models[model_name]
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "model": model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost": cost,
                    "cost_breakdown": {
                        "input_cost": (input_tokens / 1000) * model_config.cost_per_1k_input,
                        "output_cost": (output_tokens / 1000) * model_config.cost_per_1k_output
                    }
                },
                metadata={
                    "provider": model_config.provider.value,
                    "rates": {
                        "input_per_1k": model_config.cost_per_1k_input,
                        "output_per_1k": model_config.cost_per_1k_output
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to estimate cost: {str(e)}"
            )