"""
Base tool class for all Howie tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from datetime import datetime


class ToolStatus(str, Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class ToolResult(BaseModel):
    """Standard result from tool execution"""
    status: ToolStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class ToolParameter(BaseModel):
    """Parameter definition for tools"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None


class BaseTool(ABC):
    """Base class for all Howie tools"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = self.__doc__ or "No description available"
        self.parameters: List[ToolParameter] = []
        self.requires_confirmation = False
        self.category = "general"
        
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    async def validate_params(self, **kwargs) -> bool:
        """Validate parameters before execution"""
        required_params = [p.name for p in self.parameters if p.required]
        for param in required_params:
            if param not in kwargs:
                return False
        return True
    
    async def run(self, **kwargs) -> ToolResult:
        """Run the tool with validation and error handling"""
        start_time = datetime.now()
        
        try:
            # Validate parameters
            if not await self.validate_params(**kwargs):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Missing required parameters for {self.name}"
                )
            
            # Execute the tool
            result = await self.execute(**kwargs)
            
            # Add execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Tool execution failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def get_schema(self) -> Dict:
        """Get JSON schema for the tool"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [p.dict() for p in self.parameters],
            "requires_confirmation": self.requires_confirmation
        }


class CompositeTool(BaseTool):
    """Tool that chains multiple tools together"""
    
    def __init__(self, tools: List[BaseTool]):
        super().__init__()
        self.tools = tools
        self.name = "CompositeTool"
        self.description = "Chains multiple tools together"
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute tools in sequence, passing results forward"""
        results = []
        current_data = kwargs
        
        for tool in self.tools:
            result = await tool.run(**current_data)
            results.append(result)
            
            if result.status == ToolStatus.ERROR:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Chain failed at {tool.name}: {result.error}",
                    metadata={"results": results}
                )
            
            # Pass result data forward
            if result.data:
                if isinstance(result.data, dict):
                    current_data.update(result.data)
                else:
                    current_data["previous_result"] = result.data
        
        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=results[-1].data if results else None,
            metadata={"chain_results": results}
        )