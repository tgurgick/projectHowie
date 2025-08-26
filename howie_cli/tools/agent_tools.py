"""
Tools for spawning and managing sub-agents
Similar to Claude's Task tool
"""

from typing import Dict, List, Optional, Any
import asyncio
import json

from ..core.base_tool import BaseTool, ToolResult, ToolStatus, ToolParameter
from ..core.subagent import AgentManager, AgentStatus


class SpawnAgentTool(BaseTool):
    """Spawn an autonomous agent to handle complex tasks"""
    
    def __init__(self):
        super().__init__()
        self.name = "spawn_agent"
        self.category = "agents"
        self.description = "Spawn an autonomous agent to handle complex, multi-step tasks"
        self.parameters = [
            ToolParameter(
                name="task",
                type="string",
                description="The task for the agent to complete",
                required=True
            ),
            ToolParameter(
                name="agent_type",
                type="string",
                description="Type of agent to spawn",
                required=False,
                default="research",
                choices=["research", "analysis", "code", "optimization", "workflow"]
            ),
            ToolParameter(
                name="wait_for_result",
                type="bool",
                description="Wait for agent to complete",
                required=False,
                default=True
            ),
            ToolParameter(
                name="context",
                type="dict",
                description="Additional context for the agent",
                required=False
            )
        ]
        self.agent_manager = None
    
    def _get_manager(self) -> AgentManager:
        """Get or create agent manager"""
        if not self.agent_manager:
            # Import here to avoid circular dependency
            import os
            from openai import AsyncOpenAI
            from ..tools.registry import global_registry
            
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.agent_manager = AgentManager(global_registry, client)
        return self.agent_manager
    
    async def execute(self, task: str, agent_type: str = "research",
                     wait_for_result: bool = True, context: Dict = None, **kwargs) -> ToolResult:
        """Spawn and execute an agent"""
        try:
            manager = self._get_manager()
            
            # Determine best agent type if not specified
            if agent_type == "auto":
                agent_type = self._determine_agent_type(task)
            
            # Create and configure agent
            agent = manager.create_agent(agent_type)
            
            # Execute agent task
            if wait_for_result:
                # Synchronous execution - wait for result
                result = await agent.execute(task, **(context or {}))
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data=result.result,
                    metadata={
                        "agent_type": agent_type,
                        "execution_time": result.execution_time,
                        "steps_completed": result.steps_completed,
                        "agent_status": result.status
                    }
                )
            else:
                # Asynchronous execution - return agent ID
                agent_id = await manager.spawn_agent(agent_type, task, **(context or {}))
                
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={"agent_id": agent_id, "status": "running"},
                    metadata={
                        "agent_type": agent_type,
                        "task": task,
                        "async": True
                    }
                )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to spawn agent: {str(e)}"
            )
    
    def _determine_agent_type(self, task: str) -> str:
        """Determine the best agent type for a task"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["research", "find", "search", "investigate"]):
            return "research"
        elif any(word in task_lower for word in ["analyze", "compare", "evaluate"]):
            return "analysis"
        elif any(word in task_lower for word in ["code", "script", "generate", "create"]):
            return "code"
        elif any(word in task_lower for word in ["optimize", "lineup", "best"]):
            return "optimization"
        elif any(word in task_lower for word in ["workflow", "multiple", "complex"]):
            return "workflow"
        else:
            return "research"


class ParallelAgentsTool(BaseTool):
    """Spawn multiple agents to work in parallel"""
    
    def __init__(self):
        super().__init__()
        self.name = "parallel_agents"
        self.category = "agents"
        self.description = "Spawn multiple agents to work on tasks in parallel"
        self.parameters = [
            ToolParameter(
                name="tasks",
                type="list",
                description="List of tasks for agents",
                required=True
            ),
            ToolParameter(
                name="agent_types",
                type="list",
                description="Types of agents for each task",
                required=False
            )
        ]
        self.agent_manager = None
    
    def _get_manager(self) -> AgentManager:
        """Get or create agent manager"""
        if not self.agent_manager:
            import os
            from openai import AsyncOpenAI
            from ..tools.registry import global_registry
            
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.agent_manager = AgentManager(global_registry, client)
        return self.agent_manager
    
    async def execute(self, tasks: List[Any], agent_types: Optional[List[str]] = None, **kwargs) -> ToolResult:
        """Execute multiple agents in parallel"""
        try:
            manager = self._get_manager()
            
            # Prepare task configurations
            task_configs = []
            for i, task in enumerate(tasks):
                if isinstance(task, dict):
                    task_configs.append(task)
                else:
                    agent_type = agent_types[i] if agent_types and i < len(agent_types) else "research"
                    task_configs.append({
                        "type": agent_type,
                        "task": str(task),
                        "kwargs": {}
                    })
            
            # Run agents in parallel
            results = await manager.spawn_parallel_agents(task_configs)
            
            # Compile results
            compiled_results = []
            for i, result in enumerate(results):
                compiled_results.append({
                    "task": task_configs[i]["task"],
                    "agent_type": task_configs[i]["type"],
                    "status": result.status,
                    "result": result.result if result.status == AgentStatus.COMPLETED else None,
                    "error": result.error if result.status == AgentStatus.FAILED else None,
                    "execution_time": result.execution_time
                })
            
            # Determine overall status
            if all(r.status == AgentStatus.COMPLETED for r in results):
                overall_status = ToolStatus.SUCCESS
            elif any(r.status == AgentStatus.FAILED for r in results):
                overall_status = ToolStatus.PARTIAL
            else:
                overall_status = ToolStatus.SUCCESS
            
            return ToolResult(
                status=overall_status,
                data=compiled_results,
                metadata={
                    "total_agents": len(tasks),
                    "completed": sum(1 for r in results if r.status == AgentStatus.COMPLETED),
                    "failed": sum(1 for r in results if r.status == AgentStatus.FAILED)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to run parallel agents: {str(e)}"
            )


class CheckAgentTool(BaseTool):
    """Check the status of a running agent"""
    
    def __init__(self):
        super().__init__()
        self.name = "check_agent"
        self.category = "agents"
        self.description = "Check the status of a running agent"
        self.parameters = [
            ToolParameter(
                name="agent_id",
                type="string",
                description="ID of the agent to check",
                required=True
            )
        ]
        self.agent_manager = None
    
    def _get_manager(self) -> AgentManager:
        """Get or create agent manager"""
        if not self.agent_manager:
            import os
            from openai import AsyncOpenAI
            from ..tools.registry import global_registry
            
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.agent_manager = AgentManager(global_registry, client)
        return self.agent_manager
    
    async def execute(self, agent_id: str, **kwargs) -> ToolResult:
        """Check agent status"""
        try:
            manager = self._get_manager()
            result = await manager.get_agent_result(agent_id)
            
            if result:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    data={
                        "agent_id": agent_id,
                        "status": result.status,
                        "result": result.result if result.status == AgentStatus.COMPLETED else None,
                        "steps_completed": result.steps_completed
                    },
                    metadata={
                        "execution_time": result.execution_time if hasattr(result, 'execution_time') else None
                    }
                )
            else:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Agent {agent_id} not found"
                )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Failed to check agent: {str(e)}"
            )


class WorkflowTool(BaseTool):
    """Execute a complex workflow using multiple coordinated agents"""
    
    def __init__(self):
        super().__init__()
        self.name = "workflow"
        self.category = "agents"
        self.description = "Execute a complex workflow using multiple coordinated agents"
        self.parameters = [
            ToolParameter(
                name="objective",
                type="string",
                description="The overall objective to achieve",
                required=True
            ),
            ToolParameter(
                name="steps",
                type="list",
                description="Specific workflow steps (optional)",
                required=False
            ),
            ToolParameter(
                name="constraints",
                type="dict",
                description="Constraints or requirements",
                required=False
            )
        ]
    
    async def execute(self, objective: str, steps: Optional[List] = None, 
                     constraints: Optional[Dict] = None, **kwargs) -> ToolResult:
        """Execute workflow"""
        try:
            # Use the workflow agent
            spawn_tool = SpawnAgentTool()
            
            workflow_context = {
                "objective": objective,
                "steps": steps,
                "constraints": constraints
            }
            
            result = await spawn_tool.execute(
                task=objective,
                agent_type="workflow",
                wait_for_result=True,
                context=workflow_context
            )
            
            return result
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Workflow failed: {str(e)}"
            )