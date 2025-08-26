"""
Sub-agent system for autonomous task execution
Similar to Claude's agent spawning capability
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import json
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from enum import Enum

from .context import ConversationContext
from .workspace import WorkspaceManager
from ..tools.registry import ToolRegistry


class AgentStatus(str, Enum):
    """Status of agent execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentResult(BaseModel):
    """Result from agent execution"""
    status: AgentStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    steps_completed: List[str] = Field(default_factory=list)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SubAgent(ABC):
    """Base class for all sub-agents"""
    
    def __init__(self, name: str, tools: ToolRegistry, 
                 model_manager=None, preferred_model: Optional[str] = None):
        self.name = name
        self.tools = tools
        self.context = ConversationContext()
        self.workspace = WorkspaceManager()
        self.status = AgentStatus.PENDING
        self.steps_completed = []
        
        # Model management
        if model_manager:
            self.model_manager = model_manager
        else:
            # Create default model manager if not provided
            from .model_manager import ModelManager
            self.model_manager = ModelManager()
        
        self.preferred_model = preferred_model
        
    @abstractmethod
    async def execute(self, task: str, **kwargs) -> AgentResult:
        """Execute the agent's task"""
        pass
    
    async def think(self, prompt: str, task_type: Optional[str] = None) -> str:
        """Use AI to think through a problem with appropriate model"""
        messages = [
            {"role": "system", "content": f"You are {self.name}, a specialized sub-agent."},
            {"role": "user", "content": prompt}
        ]
        
        # Use preferred model if set, otherwise let model manager decide
        model = self.preferred_model
        
        response = await self.model_manager.complete(
            messages=messages,
            model=model,
            task_type=task_type,
            temperature=0.7
        )
        return response
    
    async def use_tool(self, tool_name: str, **params) -> Any:
        """Execute a tool and return result"""
        result = await self.tools.execute(tool_name, **params)
        self.steps_completed.append(f"Used {tool_name}: {result.status}")
        return result
    
    def report_progress(self, step: str):
        """Report progress of execution"""
        self.steps_completed.append(f"[{datetime.now().strftime('%H:%M:%S')}] {step}")


class ResearchAgent(SubAgent):
    """Agent for researching and gathering information"""
    
    def __init__(self, tools: ToolRegistry, model_manager=None):
        # Research agents should use Perplexity for research tasks
        super().__init__("ResearchAgent", tools, model_manager, preferred_model="perplexity-sonar")
        
    async def execute(self, task: str, max_depth: int = 3, **kwargs) -> AgentResult:
        """Research a topic thoroughly"""
        try:
            self.status = AgentStatus.RUNNING
            start_time = datetime.now()
            
            # Plan research approach using research model
            self.report_progress("Planning research approach")
            plan = await self.think(f"""
            Task: {task}
            
            Create a research plan with specific steps to gather information.
            Focus on fantasy football data and analysis.
            """, task_type="research")
            
            # Execute research steps
            research_results = []
            
            # Step 1: Database research
            self.report_progress("Searching database for relevant data")
            db_result = await self.use_tool(
                "database_query",
                query=f"Data related to: {task}"
            )
            if db_result.status.value == "success":
                research_results.append(("Database findings", db_result.data))
            
            # Step 2: Statistical analysis
            self.report_progress("Analyzing statistics")
            stats_prompt = f"Analyze this data for: {task}\nData: {db_result.data if db_result else 'No data'}"
            analysis = await self.think(stats_prompt)
            research_results.append(("Statistical analysis", analysis))
            
            # Step 3: Generate insights
            self.report_progress("Generating insights")
            insights = await self.think(f"""
            Based on the research about {task}:
            {research_results}
            
            Provide key insights and recommendations.
            """)
            
            # Compile final report
            final_report = {
                "task": task,
                "research_plan": plan,
                "findings": research_results,
                "insights": insights,
                "steps": self.steps_completed
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                status=AgentStatus.COMPLETED,
                result=final_report,
                steps_completed=self.steps_completed,
                execution_time=execution_time
            )
            
        except Exception as e:
            return AgentResult(
                status=AgentStatus.FAILED,
                error=str(e),
                steps_completed=self.steps_completed
            )


class AnalysisAgent(SubAgent):
    """Agent for deep analysis of players/teams"""
    
    def __init__(self, tools: ToolRegistry, model_manager=None):
        # Analysis agents should use GPT-4o for complex analysis
        super().__init__("AnalysisAgent", tools, model_manager, preferred_model="gpt-4o")
    
    async def execute(self, task: str, entities: List[str] = None, **kwargs) -> AgentResult:
        """Perform deep analysis"""
        try:
            self.status = AgentStatus.RUNNING
            start_time = datetime.now()
            
            self.report_progress(f"Starting analysis: {task}")
            
            # Identify entities to analyze
            if not entities:
                self.report_progress("Identifying entities to analyze")
                entities_prompt = f"Extract player/team names from: {task}"
                entities_response = await self.think(entities_prompt)
                entities = json.loads(entities_response) if entities_response else []
            
            analysis_results = {}
            
            for entity in entities:
                self.report_progress(f"Analyzing {entity}")
                
                # Get stats
                stats_result = await self.use_tool(
                    "player_stats",
                    player_name=entity
                )
                
                # Get trends
                trends_result = await self.use_tool(
                    "historical_trends",
                    entity=entity
                )
                
                # Generate analysis
                entity_analysis = await self.think(f"""
                Analyze {entity} with this data:
                Stats: {stats_result.data if stats_result else 'No data'}
                Trends: {trends_result.data if trends_result else 'No data'}
                
                Provide comprehensive analysis.
                """)
                
                analysis_results[entity] = {
                    "stats": stats_result.data if stats_result else None,
                    "trends": trends_result.data if trends_result else None,
                    "analysis": entity_analysis
                }
            
            # Generate comparative analysis if multiple entities
            if len(entities) > 1:
                self.report_progress("Generating comparative analysis")
                comparison = await self.think(f"""
                Compare these entities:
                {json.dumps(analysis_results, indent=2)}
                
                Provide comparative insights.
                """)
                analysis_results["comparison"] = comparison
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                status=AgentStatus.COMPLETED,
                result=analysis_results,
                steps_completed=self.steps_completed,
                execution_time=execution_time
            )
            
        except Exception as e:
            return AgentResult(
                status=AgentStatus.FAILED,
                error=str(e),
                steps_completed=self.steps_completed
            )


class CodeGenerationAgent(SubAgent):
    """Agent for generating analysis code"""
    
    def __init__(self, tools: ToolRegistry, model_manager=None):
        # Code generation agents should use Claude Sonnet for code
        super().__init__("CodeGenerationAgent", tools, model_manager, preferred_model="claude-3-5-sonnet")
    
    async def execute(self, task: str, language: str = "python", **kwargs) -> AgentResult:
        """Generate code for analysis"""
        try:
            self.status = AgentStatus.RUNNING
            start_time = datetime.now()
            
            self.report_progress("Understanding requirements")
            
            # Analyze the database schema
            self.report_progress("Analyzing database schema")
            db_info = await self.use_tool("database_info", info_type="schema")
            
            # Generate code
            self.report_progress(f"Generating {language} code")
            code_result = await self.use_tool(
                "generate_analysis_script",
                requirements=task
            )
            
            # Test the code (conceptually)
            self.report_progress("Validating generated code")
            validation = await self.think(f"""
            Validate this code for the task: {task}
            Code: {code_result.data if code_result else 'No code'}
            
            Check for:
            1. Syntax correctness
            2. Logic errors
            3. Completeness
            """)
            
            # Generate documentation
            self.report_progress("Generating documentation")
            docs = await self.think(f"""
            Create documentation for this code:
            {code_result.data if code_result else 'No code'}
            
            Include: purpose, usage, examples
            """)
            
            result = {
                "code": code_result.data if code_result else None,
                "validation": validation,
                "documentation": docs,
                "language": language
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                status=AgentStatus.COMPLETED,
                result=result,
                steps_completed=self.steps_completed,
                execution_time=execution_time
            )
            
        except Exception as e:
            return AgentResult(
                status=AgentStatus.FAILED,
                error=str(e),
                steps_completed=self.steps_completed
            )


class OptimizationAgent(SubAgent):
    """Agent for lineup optimization and strategy"""
    
    def __init__(self, tools: ToolRegistry, model_manager=None):
        # Optimization agents should use GPT-4o for complex reasoning
        super().__init__("OptimizationAgent", tools, model_manager, preferred_model="gpt-4o")
    
    async def execute(self, task: str, constraints: Dict = None, **kwargs) -> AgentResult:
        """Optimize lineup or strategy"""
        try:
            self.status = AgentStatus.RUNNING
            start_time = datetime.now()
            
            self.report_progress("Analyzing optimization requirements")
            
            # Parse requirements
            requirements = await self.think(f"""
            Parse these optimization requirements: {task}
            Extract: players, constraints, objectives
            Return as JSON.
            """)
            
            # Get player pool
            self.report_progress("Gathering player data")
            # Implementation would fetch relevant players
            
            # Run optimization
            self.report_progress("Running optimization algorithm")
            optimization_result = await self.use_tool(
                "lineup_optimizer",
                available_players=kwargs.get("players", []),
                constraints=constraints
            )
            
            # Generate alternative lineups
            self.report_progress("Generating alternative options")
            alternatives = await self.think(f"""
            Based on this optimal lineup: {optimization_result.data if optimization_result else 'No data'}
            Generate 2-3 alternative lineups with different risk profiles.
            """)
            
            # Risk analysis
            self.report_progress("Performing risk analysis")
            risk_analysis = await self.think(f"""
            Analyze risks for this lineup:
            {optimization_result.data if optimization_result else 'No data'}
            
            Consider: injuries, matchups, weather, variance
            """)
            
            result = {
                "optimal_lineup": optimization_result.data if optimization_result else None,
                "alternatives": alternatives,
                "risk_analysis": risk_analysis,
                "requirements": requirements
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                status=AgentStatus.COMPLETED,
                result=result,
                steps_completed=self.steps_completed,
                execution_time=execution_time
            )
            
        except Exception as e:
            return AgentResult(
                status=AgentStatus.FAILED,
                error=str(e),
                steps_completed=self.steps_completed
            )


class WorkflowAgent(SubAgent):
    """Agent that can spawn and coordinate other agents"""
    
    def __init__(self, tools: ToolRegistry, model_manager=None):
        super().__init__("WorkflowAgent", tools, model_manager, preferred_model="gpt-4o")
        self.sub_agents = {
            "research": ResearchAgent(self.tools, self.model_manager),
            "analysis": AnalysisAgent(self.tools, self.model_manager),
            "code": CodeGenerationAgent(self.tools, self.model_manager),
            "optimization": OptimizationAgent(self.tools, self.model_manager)
        }
    
    async def execute(self, task: str, workflow_type: str = "auto", **kwargs) -> AgentResult:
        """Execute a complex workflow using multiple agents"""
        try:
            self.status = AgentStatus.RUNNING
            start_time = datetime.now()
            
            self.report_progress(f"Starting workflow: {task}")
            
            # Plan workflow
            self.report_progress("Planning workflow steps")
            workflow_plan = await self.think(f"""
            Task: {task}
            
            Create a workflow plan using available agents:
            - research: For gathering information
            - analysis: For deep analysis
            - code: For generating scripts
            - optimization: For lineup/strategy optimization
            
            Return as JSON list of steps with agent and subtask.
            """)
            
            try:
                steps = json.loads(workflow_plan)
            except:
                # Fallback to simple workflow
                steps = [
                    {"agent": "research", "subtask": task},
                    {"agent": "analysis", "subtask": f"Analyze findings for {task}"}
                ]
            
            # Execute workflow steps
            workflow_results = []
            
            for i, step in enumerate(steps):
                agent_name = step.get("agent", "research")
                subtask = step.get("subtask", task)
                
                self.report_progress(f"Step {i+1}: {agent_name} - {subtask}")
                
                if agent_name in self.sub_agents:
                    agent = self.sub_agents[agent_name]
                    result = await agent.execute(subtask, **kwargs)
                    workflow_results.append({
                        "step": i + 1,
                        "agent": agent_name,
                        "subtask": subtask,
                        "result": result.result,
                        "status": result.status
                    })
                    
                    # Pass results to next step if needed
                    if result.result:
                        kwargs["previous_result"] = result.result
            
            # Synthesize results
            self.report_progress("Synthesizing workflow results")
            synthesis = await self.think(f"""
            Synthesize these workflow results for task: {task}
            
            Results: {json.dumps(workflow_results, indent=2, default=str)}
            
            Provide comprehensive summary and recommendations.
            """)
            
            final_result = {
                "task": task,
                "workflow_plan": steps,
                "step_results": workflow_results,
                "synthesis": synthesis
            }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                status=AgentStatus.COMPLETED,
                result=final_result,
                steps_completed=self.steps_completed,
                execution_time=execution_time
            )
            
        except Exception as e:
            return AgentResult(
                status=AgentStatus.FAILED,
                error=str(e),
                steps_completed=self.steps_completed
            )


class AgentManager:
    """Manages and coordinates multiple agents"""
    
    def __init__(self, tools: ToolRegistry, model_manager=None):
        self.tools = tools
        self.model_manager = model_manager
        if not self.model_manager:
            from .model_manager import ModelManager
            self.model_manager = ModelManager()
        self.agents = {}
        self.running_agents = {}
        
    def create_agent(self, agent_type: str, name: Optional[str] = None, 
                    preferred_model: Optional[str] = None) -> SubAgent:
        """Create a new agent instance with optional model preference"""
        agent_classes = {
            "research": ResearchAgent,
            "analysis": AnalysisAgent,
            "code": CodeGenerationAgent,
            "optimization": OptimizationAgent,
            "workflow": WorkflowAgent
        }
        
        agent_class = agent_classes.get(agent_type, ResearchAgent)
        agent = agent_class(self.tools, self.model_manager)
        
        # Override model if specified
        if preferred_model:
            agent.preferred_model = preferred_model
        
        if name:
            agent.name = name
        
        agent_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.agents[agent_id] = agent
        
        return agent
    
    async def spawn_agent(self, agent_type: str, task: str, **kwargs) -> str:
        """Spawn an agent to work on a task asynchronously"""
        agent = self.create_agent(agent_type)
        agent_id = f"{agent_type}_{id(agent)}"
        
        # Run agent in background
        self.running_agents[agent_id] = asyncio.create_task(
            agent.execute(task, **kwargs)
        )
        
        return agent_id
    
    async def get_agent_result(self, agent_id: str) -> Optional[AgentResult]:
        """Get result from a spawned agent"""
        if agent_id in self.running_agents:
            task = self.running_agents[agent_id]
            if task.done():
                result = await task
                del self.running_agents[agent_id]
                return result
            else:
                return AgentResult(
                    status=AgentStatus.RUNNING,
                    result={"message": "Agent still working..."}
                )
        return None
    
    async def spawn_parallel_agents(self, tasks: List[Dict]) -> List[AgentResult]:
        """Spawn multiple agents to work in parallel"""
        agent_tasks = []
        
        for task_config in tasks:
            agent_type = task_config.get("type", "research")
            task = task_config.get("task", "")
            kwargs = task_config.get("kwargs", {})
            
            agent = self.create_agent(agent_type)
            agent_tasks.append(agent.execute(task, **kwargs))
        
        # Run all agents in parallel
        results = await asyncio.gather(*agent_tasks)
        return results