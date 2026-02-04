"""
Flexcyz Orchestrator - The Brain
Manages multi-agent workflows, concurrency, competition, and self-improvement
"""

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from models.schemas import (
    AgentExecution, AgentType, CodeGenerationResult, FileState, PlanStep,
    Project, ProjectConfig, ProjectPlan, ProjectStatus, ReviewResult, Task,
    TaskStatus, WSMessageType, WebSocketMessage
)
from core.memory import ProjectMemory
from core.state_manager import state_manager


class AgentPool:
    """
    Manages agent lifecycle and concurrency
    Implements hot-swapping and resource limits
    """
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_agents: Dict[str, Any] = {}
        self.agent_status: Dict[str, dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    def register(self, name: str, agent_instance: Any):
        """Hot-swap capable registration"""
        self.active_agents[name] = agent_instance
        self.agent_status[name] = {
            "status": "idle",
            "current_task": None,
            "completed_tasks": 0,
            "failed_tasks": 0
        }
        print(f"🤖 Agent registered: {name}")
    
    def unregister(self, name: str):
        """Remove agent dynamically"""
        if name in self.active_agents:
            del self.active_agents[name]
            del self.agent_status[name]
    
    async def execute(self, agent_type: str, task: Task, context: dict) -> Any:
        """Execute with concurrency control and monitoring"""
        if agent_type not in self.active_agents:
            raise ValueError(f"Agent {agent_type} not registered")
        
        async with self.semaphore:
            agent = self.active_agents[agent_type]
            self.agent_status[agent_type]["status"] = "busy"
            self.agent_status[agent_type]["current_task"] = task.id
            
            try:
                # Run in thread pool if agent is sync
                if asyncio.iscoroutinefunction(agent.execute):
                    result = await agent.execute(task.description, context)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, 
                        agent.execute, 
                        task.description, 
                        context
                    )
                
                self.agent_status[agent_type]["completed_tasks"] += 1
                return result
                
            except Exception as e:
                self.agent_status[agent_type]["failed_tasks"] += 1
                raise e
            finally:
                self.agent_status[agent_type]["status"] = "idle"
                self.agent_status[agent_type]["current_task"] = None


class CompetitionManager:
    """
    Multi-agent competition system
    Multiple agents solve same task, best result wins
    """
    
    def __init__(self, pool: AgentPool):
        self.pool = pool
    
    async def run_competition(
        self, 
        task_description: str, 
        agent_types: List[str], 
        context: dict,
        evaluator: Callable[[List[Any]], int]
    ) -> Any:
        """
        Run multiple agents in parallel, return best result
        
        evaluator: function that takes results list and returns index of best
        """
        print(f"🏆 Starting competition with {len(agent_types)} agents")
        
        # Execute all agents concurrently
        tasks = []
        for agent_type in agent_types:
            # Create temporary task
            temp_task = Task(
                id=str(uuid4()),
                project_id=context.get("project_id", ""),
                description=task_description,
                agent_type=AgentType(agent_type)
            )
            t = self.pool.execute(agent_type, temp_task, context)
            tasks.append(t)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures
        valid_results = [(i, r) for i, r in enumerate(results) if not isinstance(r, Exception)]
        
        if not valid_results:
            raise Exception("All agents failed in competition")
        
        # Evaluate
        best_idx = evaluator([r for _, r in valid_results])
        actual_idx = valid_results[best_idx][0]
        winner = agent_types[actual_idx]
        
        print(f"✅ Competition winner: {winner}")
        
        return {
            "winner": winner,
            "result": valid_results[best_idx][1],
            "all_results": results,
            "participants": agent_types
        }


class SelfImprovementTracker:
    """
    Tracks agent performance and suggests improvements
    Learns from errors and user feedback
    """
    
    def __init__(self):
        self.performance_history: Dict[str, List[dict]] = {}
        self.error_patterns: Dict[str, int] = {}
        self.success_patterns: Dict[str, int] = {}
    
    def record_execution(
        self, 
        agent_type: str, 
        task: str, 
        success: bool, 
        duration: float,
        error_type: Optional[str] = None,
        user_feedback: Optional[str] = None
    ):
        """Record execution metrics"""
        if agent_type not in self.performance_history:
            self.performance_history[agent_type] = []
        
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "task": task[:100],
            "success": success,
            "duration": duration,
            "error_type": error_type,
            "feedback": user_feedback
        }
        
        self.performance_history[agent_type].append(record)
        
        # Pattern tracking
        if success:
            self.success_patterns[task[:50]] = self.success_patterns.get(task[:50], 0) + 1
        else:
            self.error_patterns[error_type or "unknown"] = self.error_patterns.get(error_type or "unknown", 0) + 1
    
    def get_agent_stats(self, agent_type: str) -> dict:
        """Get performance statistics for agent"""
        history = self.performance_history.get(agent_type, [])
        if not history:
            return {}
        
        total = len(history)
        successes = sum(1 for h in history if h["success"])
        
        return {
            "total_executions": total,
            "success_rate": successes / total,
            "avg_duration": sum(h["duration"] for h in history) / total,
            "common_errors": self._get_common_errors(agent_type),
            "improvement_suggestions": self._generate_suggestions(agent_type)
        }
    
    def _get_common_errors(self, agent_type: str) -> List[tuple]:
        """Get most common errors for agent"""
        agent_errors = [
            h["error_type"] for h in self.performance_history.get(agent_type, [])
            if h["error_type"]
        ]
        from collections import Counter
        return Counter(agent_errors).most_common(3)
    
    def _generate_suggestions(self, agent_type: str) -> List[str]:
        """Generate improvement suggestions based on patterns"""
        suggestions = []
        stats = self.get_agent_stats(agent_type)
        
        if stats.get("success_rate", 1.0) < 0.8:
            suggestions.append(f"Low success rate ({stats['success_rate']:.1%}). Consider refining prompts.")
        
        common_errors = self._get_common_errors(agent_type)
        if common_errors:
            suggestions.append(f"Common error: {common_errors[0][0]}. Add error handling.")
        
        return suggestions
    
    def get_best_practices(self, task_type: str) -> List[str]:
        """Extract best practices from successful executions"""
        # Find similar successful tasks
        practices = []
        for agent, history in self.performance_history.items():
            successful = [h for h in history if h["success"] and task_type in h["task"]]
            if len(successful) > 2:
                practices.append(f"Agent {agent} consistently succeeds at {task_type}")
        
        return practices


class Orchestrator:
    """
    Central brain of Flexcyz
    Coordinates all agents, manages workflow, handles real-time updates
    """
    
    def __init__(self):
        self.pool = AgentPool(max_concurrent=5)
        self.competition = CompetitionManager(self.pool)
        self.improvement = SelfImprovementTracker()
        self.websocket_clients: Dict[str, Any] = {}  # project_id -> websocket
        self.running_projects: Set[str] = set()
        self._shutdown_event = asyncio.Event()
        
        # Workflow definitions: which agents run in what order
        self.workflow_definitions = {
            "default": [
                AgentType.PLANNER,
                AgentType.ARCHITECT,
                AgentType.CODER,
                AgentType.REVIEWER,
                AgentType.CODER,  # Fix issues
                AgentType.DEPLOYER
            ],
            "research_heavy": [
                AgentType.PLANNER,
                AgentType.RESEARCHER,
                AgentType.ARCHITECT,
                AgentType.CODER,
                AgentType.REVIEWER
            ],
            "quick_fix": [
                AgentType.RESEARCHER,
                AgentType.CODER,
                AgentType.REVIEWER
            ]
        }
    
    def register_agent(self, name: str, agent: Any):
        """Register an agent (hot-swappable)"""
        self.pool.register(name, agent)
    
    def register_websocket(self, project_id: str, websocket: Any):
        """Register WebSocket for real-time updates"""
        self.websocket_clients[project_id] = websocket
    
    async def send_update(self, project_id: str, message_type: WSMessageType, payload: dict):
        """Send real-time update to client"""
        if project_id in self.websocket_clients:
            ws = self.websocket_clients[project_id]
            msg = WebSocketMessage(type=message_type, project_id=project_id, payload=payload)
            try:
                await ws.send_json(msg.model_dump())
            except:
                pass  # Client disconnected
    
    async def create_project(self, prompt: str, config: Optional[ProjectConfig] = None) -> Project:
        """Initialize new project from prompt"""
        # Auto-generate config if not provided
        if not config:
            config = ProjectConfig(
                name=f"Project_{str(uuid4())[:8]}",
                description=prompt,
                tech_stack=[]
            )
        
        project = state_manager.create_project(config)
        memory = ProjectMemory(project.id)
        
        # Store initial prompt
        memory.add_context("initial_prompt", prompt)
        memory.add_message("user", prompt)
        
        await self.send_update(
            project.id, 
            WSMessageType.PROGRESS,
            {"message": "Project initialized", "status": "initializing"}
        )
        
        return project
    
    async def run_project(self, project_id: str, prompt: str, mode: str = "default"):
        """
        Main entry: Run complete workflow from prompt to deployment
        
        mode: "default", "research_heavy", "quick_fix", "competition"
        """
        if project_id in self.running_projects:
            raise ValueError("Project already running")
        
        self.running_projects.add(project_id)
        project = state_manager.get_project(project_id)
        
        try:
            # Phase 1: Planning
            await self._phase_planning(project, prompt)
            
            # Phase 2: Execution (iterative)
            await self._phase_execution(project, mode)
            
            # Phase 3: Completion
            await self._phase_completion(project)
            
        except Exception as e:
            await self._handle_error(project, e)
        finally:
            self.running_projects.discard(project_id)
            if project_id in self.websocket_clients:
                del self.websocket_clients[project_id]
    
    async def _phase_planning(self, project: Project, prompt: str):
        """Phase 1: Expand prompt into detailed plan"""
        print(f"📋 Phase 1: Planning for project {project.id}")
        
        state_manager.update_project_status(project.id, ProjectStatus.PLANNING)
        
        memory = ProjectMemory(project.id)
        context = {
            "project_id": project.id,
            "prompt": prompt,
            "preferences": memory.get_all_preferences()
        }
        
        # Run planner agent
        planner_task = Task(
            project_id=project.id,
            description=f"Create detailed plan for: {prompt}",
            agent_type=AgentType.PLANNER,
            priority=5
        )
        
        await self.send_update(project.id, WSMessageType.PROGRESS, {
            "message": "Analyzing requirements and creating plan...",
            "agent": "planner",
            "status": "running"
        })
        
        start_time = datetime.utcnow()
        try:
            plan = await self.pool.execute("planner", planner_task, context)
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.improvement.record_execution("planner", prompt, True, duration)
            
            # Store plan
            if isinstance(plan, dict):
                plan_obj = ProjectPlan(**plan)
            else:
                plan_obj = plan
            
            memory.set("plan", "current", plan_obj.model_dump())
            
            # Create tasks from plan
            for step in plan_obj.steps:
                task = Task(
                    project_id=project.id,
                    description=step.description,
                    agent_type=step.agent_type,
                    dependencies=step.dependencies,
                    priority=step.priority,
                    metadata={"expected_files": step.expected_files}
                )
                state_manager.add_task(project.id, task)
            
            await self.send_update(project.id, WSMessageType.PROGRESS, {
                "message": f"Plan created: {len(plan_obj.steps)} tasks",
                "plan": plan_obj.summary,
                "status": "completed"
            })
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.improvement.record_execution(
                "planner", prompt, False, duration, 
                error_type=type(e).__name__
            )
            raise
    
    async def _phase_execution(self, project: Project, mode: str):
        """Phase 2: Execute all tasks with intelligent routing"""
        print(f"⚙️ Phase 2: Execution for project {project.id}")
        
        state_manager.update_project_status(project.id, ProjectStatus.CODING)
        memory = ProjectMemory(project.id)
        
        iteration = 0
        max_iterations = 50  # Safety limit
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get ready tasks
            ready_tasks = state_manager.get_ready_tasks(project.id)
            
            if not ready_tasks:
                # Check if all done
                all_tasks = state_manager.get_project_tasks(project.id)
                if all(t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for t in all_tasks):
                    break
                await asyncio.sleep(0.5)
                continue
            
            # Execute ready tasks concurrently (up to 3 at a time)
            batch = ready_tasks[:3]
            await asyncio.gather(*[
                self._execute_task(project, task, mode)
                for task in batch
            ])
            
            # Check for cancellation
            if self._shutdown_event.is_set():
                break
        
        if iteration >= max_iterations:
            raise Exception("Max iterations reached - possible infinite loop")
    
    async def _execute_task(self, project: Project, task: Task, mode: str):
        """Execute single task with error handling and retries"""
        print(f"🔧 Executing: {task.description[:60]}...")
        
        state_manager.update_task(task)
        
        await self.send_update(project.id, WSMessageType.TASK_UPDATE, {
            "task_id": task.id,
            "description": task.description,
            "agent": task.agent_type.value,
            "status": "started"
        })
        
        memory = ProjectMemory(project.id)
        
        # Build rich context
        context = {
            "project_id": project.id,
            "task": task.description,
            "project_files": list(project.files.keys()),
            "tech_stack": project.config.tech_stack,
            "chat_history": memory.get_chat_history()[-5:],  # Last 5 messages
            "previous_outputs": {
                agent: memory.get_agent_output(agent, task.id)
                for agent in ["architect", "researcher"]
            }
        }
        
        # Competition mode for coding tasks
        if mode == "competition" and task.agent_type == AgentType.CODER:
            result = await self._run_coding_competition(project, task, context)
        else:
            # Normal execution
            start_time = datetime.utcnow()
            try:
                result = await self.pool.execute(
                    task.agent_type.value, 
                    task, 
                    context
                )
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                self.improvement.record_execution(
                    task.agent_type.value,
                    task.description,
                    True,
                    duration
                )
                
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                self.improvement.record_execution(
                    task.agent_type.value,
                    task.description,
                    False,
                    duration,
                    error_type=type(e).__name__
                )
                
                # Retry logic
                if task.metadata.get("retries", 0) < 2:
                    task.metadata["retries"] = task.metadata.get("retries", 0) + 1
                    task.status = TaskStatus.PENDING
                    state_manager.update_task(task)
                    return
                else:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    state_manager.update_task(task)
                    
                    await self.send_update(project.id, WSMessageType.ERROR, {
                        "task_id": task.id,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    return
        
        # Process result
        await self._process_result(project, task, result)
        
        task.status = TaskStatus.COMPLETED
        task.result = result if isinstance(result, dict) else {"output": str(result)}
        task.completed_at = datetime.utcnow()
        state_manager.update_task(task)
        
        await self.send_update(project.id, WSMessageType.TASK_UPDATE, {
            "task_id": task.id,
            "status": "completed",
            "result_summary": str(result)[:200] if result else None
        })
    
    async def _run_coding_competition(self, project: Project, task: Task, context: dict):
        """Run multiple coders and pick best result"""
        from agents.coder import CoderAgent  # Import here to avoid circular
        
        # Create temporary agents with different models
        coders = {
            "coder_gpt4": CoderAgent(model="gpt-4-turbo"),
            "coder_claude": CoderAgent(model="claude-3-opus"),
            "coder_groq": CoderAgent(model="groq/llama-3.1-70b")
        }
        
        # Temporarily register
        for name, agent in coders.items():
            self.pool.register(name, agent)
        
        try:
            # Define evaluator
            def evaluate_results(results: List[CodeGenerationResult]) -> int:
                scores = []
                for r in results:
                    if not isinstance(r, CodeGenerationResult):
                        scores.append(0)
                        continue
                    
                    # Score based on: file count, explanation quality, dependencies
                    score = len(r.files) * 10
                    score += len(r.explanation) / 100  # Longer explanation = more thought
                    score += len(r.dependencies_to_install) * 5  # Proper dependency management
                    scores.append(score)
                
                return scores.index(max(scores))
            
            # Run competition
            competition_result = await self.competition.run_competition(
                task.description,
                list(coders.keys()),
                context,
                evaluate_results
            )
            
            return competition_result["result"]
            
        finally:
            # Unregister temp agents
            for name in coders.keys():
                self.pool.unregister(name)
    
    async def _process_result(self, project: Project, task: Task, result: Any):
        """Process and store agent result"""
        memory = ProjectMemory(project.id)
        
        # Store output for other agents to reference
        memory.store_agent_output(task.agent_type.value, task.id, {
            "description": task.description,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Handle file generation
        if task.agent_type == AgentType.CODER and isinstance(result, CodeGenerationResult):
            for file in result.files:
                file_state = FileState(
                    path=file.path,
                    content=file.content,
                    language=file.language
                )
                state_manager.update_file_state(project.id, file_state)
                
                await self.send_update(project.id, WSMessageType.CODE_GENERATED, {
                    "path": file.path,
                    "language": file.language,
                    "preview": file.content[:500]
                })
        
        # Handle review results
        elif task.agent_type == AgentType.REVIEWER and isinstance(result, ReviewResult):
            if not result.passed:
                # Create fix tasks for failed review
                for issue in result.issues:
                    if issue.severity == "error":
                        fix_task = Task(
                            project_id=project.id,
                            description=f"Fix: {issue.message} in {issue.file_path}",
                            agent_type=AgentType.CODER,
                            priority=4,
                            metadata={"fix_for": task.id, "issue": issue.model_dump()}
                        )
                        state_manager.add_task(project.id, fix_task)
            
            await self.send_update(project.id, WSMessageType.PROGRESS, {
                "message": f"Code review: {result.score}/100",
                "issues_found": len(result.issues),
                "passed": result.passed
            })
    
    async def _phase_completion(self, project: Project):
        """Phase 3: Finalize and deploy"""
        print(f"✅ Phase 3: Completion for project {project.id}")
        
        state_manager.update_project_status(project.id, ProjectStatus.COMPLETED)
        
        # Generate summary
        stats = state_manager.get_project_stats(project.id)
        
        await self.send_update(project.id, WSMessageType.COMPLETED, {
            "message": "Project completed successfully!",
            "stats": stats,
            "files_created": list(project.files.keys())
        })
        
        # Auto-deploy if enabled
        if project.config.auto_deploy:
            await self._auto_deploy(project)
    
    async def _auto_deploy(self, project: Project):
        """Handle automatic deployment"""
        deploy_task = Task(
            project_id=project.id,
            description="Deploy project to production",
            agent_type=AgentType.DEPLOYER,
            priority=5
        )
        
        await self.send_update(project.id, WSMessageType.PROGRESS, {
            "message": "Starting deployment...",
            "agent": "deployer"
        })
        
        try:
            result = await self.pool.execute("deployer", deploy_task, {
                "project_id": project.id,
                "files": project.files,
                "config": project.config
            })
            
            await self.send_update(project.id, WSMessageType.PROGRESS, {
                "message": f"Deployed to {result.get('url', 'production')}",
                "deploy_url": result.get("url")
            })
            
        except Exception as e:
            await self.send_update(project.id, WSMessageType.ERROR, {
                "message": "Deployment failed",
                "error": str(e)
            })
    
    async def _handle_error(self, project: Project, error: Exception):
        """Handle project-level errors"""
        print(f"❌ Error in project {project.id}: {error}")
        
        state_manager.update_project_status(project.id, ProjectStatus.ERROR)
        
        await self.send_update(project.id, WSMessageType.ERROR, {
            "message": "Project failed",
            "error": str(error),
            "traceback": traceback.format_exc()
        })
        
        # Log for improvement
        self.improvement.record_execution(
            "orchestrator",
            f"project_{project.id}",
            False,
            0,
            error_type=type(error).__name__
        )
    
    async def pause_project(self, project_id: str):
        """Pause execution (for human review)"""
        # Implementation: Set flag, agents check before continuing
        pass
    
    async def resume_project(self, project_id: str):
        """Resume paused project"""
        pass
    
    async def cancel_project(self, project_id: str):
        """Cancel running project"""
        self._shutdown_event.set()
        self.running_projects.discard(project_id)
    
    def get_system_status(self) -> dict:
        """Get complete system status"""
        return {
            "active_projects": len(self.running_projects),
            "agent_status": self.pool.agent_status,
            "performance": {
                agent: self.improvement.get_agent_stats(agent)
                for agent in self.pool.active_agents.keys()
            }
        }


# Global orchestrator instance
orchestrator = Orchestrator()
