"""
Project state management and lifecycle
Handles transitions between planning, coding, reviewing, deploying
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from models.schemas import (
    FileState, Project, ProjectConfig, ProjectStatus, Task, TaskStatus
)
from core.memory import ProjectMemory


class StateManager:
    """
    Manages the complete lifecycle of a project
    Persists state to disk + memory
    """
    
    def __init__(self, projects_dir: str):
        self.projects_dir = projects_dir
        os.makedirs(projects_dir, exist_ok=True)
        self._projects: Dict[str, Project] = {}
        self._tasks: Dict[str, Task] = {}
    
    def _get_project_path(self, project_id: str) -> str:
        return os.path.join(self.projects_dir, project_id)
    
    def _get_state_file(self, project_id: str) -> str:
        return os.path.join(self._get_project_path(project_id), "state.json")
    
    def create_project(self, config: ProjectConfig, user_id: Optional[str] = None) -> Project:
        """Initialize a new project"""
        project = Project(
            user_id=user_id,
            config=config
        )
        
        # Create directory structure
        project_path = self._get_project_path(project.id)
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(os.path.join(project_path, "src"), exist_ok=True)
        
        # Initialize memory
        memory = ProjectMemory(project.id)
        memory.add_context("created", datetime.utcnow().isoformat())
        memory.learn_preference("user", "tech_stack", config.tech_stack)
        
        # Save state
        self._projects[project.id] = project
        self._persist_project(project)
        
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get project by ID"""
        if project_id in self._projects:
            return self._projects[project_id]
        
        # Try to load from disk
        state_file = self._get_state_file(project_id)
        if os.path.exists(state_file):
            import json
            with open(state_file, 'r') as f:
                data = json.load(f)
                project = Project(**data)
                self._projects[project_id] = project
                return project
        
        return None
    
    def update_project_status(self, project_id: str, status: ProjectStatus, current_task_id: Optional[str] = None):
        """Update project status"""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        project.status = status
        project.updated_at = datetime.utcnow()
        if current_task_id:
            project.current_task_id = current_task_id
        
        self._persist_project(project)
        
        # Log event
        memory = ProjectMemory(project_id)
        memory.log_event("status_change", {
            "from": project.status,
            "to": status,
            "task_id": current_task_id
        })
    
    def add_task(self, project_id: str, task: Task) -> Task:
        """Add a task to project"""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        task.project_id = project_id
        self._tasks[task.id] = task
        project.tasks.append(task.id)
        
        self._persist_project(project)
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    def update_task(self, task: Task):
        """Update task state"""
        self._tasks[task.id] = task
        
        # Persist to project
        project = self.get_project(task.project_id)
        if project:
            self._persist_project(project)
    
    def get_project_tasks(self, project_id: str, status: Optional[TaskStatus] = None) -> List[Task]:
        """Get all tasks for a project, optionally filtered by status"""
        project = self.get_project(project_id)
        if not project:
            return []
        
        tasks = [self._tasks[tid] for tid in project.tasks if tid in self._tasks]
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    def get_ready_tasks(self, project_id: str) -> List[Task]:
        """
        Get tasks that are ready to execute (pending + all deps completed)
        """
        tasks = self.get_project_tasks(project_id, TaskStatus.PENDING)
        ready = []
        
        for task in tasks:
            deps_satisfied = all(
                self._tasks.get(dep_id) and 
                self._tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
            )
            if deps_satisfied:
                ready.append(task)
        
        # Sort by priority
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready
    
    def update_file_state(self, project_id: str, file_state: FileState):
        """Update file in project"""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        file_state.last_modified = datetime.utcnow()
        project.files[file_state.path] = file_state
        
        # Write actual file
        project_path = self._get_project_path(project_id)
        full_path = os.path.join(project_path, file_state.path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(file_state.content)
        
        self._persist_project(project)
    
    def get_file_content(self, project_id: str, path: str) -> Optional[str]:
        """Get file content"""
        project_path = self._get_project_path(project_id)
        full_path = os.path.join(project_path, path)
        
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                return f.read()
        
        # Check in memory
        project = self.get_project(project_id)
        if project and path in project.files:
            return project.files[path].content
        
        return None
    
    def delete_file(self, project_id: str, path: str):
        """Delete a file"""
        project = self.get_project(project_id)
        if not project:
            return
        
        # Remove from disk
        project_path = self._get_project_path(project_id)
        full_path = os.path.join(project_path, path)
        if os.path.exists(full_path):
            os.remove(full_path)
        
        # Remove from state
        if path in project.files:
            del project.files[path]
        
        self._persist_project(project)
    
    def list_files(self, project_id: str) -> List[FileState]:
        """List all files in project"""
        project = self.get_project(project_id)
        if not project:
            return []
        
        return list(project.files.values())
    
    def _persist_project(self, project: Project):
        """Save project state to disk"""
        state_file = self._get_state_file(project.id)
        with open(state_file, 'w') as f:
            # Custom JSON serialization for datetime
            data = project.model_dump()
            import json
            json.dump(data, f, indent=2, default=str)
    
    def get_project_stats(self, project_id: str) -> dict:
        """Get project statistics"""
        project = self.get_project(project_id)
        if not project:
            return {}
        
        tasks = self.get_project_tasks(project_id)
        
        return {
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "in_progress_tasks": len([t for t in tasks if t.status == TaskStatus.IN_PROGRESS]),
            "total_files": len(project.files),
            "total_tokens": project.total_tokens_used,
            "total_cost": project.total_cost,
            "duration_minutes": (datetime.utcnow() - project.created_at).total_seconds() / 60
        }


# Global instance
state_manager = StateManager("./projects")
