"""
Flexcyz Backend - FastAPI + WebSocket Server
Real-time AI IDE with multi-agent orchestration
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config import get_settings
from core import orchestrator, state_manager
from core.memory import ProjectMemory
from models.schemas import (
    CreateProjectRequest, Project, ProjectResponse, ProjectStatus,
    TaskResponse, WSMessageType, WebSocketMessage
)

# ============== LIFESPAN MANAGEMENT ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("🚀 Flexcyz starting up...")
    
    # Initialize agents (will be imported here to avoid circular deps)
    await initialize_agents()
    
    print("✅ All systems operational")
    yield
    
    print("🛑 Shutting down...")
    # Cleanup
    for project_id in list(orchestrator.running_projects):
        await orchestrator.cancel_project(project_id)


# ============== APP INITIALIZATION ==============

app = FastAPI(
    title="Flexcyz IDE",
    description="Autonomous AI Software Engineer",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== AGENT INITIALIZATION ==============

async def initialize_agents():
    """Register all agents with the orchestrator"""
    from agents.planner import PlannerAgent
    from agents.architect import ArchitectAgent
    from agents.coder import CoderAgent
    from agents.reviewer import ReviewerAgent
    from agents.researcher import ResearcherAgent
    from agents.github_agent import GitHubAgent
    from agents.deployer import DeployerAgent
    
    # Core agents
    orchestrator.register_agent("planner", PlannerAgent())
    orchestrator.register_agent("architect", ArchitectAgent())
    orchestrator.register_agent("coder", CoderAgent())
    orchestrator.register_agent("reviewer", ReviewerAgent())
    
    # Support agents
    orchestrator.register_agent("researcher", ResearcherAgent())
    orchestrator.register_agent("github", GitHubAgent())
    orchestrator.register_agent("deployer", DeployerAgent())
    
    print("🤖 Agents registered:")
    for name in orchestrator.pool.active_agents.keys():
        print(f"   - {name}")


# ============== REST API ENDPOINTS ==============

@app.get("/")
async def root():
    return {
        "name": "Flexcyz IDE",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "projects": "/api/projects",
            "websocket": "/ws/{project_id}"
        }
    }


@app.get("/health")
async def health_check():
    """System health and status"""
    return {
        "status": "healthy",
        "agents": orchestrator.get_system_status(),
        "projects_active": len(orchestrator.running_projects)
    }


# ============== PROJECT ENDPOINTS ==============

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(request: CreateProjectRequest):
    """
    Create new project from prompt
    Returns immediately, use WebSocket to track progress
    """
    try:
        project = await orchestrator.create_project(
            prompt=request.prompt,
            config=request.config
        )
        
        # Start execution in background
        asyncio.create_task(
            orchestrator.run_project(
                project.id, 
                request.prompt,
                mode="default"
            )
        )
        
        return ProjectResponse(
            project=project,
            message="Project created and execution started. Connect to WebSocket to track progress."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get project details"""
    project = state_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.get("/api/projects/{project_id}/status")
async def get_project_status(project_id: str):
    """Get detailed project status"""
    project = state_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    stats = state_manager.get_project_stats(project_id)
    tasks = state_manager.get_project_tasks(project_id)
    
    return {
        "project": project,
        "stats": stats,
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "agent": t.agent_type.value,
                "status": t.status.value,
                "created_at": t.created_at
            }
            for t in tasks
        ]
    }


@app.post("/api/projects/{project_id}/action")
async def project_action(project_id: str, action: str):
    """
    Control project execution
    Actions: pause, resume, cancel
    """
    if action == "cancel":
        await orchestrator.cancel_project(project_id)
        return {"message": "Project cancelled"}
    elif action == "pause":
        await orchestrator.pause_project(project_id)
        return {"message": "Project paused"}
    elif action == "resume":
        await orchestrator.resume_project(project_id)
        return {"message": "Project resumed"}
    else:
        raise HTTPException(status_code=400, detail="Invalid action")


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete project and all data"""
    # Cancel if running
    if project_id in orchestrator.running_projects:
        await orchestrator.cancel_project(project_id)
    
    # Clear memory
    memory = ProjectMemory(project_id)
    memory.clear()
    
    # Remove from disk
    import shutil
    project_path = os.path.join(get_settings().PROJECTS_DIR, project_id)
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
    
    return {"message": "Project deleted"}


# ============== FILE ENDPOINTS ==============

@app.get("/api/projects/{project_id}/files")
async def list_files(project_id: str):
    """List all files in project"""
    files = state_manager.list_files(project_id)
    return {
        "project_id": project_id,
        "files": [
            {
                "path": f.path,
                "language": f.language,
                "size": len(f.content),
                "modified": f.last_modified
            }
            for f in files
        ]
    }


@app.get("/api/projects/{project_id}/files/{file_path:path}")
async def get_file(project_id: str, file_path: str):
    """Get file content"""
    content = state_manager.get_file_content(project_id, file_path)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "path": file_path,
        "content": content
    }


@app.post("/api/projects/{project_id}/files/{file_path:path}")
async def update_file(project_id: str, file_path: str, content: str):
    """Update file content (user edit)"""
    from models.schemas import FileState
    
    file_state = FileState(
        path=file_path,
        content=content
    )
    state_manager.update_file_state(project_id, file_state)
    
    # Log user edit
    memory = ProjectMemory(project_id)
    memory.log_event("user_edit", {"file": file_path})
    
    return {"message": "File updated"}


# ============== AGENT ENDPOINTS ==============

@app.get("/api/agents")
async def list_agents():
    """List all registered agents and their status"""
    return orchestrator.get_system_status()


@app.post("/api/agents/{agent_name}/reload")
async def reload_agent(agent_name: str):
    """Hot-reload an agent (for development)"""
    # Implementation would reimport module and reregister
    return {"message": f"Agent {agent_name} reloaded"}


# ============== WEBSOCKET ENDPOINT ==============

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    """
    Real-time bidirectional communication
    - Server sends: progress updates, file changes, agent outputs
    - Client sends: feedback, pause/resume, file edits
    """
    await websocket.accept()
    
    # Register with orchestrator
    orchestrator.register_websocket(project_id, websocket)
    
    try:
        # Send initial state
        project = state_manager.get_project(project_id)
        if project:
            await websocket.send_json({
                "type": "init",
                "project": project.model_dump(),
                "files": list(project.files.keys())
            })
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type")
            payload = message.get("payload", {})
            
            if msg_type == "feedback":
                # User feedback on generated code
                memory = ProjectMemory(project_id)
                memory.add_message(
                    "user", 
                    payload.get("message"),
                    {"feedback_type": payload.get("feedback_type")}
                )
                
                # If negative feedback, trigger revision
                if payload.get("feedback_type") == "negative":
                    await handle_negative_feedback(project_id, payload)
                
            elif msg_type == "chat":
                # User chat message during execution
                memory = ProjectMemory(project_id)
                memory.add_message("user", payload.get("message"))
                
                # Could trigger agent to respond
                
            elif msg_type == "action":
                # User action (pause, resume, etc.)
                action = payload.get("action")
                if action == "cancel":
                    await orchestrator.cancel_project(project_id)
                    
            elif msg_type == "file_request":
                # User requesting specific file
                file_path = payload.get("path")
                content = state_manager.get_file_content(project_id, file_path)
                await websocket.send_json({
                    "type": "file_content",
                    "path": file_path,
                    "content": content
                })
                
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for project {project_id}")
        orchestrator.websocket_clients.pop(project_id, None)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


async def handle_negative_feedback(project_id: str, payload: dict):
    """Handle user negative feedback - trigger fixes"""
    from models.schemas import AgentType, Task
    
    # Create correction task
    correction_task = Task(
        project_id=project_id,
        description=f"Fix based on user feedback: {payload.get('message')}",
        agent_type=AgentType.CODER,
        priority=5,
        metadata={
            "feedback": payload.get("message"),
            "original_file": payload.get("file_path")
        }
    )
    
    state_manager.add_task(project_id, correction_task)
    
    # Notify
    ws = orchestrator.websocket_clients.get(project_id)
    if ws:
        await ws.send_json({
            "type": "correction_queued",
            "message": "Fix task created based on your feedback"
        })


# ============== SPECIAL ENDPOINTS ==============

@app.post("/api/projects/{project_id}/competition")
async def run_competition_mode(project_id: str, prompt: str):
    """
    Run project in competition mode
    Multiple agents compete to produce best result
    """
    project = await orchestrator.create_project(
        prompt=prompt,
        config=None
    )
    
    asyncio.create_task(
        orchestrator.run_project(project.id, prompt, mode="competition")
    )
    
    return {"project_id": project.id, "mode": "competition"}


@app.get("/api/projects/{project_id}/download")
async def download_project(project_id: str):
    """Download project as zip"""
    import zipfile
    from io import BytesIO
    
    project_path = os.path.join(get_settings().PROJECTS_DIR, project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Create zip in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file == "state.json":  # Skip internal state
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_path)
                zip_file.write(file_path, arcname)
    
    zip_buffer.seek(0)
    
    return FileResponse(
        zip_buffer,
        media_type="application/zip",
        filename=f"{project_id}.zip"
    )


# ============== ERROR HANDLERS ==============

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global error handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "detail": "Internal server error"
        }
    )


# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=get_settings().HOST,
        port=get_settings().PORT,
        reload=get_settings().DEBUG,
        workers=1  # Single worker for WebSocket state
    )
