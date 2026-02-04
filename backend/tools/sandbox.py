"""
Sandbox Executor - Safe code execution in Docker containers
Prevents malicious code from affecting host system
"""

import asyncio
import os
import tempfile
import uuid

import docker
from docker.errors import DockerException, ContainerError


class SandboxExecutor:
    """
    Secure code execution environment
    """
    
    def __init__(self):
        self.memory_limit = os.getenv("SANDBOX_MEMORY", "256m")
        self.cpu_limit = float(os.getenv("SANDBOX_CPU", "0.5"))
        self.timeout = int(os.getenv("SANDBOX_TIMEOUT", "30"))
        
        try:
            self.client = docker.from_env()
            self.available = True
        except DockerException:
            print("⚠️ Docker not available, sandbox disabled")
            self.available = False
    
    async def execute_python(
        self, 
        code: str, 
        dependencies: list = None,
        files: dict = None
    ) -> dict:
        """
        Execute Python code safely in container
        """
        if not self.available:
            return {
                "success": False,
                "error": "Sandbox not available",
                "output": ""
            }
        
        # Create temp directory for execution
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write main script
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)
            
            # Write additional files if provided
            if files:
                for path, content in files.items():
                    full_path = os.path.join(tmpdir, path)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    with open(full_path, "w") as f:
                        f.write(content)
            
            # Create requirements.txt if dependencies
            if dependencies:
                req_path = os.path.join(tmpdir, "requirements.txt")
                with open(req_path, "w") as f:
                    f.write("\n".join(dependencies))
            
            try:
                # Run container
                container = self.client.containers.run(
                    "python:3.11-slim",
                    command="python /code/script.py",
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    mem_limit=self.memory_limit,
                    cpu_quota=int(self.cpu_limit * 100000),
                    network_mode="none",  # No network access
                    detach=True,
                    working_dir="/code"
                )
                
                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=self.timeout)
                    logs = container.logs().decode("utf-8", errors="ignore")
                    
                    return {
                        "success": result["StatusCode"] == 0,
                        "exit_code": result["StatusCode"],
                        "output": logs,
                        "error": None if result["StatusCode"] == 0 else "Execution failed"
                    }
                    
                except Exception as e:
                    container.kill()
                    return {
                        "success": False,
                        "error": f"Timeout or error: {str(e)}",
                        "output": ""
                    }
                finally:
                    container.remove(force=True)
                    
            except ContainerError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": ""
                }
    
    async def execute_node(
        self,
        code: str,
        dependencies: list = None
    ) -> dict:
        """
        Execute Node.js code safely
        """
        if not self.available:
            return {"success": False, "error": "Sandbox not available"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write script
            script_path = os.path.join(tmpdir, "script.js")
            with open(script_path, "w") as f:
                f.write(code)
            
            # Write package.json if dependencies
            if dependencies:
                pkg = {
                    "name": "sandbox",
                    "version": "1.0.0",
                    "dependencies": {dep: "latest" for dep in dependencies}
                }
                import json
                with open(os.path.join(tmpdir, "package.json"), "w") as f:
                    json.dump(pkg, f)
            
            try:
                container = self.client.containers.run(
                    "node:20-slim",
                    command="sh -c 'npm install && node /code/script.js'",
                    volumes={tmpdir: {"bind": "/code", "mode": "rw"}},
                    mem_limit=self.memory_limit,
                    cpu_quota=int(self.cpu_limit * 100000),
                    network_mode="none",
                    detach=True,
                    working_dir="/code"
                )
                
                result = container.wait(timeout=self.timeout)
                logs = container.logs().decode("utf-8", errors="ignore")
                
                container.remove(force=True)
                
                return {
                    "success": result["StatusCode"] == 0,
                    "exit_code": result["StatusCode"],
                    "output": logs
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def validate_syntax(self, code: str, language: str) -> tuple[bool, str]:
        """
        Validate code syntax without executing
        """
        if language == "python":
            try:
                import ast
                ast.parse(code)
                return True, "Valid Python syntax"
            except SyntaxError as e:
                return False, f"Syntax error at line {e.lineno}: {e.msg}"
        
        elif language in ["javascript", "typescript"]:
            # Could use Node.js --check or similar
            return True, "Syntax check not implemented for JS/TS"
        
        return True, "Unknown language, skipping validation"
