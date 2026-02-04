"""
Deployer Agent - Deploy to Vercel, Railway, Supabase
Handles build configuration, environment variables, and deployment
"""

import os
import subprocess
import json
from typing import Any, Dict, List, Optional

import httpx

from agents.base import AgentCapability, BaseAgent
from models.schemas import DeploymentConfig, DeploymentTarget, DeploymentResult


class DeployerAgent(BaseAgent):
    """
    Multi-platform deployment automation
    """
    
    def __init__(self):
        super().__init__(
            name="Deployer",
            capabilities=[AgentCapability.DEPLOYING],
            preferred_model=None,
            system_prompt=""
        )
        
        # Platform credentials
        self.vercel_token = os.getenv("VERCEL_TOKEN")
        self.railway_token = os.getenv("RAILWAY_TOKEN")
        self.supabase_key = os.getenv("SUPABASE_KEY")
    
    async def execute(self, instruction: str, context: Dict[str, Any]) -> DeploymentResult:
        """
        Deploy project to specified platform
        """
        print(f"🚀 Deploying: {instruction[:80]}...")
        
        project_id = context.get("project_id")
        project_path = f"./projects/{project_id}"
        
        # Detect target from instruction or context
        target = self._detect_target(instruction, context)
        
        if target == DeploymentTarget.VERCEL:
            return await self._deploy_vercel(project_path, context)
        elif target == DeploymentTarget.RAILWAY:
            return await self._deploy_railway(project_path, context)
        elif target == DeploymentTarget.SUPABASE:
            return await self._deploy_supabase(project_path, context)
        else:
            return DeploymentResult(
                success=False,
                error="Unknown deployment target"
            )
    
    def _detect_target(self, instruction: str, context: Dict) -> DeploymentTarget:
        """Detect deployment target from context"""
        instruction_lower = instruction.lower()
        
        if "vercel" in instruction_lower:
            return DeploymentTarget.VERCEL
        elif "railway" in instruction_lower:
            return DeploymentTarget.RAILWAY
        elif "supabase" in instruction_lower:
            return DeploymentTarget.SUPABASE
        
        # Detect from project type
        tech_stack = context.get("tech_stack", [])
        if "Next.js" in tech_stack or "React" in tech_stack:
            return DeploymentTarget.VERCEL
        elif "Node.js" in tech_stack:
            return DeploymentTarget.RAILWAY
        
        return DeploymentTarget.VERCEL  # Default
    
    async def _deploy_vercel(self, project_path: str, context: Dict) -> DeploymentResult:
        """Deploy to Vercel"""
        if not self.vercel_token:
            return DeploymentResult(
                success=False,
                error="VERCEL_TOKEN not configured"
            )
        
        try:
            # Check if vercel CLI is available
            result = subprocess.run(
                ["which", "vercel"],
                capture_output=True
            )
            
            if result.returncode != 0:
                # Try using API directly
                return await self._deploy_vercel_api(project_path, context)
            
            # Use CLI
            os.chdir(project_path)
            
            # Link or create project
            env = os.environ.copy()
            env["VERCEL_TOKEN"] = self.vercel_token
            
            # Deploy
            process = subprocess.run(
                ["vercel", "--yes", "--prod"],
                capture_output=True,
                text=True,
                env=env
            )
            
            if process.returncode != 0:
                return DeploymentResult(
                    success=False,
                    error=process.stderr,
                    logs=[process.stdout, process.stderr]
                )
            
            # Extract URL from output
            url = self._extract_url(process.stdout)
            
            return DeploymentResult(
                success=True,
                url=url,
                logs=[process.stdout]
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=str(e)
            )
    
    async def _deploy_vercel_api(self, project_path: str, context: Dict) -> DeploymentResult:
        """Deploy using Vercel REST API"""
        # Create deployment via API
        async with httpx.AsyncClient() as client:
            # Get project files
            files = self._get_deployment_files(project_path)
            
            # Create deployment
            response = await client.post(
                "https://api.vercel.com/v13/deployments",
                headers={"Authorization": f"Bearer {self.vercel_token}"},
                json={
                    "name": context.get("project_name", "flexcyz-project"),
                    "files": files,
                    "framework": self._detect_framework(project_path)
                }
            )
            
            if response.status_code != 200:
                return DeploymentResult(
                    success=False,
                    error=f"Vercel API error: {response.text}"
                )
            
            data = response.json()
            
            return DeploymentResult(
                success=True,
                url=data.get("url"),
                logs=["Deployed via API"]
            )
    
    async def _deploy_railway(self, project_path: str, context: Dict) -> DeploymentResult:
        """Deploy to Railway"""
        if not self.railway_token:
            return DeploymentResult(
                success=False,
                error="RAILWAY_TOKEN not configured"
            )
        
        try:
            os.chdir(project_path)
            
            # Check for railway CLI
            result = subprocess.run(
                ["railway", "version"],
                capture_output=True
            )
            
            if result.returncode != 0:
                return DeploymentResult(
                    success=False,
                    error="Railway CLI not installed"
                )
            
            # Login and deploy
            env = os.environ.copy()
            env["RAILWAY_TOKEN"] = self.railway_token
            
            # Link project
            subprocess.run(
                ["railway", "link"],
                capture_output=True,
                env=env
            )
            
            # Deploy
            process = subprocess.run(
                ["railway", "up"],
                capture_output=True,
                text=True,
                env=env
            )
            
            if process.returncode != 0:
                return DeploymentResult(
                    success=False,
                    error=process.stderr
                )
            
            # Get domain
            domain_process = subprocess.run(
                ["railway", "domain"],
                capture_output=True,
                text=True,
                env=env
            )
            
            return DeploymentResult(
                success=True,
                url=domain_process.stdout.strip(),
                logs=[process.stdout]
            )
            
        except Exception as e:
            return DeploymentResult(
                success=False,
                error=str(e)
            )
    
    async def _deploy_supabase(self, project_path: str, context: Dict) -> DeploymentResult:
        """Deploy to Supabase (Edge Functions or DB)"""
        # Implementation for Supabase deployment
        return DeploymentResult(
            success=False,
            error="Supabase deployment not yet implemented"
        )
    
    def _get_deployment_files(self, project_path: str) -> List[Dict]:
        """Get files for API deployment"""
        files = []
        
        for root, dirs, filenames in os.walk(project_path):
            # Skip node_modules and git
            dirs[:] = [d for d in dirs if d not in ["node_modules", ".git", "__pycache__"]]
            
            for filename in filenames:
                filepath = os.path.join(root, filename)
                relpath = os.path.relpath(filepath, project_path)
                
                with open(filepath, "rb") as f:
                    content = f.read()
                
                files.append({
                    "file": relpath,
                    "data": content.decode("utf-8", errors="ignore")
                })
        
        return files
    
    def _detect_framework(self, project_path: str) -> Optional[str]:
        """Detect framework from project files"""
        if os.path.exists(os.path.join(project_path, "next.config.js")):
            return "nextjs"
        elif os.path.exists(os.path.join(project_path, "vite.config.ts")):
            return "vite"
        elif os.path.exists(os.path.join(project_path, "package.json")):
            return "create-react-app"
        
        return None
    
    def _extract_url(self, output: str) -> Optional[str]:
        """Extract deployment URL from CLI output"""
        import re
        match = re.search(r"(https?://[^\s]+)", output)
        return match.group(1) if match else None
    
    async def preview_locally(self, project_path: str) -> Dict:
        """
        Start local preview server (simulation mode)
        """
        try:
            # Detect project type and start appropriate server
            if os.path.exists(os.path.join(project_path, "package.json")):
                with open(os.path.join(project_path, "package.json")) as f:
                    pkg = json.load(f)
                
                if "dev" in pkg.get("scripts", {}):
                    cmd = ["npm", "run", "dev"]
                elif "start" in pkg.get("scripts", {}):
                    cmd = ["npm", "start"]
                else:
                    cmd = ["npx", "serve", "dist"] if os.path.exists(os.path.join(project_path, "dist")) else ["python", "-m", "http.server", "3000"]
            else:
                cmd = ["python", "-m", "http.server", "3000"]
            
            # Start process (non-blocking)
            process = subprocess.Popen(
                cmd,
                cwd=project_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            return {
                "success": True,
                "pid": process.pid,
                "command": " ".join(cmd),
                "url": "http://localhost:3000"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
