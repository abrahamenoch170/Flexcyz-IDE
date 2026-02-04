"""
GitHub Agent - Git operations, repository management, PR creation
Handles forking, committing, pushing, and collaboration
"""

import os
import subprocess
from typing import Any, Dict, List, Optional

from github import Github
from git import Repo, GitCommandError

from agents.base import AgentCapability, BaseAgent


class GitHubAgent(BaseAgent):
    """
    GitHub integration for repository operations
    """
    
    def __init__(self):
        super().__init__(
            name="GitHub",
            capabilities=[AgentCapability.REVIEWING],  # Closest match
            preferred_model=None,  # Uses GitHub API, not LLM
            system_prompt=""
        )
        
        self.token = os.getenv("GITHUB_TOKEN")
        self.g = Github(self.token) if self.token else None
        self.username = None
        
        if self.g:
            try:
                self.username = self.g.get_user().login
            except:
                pass
    
    async def execute(self, instruction: str, context: Dict[str, Any]) -> Dict:
        """
        Execute Git/GitHub operation
        """
        print(f"🐙 GitHub: {instruction[:80]}...")
        
        project_id = context.get("project_id")
        project_path = f"./projects/{project_id}" if project_id else None
        
        if "fork" in instruction.lower():
            return await self._fork_repo(instruction)
        elif "clone" in instruction.lower():
            return await self._clone_repo(instruction, project_path)
        elif "commit" in instruction.lower():
            return await self._commit_changes(instruction, project_path, context)
        elif "push" in instruction.lower():
            return await self._push_changes(project_path)
        elif "pr" in instruction.lower() or "pull request" in instruction.lower():
            return await self._create_pr(instruction, context)
        elif "init" in instruction.lower():
            return await self._init_repo(project_path)
        else:
            return {"error": "Unknown git operation"}
    
    async def _fork_repo(self, instruction: str) -> Dict:
        """Fork a repository"""
        import re
        
        # Extract repo from instruction
        match = re.search(r"([\w-]+/[\w-]+)", instruction)
        if not match:
            return {"error": "No repository found in instruction"}
        
        repo_name = match.group(1)
        
        if not self.g:
            return {"error": "GitHub not authenticated"}
        
        try:
            source_repo = self.g.get_repo(repo_name)
            user = self.g.get_user()
            fork = user.create_fork(source_repo)
            
            return {
                "status": "success",
                "forked_to": fork.full_name,
                "clone_url": fork.clone_url,
                "html_url": fork.html_url
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _clone_repo(self, instruction: str, project_path: str) -> Dict:
        """Clone repository to project"""
        import re
        
        url_match = re.search(r"(https?://github\.com/[\w-]+/[\w-]+)", instruction)
        if not url_match:
            return {"error": "No valid GitHub URL found"}
        
        url = url_match.group(1)
        if not url.endswith(".git"):
            url += ".git"
        
        try:
            if os.path.exists(project_path):
                # Already exists, check if it's a repo
                try:
                    Repo(project_path)
                    return {"status": "already_cloned", "path": project_path}
                except:
                    pass
            
            Repo.clone_from(url, project_path)
            return {
                "status": "success",
                "cloned_to": project_path,
                "url": url
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _init_repo(self, project_path: str) -> Dict:
        """Initialize new git repository"""
        try:
            if not os.path.exists(project_path):
                os.makedirs(project_path)
            
            repo = Repo.init(project_path)
            
            # Create .gitignore
            gitignore_content = """node_modules/
__pycache__/
*.pyc
.env
.env.local
dist/
build/
.DS_Store
"""
            with open(os.path.join(project_path, ".gitignore"), "w") as f:
                f.write(gitignore_content)
            
            repo.git.add(".gitignore")
            repo.git.commit("-m", "Initial commit: Add .gitignore")
            
            return {
                "status": "success",
                "path": project_path,
                "message": "Repository initialized"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _commit_changes(
        self, 
        instruction: str, 
        project_path: str, 
        context: Dict
    ) -> Dict:
        """Auto-generate commit message and commit"""
        if not project_path or not os.path.exists(project_path):
            return {"error": "Project path not found"}
        
        try:
            repo = Repo(project_path)
            
            # Check for changes
            if not repo.is_dirty(untracked_files=True):
                return {"status": "no_changes"}
            
            # Generate commit message
            diff = repo.git.diff("--staged") or repo.git.diff()
            
            if not diff:
                # Stage all if nothing staged
                repo.git.add(".")
                diff = repo.git.diff("--cached")
            
            # Use LLM to generate commit message
            commit_msg = await self._generate_commit_message(diff, instruction)
            
            # Commit
            repo.git.add(".")
            repo.git.commit("-m", commit_msg)
            
            return {
                "status": "committed",
                "message": commit_msg,
                "files_changed": len(repo.index.diff("HEAD"))
            }
        except GitCommandError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _generate_commit_message(self, diff: str, context: str) -> str:
        """Use LLM to generate meaningful commit message"""
        # Truncate diff if too long
        diff_preview = diff[:2000] if len(diff) > 2000 else diff
        
        # Simple heuristic if no LLM available
        if len(diff_preview) < 100:
            return "Update files"
        
        # Could call LLM here for better messages
        # For now, use pattern matching
        if "create" in context.lower() or "add" in context.lower():
            return "feat: Add initial implementation"
        elif "fix" in context.lower():
            return "fix: Resolve issues"
        elif "refactor" in context.lower():
            return "refactor: Improve code structure"
        else:
            return "feat: Update implementation"
    
    async def _push_changes(self, project_path: str) -> Dict:
        """Push to remote"""
        try:
            repo = Repo(project_path)
            
            if not repo.remotes:
                return {"error": "No remote configured"}
            
            origin = repo.remote("origin")
            origin.push()
            
            return {
                "status": "pushed",
                "branch": repo.active_branch.name,
                "remote": origin.url
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _create_pr(self, instruction: str, context: Dict) -> Dict:
        """Create pull request"""
        # Implementation for PR creation
        if not self.g:
            return {"error": "GitHub not authenticated"}
        
        # Extract repo and branch info
        return {"status": "not_implemented", "message": "PR creation coming soon"}
    
    def get_repo_info(self, repo_name: str) -> Dict:
        """Get repository information"""
        if not self.g:
            return {"error": "Not authenticated"}
        
        try:
            repo = self.g.get_repo(repo_name)
            return {
                "name": repo.full_name,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "language": repo.language,
                "description": repo.description
            }
        except Exception as e:
            return {"error": str(e)}
