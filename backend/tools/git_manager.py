"""
Git Manager - Git operations wrapper
"""

import os
from pathlib import Path
from typing import List, Optional

from git import Repo, GitCommandError


class GitManager:
    """
    High-level Git operations
    """
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.repo: Optional[Repo] = None
        
        # Try to load existing repo
        self._load_repo()
    
    def _load_repo(self):
        """Load or initialize repository"""
        try:
            self.repo = Repo(self.project_path)
        except:
            self.repo = None
    
    def init(self) -> bool:
        """Initialize new repository"""
        try:
            self.repo = Repo.init(self.project_path)
            return True
        except Exception as e:
            print(f"Git init error: {e}")
            return False
    
    def is_repo(self) -> bool:
        """Check if directory is a git repository"""
        return self.repo is not None
    
    def get_status(self) -> dict:
        """Get repository status"""
        if not self.repo:
            return {"error": "Not a repository"}
        
        try:
            return {
                "is_dirty": self.repo.is_dirty(),
                "untracked_files": self.repo.untracked_files,
                "active_branch": self.repo.active_branch.name,
                "commits_ahead": 0,  # Simplified
                "recent_commits": [
                    {"message": c.message, "author": str(c.author)}
                    for c in list(self.repo.iter_commits())[:5]
                ]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def stage_all(self) -> bool:
        """Stage all changes"""
        if not self.repo:
            return False
        
        try:
            self.repo.git.add(".")
            return True
        except Exception as e:
            print(f"Stage error: {e}")
            return False
    
    def commit(self, message: str) -> bool:
        """Create commit"""
        if not self.repo:
            return False
        
        try:
            if self.repo.is_dirty() or self.repo.untracked_files:
                self.repo.git.add(".")
                self.repo.git.commit("-m", message)
                return True
            return False
        except Exception as e:
            print(f"Commit error: {e}")
            return False
    
    def get_diff(self, staged: bool = False) -> str:
        """Get diff of changes"""
        if not self.repo:
            return ""
        
        try:
            if staged:
                return self.repo.git.diff("--cached")
            return self.repo.git.diff()
        except Exception as e:
            return str(e)
    
    def add_remote(self, name: str, url: str) -> bool:
        """Add remote repository"""
        if not self.repo:
            return False
        
        try:
            self.repo.create_remote(name, url)
            return True
        except Exception as e:
            print(f"Add remote error: {e}")
            return False
    
    def push(self, remote: str = "origin", branch: str = None) -> bool:
        """Push to remote"""
        if not self.repo:
            return False
        
        try:
            if not branch:
                branch = self.repo.active_branch.name
            
            origin = self.repo.remote(remote)
            origin.push(refspec=f"{branch}:{branch}")
            return True
        except Exception as e:
            print(f"Push error: {e}")
            return False
    
    def pull(self, remote: str = "origin", branch: str = None) -> bool:
        """Pull from remote"""
        if not self.repo:
            return False
        
        try:
            origin = self.repo.remote(remote)
            origin.pull()
            return True
        except Exception as e:
            print(f"Pull error: {e}")
            return False
    
    def create_branch(self, branch_name: str, checkout: bool = True) -> bool:
        """Create new branch"""
        if not self.repo:
            return False
        
        try:
            new_branch = self.repo.create_head(branch_name)
            if checkout:
                new_branch.checkout()
            return True
        except Exception as e:
            print(f"Branch error: {e}")
            return False
    
    def get_branches(self) -> List[str]:
        """List all branches"""
        if not self.repo:
            return []
        
        try:
            return [str(b) for b in self.repo.branches]
        except:
            return []
    
    def checkout(self, branch_name: str) -> bool:
        """Checkout branch"""
        if not self.repo:
            return False
        
        try:
            self.repo.git.checkout(branch_name)
            return True
        except Exception as e:
            print(f"Checkout error: {e}")
            return False
