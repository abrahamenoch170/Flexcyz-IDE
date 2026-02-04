"""
File Manager - CRUD operations with safety checks
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional


class FileManager:
    """
    Safe file operations within project boundaries
    """
    
    def __init__(self, base_path: str = "./projects"):
        self.base_path = Path(base_path).resolve()
    
    def _validate_path(self, path: str, project_id: str) -> Path:
        """
        Ensure path is within project directory (prevent directory traversal)
        """
        # Normalize path
        safe_path = (self.base_path / project_id / path).resolve()
        
        # Check it's within project
        project_path = (self.base_path / project_id).resolve()
        
        if not str(safe_path).startswith(str(project_path)):
            raise ValueError("Path traversal attempt detected")
        
        return safe_path
    
    def create_file(
        self,
        project_id: str,
        path: str,
        content: str,
        overwrite: bool = False
    ) -> bool:
        """
        Create new file
        """
        try:
            file_path = self._validate_path(path, project_id)
            
            # Check exists
            if file_path.exists() and not overwrite:
                return False
            
            # Create directories
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"File creation error: {e}")
            return False
    
    def read_file(self, project_id: str, path: str) -> Optional[str]:
        """
        Read file content
        """
        try:
            file_path = self._validate_path(path, project_id)
            
            if not file_path.exists():
                return None
            
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
                
        except Exception as e:
            print(f"File read error: {e}")
            return None
    
    def update_file(
        self,
        project_id: str,
        path: str,
        content: str
    ) -> bool:
        """
        Update existing file
        """
        try:
            file_path = self._validate_path(path, project_id)
            
            if not file_path.exists():
                return False
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"File update error: {e}")
            return False
    
    def delete_file(self, project_id: str, path: str) -> bool:
        """
        Delete file
        """
        try:
            file_path = self._validate_path(path, project_id)
            
            if not file_path.exists():
                return False
            
            if file_path.is_file():
                file_path.unlink()
            else:
                shutil.rmtree(file_path)
            
            return True
            
        except Exception as e:
            print(f"File delete error: {e}")
            return False
    
    def list_files(
        self,
        project_id: str,
        subpath: str = ""
    ) -> List[dict]:
        """
        List all files in project
        """
        try:
            project_path = self.base_path / project_id / subpath
            
            if not project_path.exists():
                return []
            
            files = []
            for item in project_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(self.base_path / project_id)
                    files.append({
                        "path": str(rel_path),
                        "size": item.stat().st_size,
                        "modified": item.stat().st_mtime
                    })
            
            return files
            
        except Exception as e:
            print(f"List files error: {e}")
            return []
    
    def rename_file(
        self,
        project_id: str,
        old_path: str,
        new_path: str
    ) -> bool:
        """
        Rename/move file
        """
        try:
            old_file = self._validate_path(old_path, project_id)
            new_file = self._validate_path(new_path, project_id)
            
            if not old_file.exists():
                return False
            
            new_file.parent.mkdir(parents=True, exist_ok=True)
            old_file.rename(new_file)
            
            return True
            
        except Exception as e:
            print(f"Rename error: {e}")
            return False
    
    def copy_file(
        self,
        project_id: str,
        source: str,
        destination: str
    ) -> bool:
        """
        Copy file within project
        """
        try:
            src = self._validate_path(source, project_id)
            dst = self._validate_path(destination, project_id)
            
            if not src.exists():
                return False
            
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            
            return True
            
        except Exception as e:
            print(f"Copy error: {e}")
            return False
