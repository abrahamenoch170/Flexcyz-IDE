"""
Redis-based memory layer for short-term and long-term storage
Falls back to JSON file if Redis unavailable
"""

import json
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import redis
from config import get_settings


class MemoryLayer:
    """
    Unified interface for project memory and context persistence
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.fallback_dir = os.path.join(self.settings.PROJECTS_DIR, ".memory")
        os.makedirs(self.fallback_dir, exist_ok=True)
        
        self._connect_redis()
    
    def _connect_redis(self):
        """Try to connect to Redis, fall back to file storage"""
        try:
            self.redis_client = redis.from_url(
                self.settings.REDIS_URL,
                decode_responses=False,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            print("✅ Connected to Redis")
        except Exception as e:
            print(f"⚠️ Redis unavailable, using file fallback: {e}")
            self.redis_client = None
    
    def _get_key(self, project_id: str, namespace: str) -> str:
        """Generate consistent key"""
        return f"flexcyz:{project_id}:{namespace}"
    
    def _get_fallback_path(self, project_id: str, namespace: str) -> str:
        """Get file path for fallback storage"""
        return os.path.join(self.fallback_dir, f"{project_id}_{namespace}.json")
    
    def set(self, project_id: str, namespace: str, key: str, value: Any, expire: Optional[int] = None):
        """
        Store value in memory
        expire: seconds until expiration (None = no expiration)
        """
        full_key = f"{self._get_key(project_id, namespace)}:{key}"
        
        # Serialize
        if isinstance(value, (dict, list, str, int, float, bool)):
            serialized = json.dumps(value).encode()
        else:
            serialized = pickle.dumps(value)
        
        if self.redis_client:
            self.redis_client.set(full_key, serialized, ex=expire)
        else:
            # File fallback
            path = self._get_fallback_path(project_id, namespace)
            data = {}
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
            
            data[key] = {
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "expire_at": (datetime.utcnow() + timedelta(seconds=expire)).isoformat() if expire else None
            }
            
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def get(self, project_id: str, namespace: str, key: str, default: Any = None) -> Any:
        """Retrieve value from memory"""
        full_key = f"{self._get_key(project_id, namespace)}:{key}"
        
        if self.redis_client:
            data = self.redis_client.get(full_key)
            if data:
                try:
                    return json.loads(data)
                except:
                    return pickle.loads(data)
            return default
        else:
            # File fallback
            path = self._get_fallback_path(project_id, namespace)
            if not os.path.exists(path):
                return default
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            if key not in data:
                return default
            
            entry = data[key]
            # Check expiration
            if entry.get("expire_at"):
                expire_time = datetime.fromisoformat(entry["expire_at"])
                if datetime.utcnow() > expire_time:
                    del data[key]
                    with open(path, 'w') as f:
                        json.dump(data, f)
                    return default
            
            return entry["value"]
    
    def get_all(self, project_id: str, namespace: str) -> Dict[str, Any]:
        """Get all values in a namespace"""
        pattern = self._get_key(project_id, namespace)
        
        if self.redis_client:
            result = {}
            for key in self.redis_client.scan_iter(match=f"{pattern}:*"):
                key_str = key.decode() if isinstance(key, bytes) else key
                short_key = key_str.split(":")[-1]
                result[short_key] = self.get(project_id, namespace, short_key)
            return result
        else:
            path = self._get_fallback_path(project_id, namespace)
            if not os.path.exists(path):
                return {}
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Filter expired
            now = datetime.utcnow()
            valid = {}
            for k, v in data.items():
                if v.get("expire_at"):
                    if now < datetime.fromisoformat(v["expire_at"]):
                        valid[k] = v["value"]
                else:
                    valid[k] = v["value"]
            
            return valid
    
    def delete(self, project_id: str, namespace: str, key: str):
        """Delete a key"""
        full_key = f"{self._get_key(project_id, namespace)}:{key}"
        
        if self.redis_client:
            self.redis_client.delete(full_key)
        else:
            path = self._get_fallback_path(project_id, namespace)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                
                if key in data:
                    del data[key]
                    with open(path, 'w') as f:
                        json.dump(data, f)
    
    def append(self, project_id: str, namespace: str, key: str, value: Any, max_items: int = 100):
        """
        Append to a list (useful for chat history, events)
        Automatically trims to max_items
        """
        current = self.get(project_id, namespace, key, [])
        if not isinstance(current, list):
            current = [current]
        
        current.append(value)
        
        # Trim
        if len(current) > max_items:
            current = current[-max_items:]
        
        self.set(project_id, namespace, key, current)
    
    def increment(self, project_id: str, namespace: str, key: str, amount: int = 1) -> int:
        """Atomic increment"""
        if self.redis_client:
            full_key = f"{self._get_key(project_id, namespace)}:{key}"
            return self.redis_client.incr(full_key, amount)
        else:
            current = self.get(project_id, namespace, key, 0)
            new_val = current + amount
            self.set(project_id, namespace, key, new_val)
            return new_val
    
    def clear_project(self, project_id: str):
        """Clear all data for a project"""
        if self.redis_client:
            pattern = f"flexcyz:{project_id}:*"
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)
        else:
            # Delete fallback files
            for filename in os.listdir(self.fallback_dir):
                if filename.startswith(project_id):
                    os.remove(os.path.join(self.fallback_dir, filename))


# Global instance
memory = MemoryLayer()


class ProjectMemory:
    """
    High-level interface for project-specific memory operations
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.mem = memory
    
    # Context/Session
    def add_context(self, key: str, value: Any):
        """Add short-term context"""
        self.mem.set(self.project_id, "context", key, value, expire=3600)  # 1 hour
    
    def get_context(self, key: str, default=None):
        return self.mem.get(self.project_id, "context", key, default)
    
    # Events/Logs
    def log_event(self, event_type: str, data: dict):
        """Log an event"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.mem.append(self.project_id, "events", "history", event, max_items=1000)
    
    def get_events(self, limit: int = 100) -> List[dict]:
        """Get recent events"""
        events = self.mem.get(self.project_id, "events", "history", [])
        return events[-limit:] if events else []
    
    # Learned preferences (long-term)
    def learn_preference(self, category: str, key: str, value: Any):
        """Learn user preferences for future projects"""
        current = self.mem.get(self.project_id, "preferences", category, {})
        current[key] = value
        self.mem.set(self.project_id, "preferences", category, current, expire=None)  # No expire
    
    def get_preference(self, category: str, key: str, default=None):
        return self.mem.get(self.project_id, "preferences", category, {}).get(key, default)
    
    # File operations tracking
    def track_file_operation(self, operation: dict):
        self.mem.append(self.project_id, "files", "operations", operation, max_items=500)
    
    def get_file_operations(self) -> List[dict]:
        return self.mem.get(self.project_id, "files", "operations", [])
    
    # Agent outputs
    def store_agent_output(self, agent_type: str, task_id: str, output: dict):
        key = f"{agent_type}:{task_id}"
        self.mem.set(self.project_id, "agent_outputs", key, output, expire=86400)  # 24 hours
    
    def get_agent_output(self, agent_type: str, task_id: str) -> Optional[dict]:
        key = f"{agent_type}:{task_id}"
        return self.mem.get(self.project_id, "agent_outputs", key)
    
    # Chat/Conversation history
    def add_message(self, role: str, content: str, metadata: dict = None):
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.mem.append(self.project_id, "chat", "messages", message, max_items=50)
    
    def get_chat_history(self) -> List[dict]:
        return self.mem.get(self.project_id, "chat", "messages", [])
    
    def clear(self):
        """Clear all project memory"""
        self.mem.clear_project(self.project_id)
