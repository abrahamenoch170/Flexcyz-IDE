"""
Architect Agent - Designs system structure, APIs, and data models
Makes high-level technical decisions before coding begins
"""

import json
from typing import Any, Dict, List

from agents.base import AgentCapability, BaseAgent
from models.schemas import AgentType


class ArchitectAgent(BaseAgent):
    """
    System architect that designs the blueprint for implementation
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        super().__init__(
            name="Architect",
            capabilities=[AgentCapability.PLANNING],
            preferred_model=model,
            temperature=0.2,
            system_prompt=self._load_system_prompt()
        )
    
    def _load_system_prompt(self) -> str:
        return """You are Flexcyz's Architect Agent. You design software systems, APIs, and data structures.

Your responsibilities:
1. Design clean, scalable architecture
2. Define API contracts (REST/GraphQL/WebSocket)
3. Create database schemas
4. Plan component hierarchy (for frontend)
5. Choose appropriate design patterns
6. Consider security, performance, and maintainability

ALWAYS provide concrete, implementable designs with code examples."""

    async def execute(self, instruction: str, context: Dict[str, Any]) -> Dict:
        """
        Design system architecture based on requirements
        """
        print(f"🏗️ Architect designing: {instruction[:80]}...")
        
        # Determine what to design
        if "api" in instruction.lower():
            return await self._design_api(instruction, context)
        elif "database" in instruction.lower() or "schema" in instruction.lower():
            return await self._design_database(instruction, context)
        elif "frontend" in instruction.lower() or "ui" in instruction.lower() or "component" in instruction.lower():
            return await self._design_frontend(instruction, context)
        else:
            # General system design
            return await self._design_system(instruction, context)
    
    async def _design_system(self, instruction: str, context: Dict) -> Dict:
        """
        Design overall system architecture
        """
        tech_stack = context.get("tech_stack", ["React", "Node.js"])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Design the system architecture for: {instruction}

Tech Stack: {', '.join(tech_stack)}

Provide:
1. High-level architecture diagram (describe in text)
2. Directory structure
3. Key modules/components and their responsibilities
4. Data flow
5. Technology choices with justification

Return JSON:
{{
    "architecture_pattern": "Microservices|Monolith|Serverless|etc",
    "directory_structure": {{
        "src/": "Source code",
        "src/components/": "React components",
        ...
    }},
    "components": [
        {{
            "name": "AuthService",
            "responsibility": "Handles authentication",
            "tech": "JWT + bcrypt",
            "interfaces": ["login", "register", "verify"]
        }}
    ],
    "data_flow": "Description of how data moves",
    "tech_choices": {{
        "frontend_framework": "React - for component ecosystem",
        "backend": "Express - lightweight and flexible",
        ...
    }},
    "design_decisions": [
        "Using repository pattern for data access",
        "JWT for stateless auth",
        ...
    ]
}}
"""}
        ]
        
        return await self.llm_json(messages)
    
    async def _design_api(self, instruction: str, context: Dict) -> Dict:
        """
        Design API endpoints
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Design the API for: {instruction}

Context: {context.get('prompt', 'N/A')}

Provide complete API design:
1. REST endpoints with methods, paths, request/response schemas
2. Authentication requirements
3. Error handling patterns
4. Rate limiting considerations

Return JSON:
{{
    "api_style": "REST",
    "base_path": "/api/v1",
    "endpoints": [
        {{
            "method": "POST",
            "path": "/users",
            "description": "Create new user",
            "request_body": {{
                "email": "string (required)",
                "password": "string (required, min 8 chars)"
            }},
            "response": {{
                "201": {{ "id": "uuid", "email": "string" }},
                "400": {{ "error": "Validation failed" }}
            }},
            "auth_required": false
        }}
    ],
    "auth_strategy": "JWT Bearer token",
    "error_format": {{
        "status": "error",
        "code": "ERROR_CODE",
        "message": "Human readable",
        "details": {{}}
    }}
}}
"""}
        ]
        
        return await self.llm_json(messages)
    
    async def _design_database(self, instruction: str, context: Dict) -> Dict:
        """
        Design database schema
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Design the database schema for: {instruction}

Provide:
1. Entity-Relationship diagram (describe)
2. Table/collection schemas with types and constraints
3. Indexes for performance
4. Relationships (foreign keys, references)
5. Migration strategy

Return JSON:
{{
    "database_type": "PostgreSQL|MongoDB|etc",
    "entities": [
        {{
            "name": "User",
            "table": "users",
            "fields": [
                {{"name": "id", "type": "UUID", "primary": true, "default": "gen_random_uuid()"}},
                {{"name": "email", "type": "VARCHAR(255)", "unique": true, "index": true}},
                {{"name": "password_hash", "type": "VARCHAR(255)", "nullable": false}},
                {{"name": "created_at", "type": "TIMESTAMP", "default": "NOW()"}}
            ],
            "constraints": ["email CHECK (email ~* '^\\\\S+@\\\\S+\\\\.\\\\S+$')"],
            "indexes": ["CREATE INDEX idx_users_email ON users(email)"]
        }}
    ],
    "relationships": [
        "User 1:N Posts (user_id FK)",
        "Post N:1 Category (category_id FK)"
    ],
    "migrations": [
        "001_create_users_table.sql",
        "002_create_posts_table.sql"
    ]
}}
"""}
        ]
        
        return await self.llm_json(messages)
    
    async def _design_frontend(self, instruction: str, context: Dict) -> Dict:
        """
        Design frontend architecture
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Design the frontend architecture for: {instruction}

Provide:
1. Component hierarchy/tree
2. State management approach
3. Routing structure
4. Styling strategy
5. Performance optimizations

Return JSON:
{{
    "framework": "React|Vue|etc",
    "state_management": "Zustand|Redux|Context",
    "component_tree": [
        {{
            "name": "App",
            "children": ["Header", "MainLayout", "Footer"],
            "state": "auth, theme",
            "props": []
        }},
        {{
            "name": "MainLayout",
            "children": ["Sidebar", "ContentArea"],
            "state": "sidebarOpen",
            "props": ["user"]
        }}
    ],
    "routes": [
        {{
            "path": "/",
            "component": "Home",
            "lazy": false,
            "protected": false
        }},
        {{
            "path": "/dashboard",
            "component": "Dashboard",
            "lazy": true,
            "protected": true
        }}
    ],
    "styling": {{
        "approach": "Tailwind + CSS Modules",
        "theme": "Dark mode support via CSS variables",
        "responsive": "Mobile-first breakpoints"
    }},
    "performance": [
        "Code splitting per route",
        "React.memo for list items",
        "Intersection Observer for lazy images"
    ]
}}
"""}
        ]
        
        return await self.llm_json(messages)
    
    def generate_file_structure(self, architecture: Dict) -> List[Dict]:
        """
        Convert architecture into specific files to create
        """
        files = []
        
        # Generate based on components
        for comp in architecture.get("components", []):
            if "frontend" in comp.get("type", ""):
                files.append({
                    "path": f"src/components/{comp['name']}.tsx",
                    "purpose": comp.get("responsibility", "")
                })
            else:
                files.append({
                    "path": f"src/services/{comp['name']}.ts",
                    "purpose": comp.get("responsibility", "")
                })
        
        return files
