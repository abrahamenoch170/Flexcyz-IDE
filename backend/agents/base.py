"""
Base agent class with common functionality
All agents inherit from this
"""

import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml


class AgentCapability(str, Enum):
    PLANNING = "planning"
    CODING = "coding"
    REVIEWING = "reviewing"
    RESEARCHING = "researching"
    DEPLOYING = "deploying"


class BaseAgent(ABC):
    """
    Abstract base class for all Flexcyz agents
    Provides LLM routing, memory access, and tool usage
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        preferred_model: Optional[str] = None,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.capabilities = capabilities
        self.preferred_model = preferred_model
        self.temperature = temperature
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        # Lazy initialization
        self._llm_router = None
        self._tools = {}
        
    @property
    def llm(self):
        """Lazy load LLM router"""
        if self._llm_router is None:
            from tools.llm_router import LLMRouter
            self._llm_router = LLMRouter()
        return self._llm_router
    
    def _default_system_prompt(self) -> str:
        return f"You are {self.name}, an AI agent in the Flexcyz autonomous IDE."
    
    def load_prompt_template(self, template_name: str) -> str:
        """Load prompt template from prompts/ directory"""
        template_path = f"prompts/{template_name}.yaml"
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("prompt", "")
        return ""
    
    async def llm_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        json_mode: bool = False
    ) -> str:
        """
        Complete with LLM, with error handling and retries
        """
        model = model or self.preferred_model
        temp = temperature or self.temperature
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await self.llm.complete(
                    messages=messages,
                    model=model,
                    temperature=temp,
                    max_tokens=max_tokens,
                    json_mode=json_mode
                )
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                import asyncio
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def llm_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        Get JSON response from LLM
        """
        response = await self.llm_complete(
            messages=messages,
            model=model,
            temperature=temperature,
            json_mode=True
        )
        
        # Clean up response (sometimes LLM adds markdown)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        return json.loads(response.strip())
    
    def get_relevant_context(self, context: Dict, max_tokens: int = 2000) -> str:
        """
        Extract most relevant context for token limits
        """
        relevant = []
        
        # Priority order
        if "prompt" in context:
            relevant.append(f"Task: {context['prompt']}")
        
        if "tech_stack" in context:
            relevant.append(f"Tech Stack: {', '.join(context['tech_stack'])}")
        
        if "project_files" in context:
            files = context["project_files"]
            relevant.append(f"Existing Files: {', '.join(files[:10])}")
            if len(files) > 10:
                relevant.append(f"... and {len(files) - 10} more")
        
        if "previous_outputs" in context:
            # Include summaries of previous agent outputs
            for agent, output in context["previous_outputs"].items():
                if output:
                    relevant.append(f"{agent} output: {str(output)[:200]}...")
        
        return "\n".join(relevant)
    
    @abstractmethod
    async def execute(self, instruction: str, context: Dict[str, Any]) -> Any:
        """
        Main execution method - must be implemented by subclasses
        """
        pass
    
    def validate_output(self, output: Any, schema: Dict) -> tuple[bool, str]:
        """
        Validate agent output against expected schema
        """
        try:
            if schema.get("type") == "object":
                if not isinstance(output, dict):
                    return False, "Output must be an object"
                
                required = schema.get("required", [])
                for field in required:
                    if field not in output:
                        return False, f"Missing required field: {field}"
            
            return True, ""
        except Exception as e:
            return False, str(e)
    
    async def chain_of_thought(
        self,
        problem: str,
        steps: List[str],
        context: Dict
    ) -> Dict:
        """
        Execute chain-of-thought reasoning
        """
        thoughts = []
        
        for i, step in enumerate(steps, 1):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""
Problem: {problem}

Previous thoughts: {chr(10).join(thoughts) if thoughts else "None"}

Step {i}: {step}

Context: {self.get_relevant_context(context)}
"""}
            ]
            
            response = await self.llm_complete(messages)
            thoughts.append(f"Step {i}: {response}")
        
        return {
            "thoughts": thoughts,
            "final_answer": thoughts[-1] if thoughts else ""
        }
    
    def log_execution(
        self,
        task: str,
        result: Any,
        duration: float,
        tokens_used: int = 0
    ):
        """Log execution metrics"""
        print(f"[{self.name}] Executed in {duration:.2f}s | Tokens: {tokens_used}")
    
    def __repr__(self):
        return f"<{self.name}Agent capabilities={self.capabilities}>"
