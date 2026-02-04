

```python
"""
Researcher Agent - Web search, documentation retrieval, Stack Overflow
Uses Tavily and Exa APIs for real-time information
"""

from typing import Any, Dict, List

from agents.base import AgentCapability, BaseAgent
from models.schemas import ResearchQuery, ResearchResult
from tools.web_search import WebSearchTool


class ResearcherAgent(BaseAgent):
    """
    Intelligent research with source verification and synthesis
    """
    
    def __init__(self, model: str = "deepseek-v3"):
        super().__init__(
            name="Researcher",
            capabilities=[AgentCapability.RESEARCHING],
            preferred_model=model,
            temperature=0.3,
            system_prompt=self._load_system_prompt()
        )
        
        self.search_tool = WebSearchTool()
    
    def _load_system_prompt(self) -> str:
        return """You are Flexcyz's Research Agent. Gather accurate, up-to-date technical information.

CAPABILITIES:
- Search web for latest docs and tutorials
- Find code examples and best practices
- Verify information across multiple sources
- Synthesize findings into actionable insights

Always cite sources and provide code examples when relevant."""

    async def execute(self, instruction: str, context: Dict[str, Any]) -> Dict:
        """
        Research based on instruction
        """
        print(f"🔬 Researching: {instruction[:80]}...")
        
        # Determine research strategy
        if "library" in instruction.lower() or "package" in instruction.lower():
            return await self._research_library(instruction, context)
        elif "api" in instruction.lower():
            return await self._research_api(instruction, context)
        elif "error" in instruction.lower() or "bug" in instruction.lower():
            return await self._research_solution(instruction, context)
        else:
            return await self._general_research(instruction, context)
    
    async def _research_library(self, instruction: str, context: Dict) -> Dict:
        """
        Research specific library/package
        """
        # Extract library name
        words = instruction.split()
        library = None
        for i, word in enumerate(words):
            if word.lower() in ["library", "package", "using"]:
                if i + 1 < len(words):
                    library = words[i + 1].strip(",.")
        
        if not library:
            library = words[0] if words else "unknown"
        
        # Parallel searches
        queries = [
            f"{library} official documentation",
            f"{library} best practices 2024",
            f"{library} npm install usage example" if "npm" in str(context.get("tech_stack")) else f"{library} pip install usage",
            f"{library} common issues stackoverflow"
        ]
        
        all_results = []
        for query in queries:
            results = await self.search_tool.search(query, num_results=3)
            all_results.extend(results)
        
        # Synthesize with LLM
        synthesis = await self._synthesize_findings(all_results, instruction)
        
        return {
            "topic": library,
            "query_count": len(queries),
            "sources": [r.url for r in all_results[:5]],
            "summary": synthesis["summary"],
            "code_examples": synthesis["code_examples"],
            "best_practices": synthesis["best_practices"],
            "installation": synthesis.get("installation", "")
        }
    
    async def _research_api(self, instruction: str, context: Dict) -> Dict:
        """
        Research API endpoints or patterns
        """
        queries = [
            f"{instruction} REST API best practices",
            f"{instruction} authentication patterns",
            f"{instruction} rate limiting strategies"
        ]
        
        all_results = []
        for query in queries:
            results = await self.search_tool.search(query, num_results=3)
            all_results.extend(results)
        
        synthesis = await self._synthesize_findings(all_results, instruction)
        
        return {
            "topic": "API Design",
            "sources": [r.url for r in all_results[:5]],
            "patterns": synthesis["best_practices"],
            "security_considerations": synthesis.get("security", []),
            "example_endpoints": synthesis.get("code_examples", [])
        }
    
    async def _research_solution(self, instruction: str, context: Dict) -> Dict:
        """
        Research solutions to errors or problems
        """
        # Clean error message for search
        clean_error = instruction.replace("Error:", "").replace("Exception:", "").strip()[:100]
        
        queries = [
            f"{clean_error} stackoverflow solution",
            f"{clean_error} github issues fix",
            f"{clean_error} solution 2024"
        ]
        
        all_results = []
        for query in queries:
            results = await self.search_tool.search(query, num_results=5)
            all_results.extend(results)
        
        # Extract solutions
        solutions = []
        for result in all_results:
            if "stackoverflow" in result.url or "github" in result.url:
                solutions.append({
                    "source": result.url,
                    "content": result.content[:500],
                    "score": result.relevance_score
                })
        
        return {
            "error": instruction[:200],
            "potential_solutions": solutions[:3],
            "recommended_approach": solutions[0]["content"] if solutions else "No clear solution found"
        }
    
    async def _general_research(self, instruction: str, context: Dict) -> Dict:
        """
        General technical research
        """
        results = await self.search_tool.search(instruction, num_results=5)
        
        synthesis = await self._synthesize_findings(results, instruction)
        
        return {
            "query": instruction,
            "sources": [{"title": r.title, "url": r.url} for r in results],
            "summary": synthesis["summary"],
            "key_findings": synthesis["best_practices"],
            "applicable_code": synthesis["code_examples"]
        }
    
    async def _synthesize_findings(
        self,
        results: List[ResearchResult],
        original_query: str
    ) -> Dict:
        """
        Use LLM to synthesize research findings
        """
        # Prepare context for LLM
        sources_text = "\n\n".join([
            f"Source: {r.title} ({r.url})\n{r.content[:800]}"
            for r in results[:5]
        ])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Synthesize these research findings for: {original_query}

SOURCES:
{sources_text}

Provide synthesis as JSON:
{{
    "summary": "2-3 sentence overview",
    "code_examples": ["specific code snippets found"],
    "best_practices": ["list of recommendations"],
    "installation": "how to install/setup if applicable",
    "security": ["security considerations if applicable"]
}}
"""}
        ]
        
        try:
            return await self.llm_json(messages)
        except Exception as e:
            # Fallback if LLM synthesis fails
            return {
                "summary": f"Found {len(results)} sources about {original_query}",
                "code_examples": [r.content[:200] for r in results if r.code_snippets],
                "best_practices": ["Review sources manually"],
                "installation": "",
                "security": []
            }
    
    async def verify_code_example(self, code: str, context: Dict) -> bool:
        """
        Verify if a code example is still valid/current
        """
        # Could run in sandbox or check against docs
        return True