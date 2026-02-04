"""
Web Search Tool - Tavily and Exa AI integration
Real-time research capabilities
"""

import os
from typing import List, Optional

import httpx

from models.schemas import ResearchResult


class WebSearchTool:
    """
    Unified search interface supporting multiple providers
    """
    
    def __init__(self):
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.exa_key = os.getenv("EXA_API_KEY")
        
        # Prefer Tavily for technical docs, Exa for code
        self.default_provider = "tavily" if self.tavily_key else ("exa" if self.exa_key else None)
    
    async def search(
        self, 
        query: str, 
        num_results: int = 5,
        provider: Optional[str] = None
    ) -> List[ResearchResult]:
        """
        Search using specified or default provider
        """
        prov = provider or self.default_provider
        
        if prov == "tavily":
            return await self._tavily_search(query, num_results)
        elif prov == "exa":
            return await self._exa_search(query, num_results)
        else:
            # Fallback to DuckDuckGo or similar free option
            return await self._fallback_search(query, num_results)
    
    async def _tavily_search(self, query: str, n: int) -> List[ResearchResult]:
        """Tavily AI - optimized for technical content"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_images": False,
                    "max_results": n
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Tavily error: {response.text}")
            
            data = response.json()
            results = []
            
            for r in data.get("results", []):
                result = ResearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    source="tavily",
                    relevance_score=r.get("score", 0),
                    code_snippets=self._extract_code_snippets(r.get("content", ""))
                )
                results.append(result)
            
            return results
    
    async def _exa_search(self, query: str, n: int) -> List[ResearchResult]:
        """Exa AI - neural search for code and technical content"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.exa.ai/search",
                headers={"Authorization": f"Bearer {self.exa_key}"},
                json={
                    "query": query,
                    "num_results": n,
                    "type": "neural",
                    "contents": {
                        "text": True,
                        "highlights": True
                    },
                    "use_autoprompt": True
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Exa error: {response.text}")
            
            data = response.json()
            results = []
            
            for r in data.get("results", []):
                result = ResearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("text", ""),
                    source="exa",
                    relevance_score=r.get("score", 0),
                    code_snippets=self._extract_code_snippets(r.get("text", ""))
                )
                results.append(result)
            
            return results
    
    async def _fallback_search(self, query: str, n: int) -> List[ResearchResult]:
        """
        Fallback search using free alternative
        Could use DuckDuckGo, Bing API, etc.
        """
        # Placeholder - implement if needed
        return []
    
    def _extract_code_snippets(self, content: str) -> List[str]:
        """Extract code blocks from content"""
        import re
        
        # Match code blocks
        patterns = [
            r"```[\w]*\n(.*?)```",  # Markdown code blocks
            r"<code>(.*?)</code>",   # HTML code tags
        ]
        
        snippets = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            snippets.extend(matches)
        
        # Also extract indented blocks that look like code
        lines = content.split("\n")
        current_block = []
        
        for line in lines:
            if line.startswith("    ") or line.startswith("\t"):
                current_block.append(line.strip())
            else:
                if len(current_block) > 2:  # Minimum 3 lines
                    snippets.append("\n".join(current_block))
                current_block = []
        
        return snippets[:5]  # Limit snippets
    
    async def get_page_content(self, url: str) -> str:
        """
        Fetch and extract content from a specific URL
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            
            if response.status_code != 200:
                return ""
            
            # Simple text extraction (could use BeautifulSoup)
            content = response.text
            
            # Remove scripts and styles
            import re
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
            
            # Extract text
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content)
            
            return content[:5000]  # Limit length
