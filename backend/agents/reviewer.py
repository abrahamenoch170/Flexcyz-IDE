"""
Reviewer Agent - Code review, quality assurance, and bug detection
Uses multiple review strategies: static analysis, LLM review, test execution
"""

import re
from typing import Any, Dict, List

from agents.base import AgentCapability, BaseAgent
from models.schemas import CodeFile, CodeGenerationResult, CodeIssue, ReviewResult


class ReviewerAgent(BaseAgent):
    """
    Multi-layer code review system
    """
    
    def __init__(self, model: str = "deepseek-r1"):
        super().__init__(
            name="Reviewer",
            capabilities=[AgentCapability.REVIEWING],
            preferred_model=model,
            temperature=0.1,
            system_prompt=self._load_system_prompt()
        )
        
        self.severity_patterns = {
            "error": [
                r"TODO|FIXME|XXX|HACK",
                r"eval\(|exec\(",
                r"password\s*=\s*['\"]",
                r"SELECT\s+.*\s+FROM.*\+",
                r"innerHTML\s*=",
            ],
            "warning": [
                r"console\.log\(",
                r"any\)",
                r"except\s*:",
                r"var\s+\w+",
            ]
        }
    
    def _load_system_prompt(self) -> str:
        return """You are Flexcyz's Code Reviewer Agent. Analyze code for:
1. Bugs and logic errors
2. Security vulnerabilities
3. Performance issues
4. Code style violations
5. Missing error handling
6. Type safety issues

Be thorough but constructive. For each issue found, provide:
- Severity (error, warning, suggestion)
- Specific location (line if possible)
- Clear explanation
- Suggested fix

Return JSON format only."""

    async def execute(self, instruction: str, context: Dict[str, Any]) -> ReviewResult:
        """Review code and return structured feedback"""
        print(f"🔍 Reviewer analyzing: {instruction[:80]}...")
        
        files = await self._get_files_to_review(instruction, context)
        all_issues = []
        
        for file in files:
            # Layer 1: Pattern-based static analysis
            pattern_issues = self._pattern_analysis(file)
            all_issues.extend(pattern_issues)
            
            # Layer 2: LLM-based deep review
            llm_issues = await self._llm_review(file, context)
            all_issues.extend(llm_issues)
        
        # Calculate score
        error_count = sum(1 for i in all_issues if i.severity == "error")
        warning_count = sum(1 for i in all_issues if i.severity == "warning")
        
        total_score = 100 - (error_count * 20) - (warning_count * 5)
        total_score = max(0, total_score)
        passed = error_count == 0 and total_score >= 70
        
        return ReviewResult(
            passed=passed,
            issues=all_issues,
            score=total_score,
            summary=f"Found {error_count} errors, {warning_count} warnings. Score: {total_score}/100"
        )
    
    async def _get_files_to_review(self, instruction: str, context: Dict) -> List[CodeFile]:
        """Get files from context or recent generation"""
        coder_output = context.get("previous_outputs", {}).get("coder")
        
        if coder_output and isinstance(coder_output, CodeGenerationResult):
            return coder_output.files
        
        project_id = context.get("project_id")
        if project_id:
            from core.state_manager import state_manager
            file_states = state_manager.list_files(project_id)
            return [
                CodeFile(path=f.path, content=f.content, language=f.language or "text")
                for f in file_states
            ]
        
        return []
    
    def _pattern_analysis(self, file: CodeFile) -> List[CodeIssue]:
        """Fast regex-based code analysis"""
        issues = []
        content = file.content
        
        for severity, patterns in self.severity_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    issues.append(CodeIssue(
                        severity=severity,
                        file_path=file.path,
                        line_number=line_num,
                        message=f"Pattern match: {pattern}",
                        suggested_fix=self._get_fix_suggestion(pattern, file.language)
                    ))
        
        return issues
    
    async def _llm_review(self, file: CodeFile, context: Dict) -> List[CodeIssue]:
        """Deep LLM-based code review"""
        tech_stack = context.get("tech_stack", [])
        
        # Build prompt with proper JSON format request
        prompt = f"""Review this code file:

FILE: {file.path}
LANGUAGE: {file.language}
TECH STACK: {', '.join(tech_stack)}

CODE:
```{file.language}
{file.content[:4000]}
Provide review as JSON with this exact structure: {{  "issues": [
        {{
            "severity": "error|warning|suggestion",
            "line_number": 42,
            "message": "Clear description of issue",
            "suggested_fix": "Code or explanation of fix"
        }}
    ],  "strengths": ["What's done well"],  "overall_assessment": "Brief summary" }}"""
    messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    try:
        result = await self.llm_json(messages)
        issues = []
        
        for issue_data in result.get("issues", []):
            issues.append(CodeIssue(
                severity=issue_data.get("severity", "warning"),
                file_path=file.path,
                line_number=issue_data.get("line_number"),
                message=issue_data.get("message", "Unknown issue"),
                suggested_fix=issue_data.get("suggested_fix")
            ))
        
        return issues
        
    except Exception as e:
        print(f"LLM review failed for {file.path}: {e}")
        return []

def _get_fix_suggestion(self, pattern: str, language: str) -> str:
    """Get fix suggestion for common patterns"""
    suggestions = {
        r"TODO|FIXME": "Complete or remove before production",
        r"eval\(": "Use json.loads() or safer alternatives",
        r"password\s*=": "Use environment variables for secrets",
        r"console\.log": "Remove debug logging",
        r"any\)": "Add proper TypeScript types",
    }
    
    for key, suggestion in suggestions.items():
        if re.search(key, pattern):
            return suggestion
    
    return "Review and fix manually"