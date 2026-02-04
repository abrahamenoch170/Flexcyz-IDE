"""
Planner Agent - Expands vague prompts into detailed execution plans
Uses multi-step reasoning to break down complex projects
"""

from typing import Any, Dict, List

from agents.base import AgentCapability, BaseAgent
from models.schemas import AgentType, PlanStep, ProjectPlan


class PlannerAgent(BaseAgent):
    """
    Sophisticated planning with dependency resolution and tech stack detection
    """
    
    def __init__(self, model: str = "Qwen2.5-Coder-7B-Instruct"):
        super().__init__(
            name="Planner",
            capabilities=[AgentCapability.PLANNING],
            preferred_model=model,
            temperature=0.3,
            system_prompt=self._load_system_prompt()
        )
        
        self.tech_patterns = {
            "react": ["React", "Next.js", "TypeScript", "Vite"],
            "vue": ["Vue.js", "Nuxt", "TypeScript"],
            "angular": ["Angular", "TypeScript"],
            "svelte": ["Svelte", "SvelteKit"],
            "node": ["Node.js", "Express", "Fastify"],
            "python": ["Python", "FastAPI", "Django", "Flask"],
            "go": ["Go", "Gin", "Echo"],
            "rust": ["Rust", "Actix", "Axum"],
            "database": ["PostgreSQL", "MongoDB", "Redis", "Prisma"],
            "auth": ["JWT", "OAuth2", "Clerk", "Auth0"],
            "styling": ["Tailwind CSS", "Styled Components", "CSS Modules"],
            "testing": ["Jest", "Vitest", "Pytest", "Playwright"],
            "deployment": ["Docker", "Vercel", "Railway", "AWS"],
        }
    
    def _load_system_prompt(self) -> str:
        return """You are Flexcyz's Planner Agent. Your job is to analyze user prompts and create detailed, actionable execution plans.

CRITICAL RULES:
1. ALWAYS analyze the prompt for implied tech stack (React, Vue, Python, etc.)
2. Break into logical steps with clear dependencies
3. Include setup, development, and deployment phases
4. Estimate complexity and order appropriately
5. Return ONLY valid JSON matching the schema exactly

Your output determines how 6+ specialized agents will collaborate. Be thorough."""

    async def execute(self, instruction: str, context: Dict[str, Any]) -> ProjectPlan:
        """
        Main entry: prompt -> detailed plan
        """
        print(f"🎯 Planner analyzing: {instruction[:80]}...")
        
        # Step 1: Analyze requirements and detect tech stack
        analysis = await self._analyze_requirements(instruction, context)
        
        # Step 2: Generate execution steps
        steps = await self._generate_steps(instruction, analysis, context)
        
        # Step 3: Resolve dependencies and optimize order
        optimized_steps = self._optimize_step_order(steps)
        
        plan = ProjectPlan(
            summary=analysis["summary"],
            steps=optimized_steps,
            tech_stack=analysis["tech_stack"],
            estimated_duration=analysis.get("estimated_duration")
        )
        
        print(f"📋 Plan created: {len(plan.steps)} steps, {len(plan.tech_stack)} technologies")
        return plan
    
    async def _analyze_requirements(
        self, 
        instruction: str, 
        context: Dict
    ) -> Dict:
        """
        Deep analysis of what needs to be built
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Analyze this project request and extract structured information:

PROMPT: {instruction}

Analyze:
1. What is the core product/feature?
2. What technologies are implied or mentioned?
3. What is the complexity level (1-10)?
4. What are the main components (frontend, backend, database, auth, etc.)?
5. Estimated time to build?

Respond with JSON:
{{
    "summary": "Brief description of what to build",
    "core_features": ["feature1", "feature2"],
    "tech_stack": ["detected", "technologies"],
    "components": ["frontend", "backend", "database"],
    "complexity": 5,
    "estimated_duration": "2-3 hours",
    "special_considerations": ["SEO needed", "real-time updates", etc]
}}
"""}
        ]
        
        result = await self.llm_json(messages)
        
        # Ensure tech_stack is present
        if "tech_stack" not in result or not result["tech_stack"]:
            result["tech_stack"] = self._detect_tech_stack(instruction)
        
        return result
    
    def _detect_tech_stack(self, instruction: str) -> List[str]:
        """Pattern matching for tech stack detection"""
        instruction_lower = instruction.lower()
        detected = []
        
        for category, techs in self.tech_patterns.items():
            if any(tech.lower() in instruction_lower for tech in techs):
                detected.extend(techs)
        
        # Defaults if nothing detected
        if not detected:
            detected = ["React", "TypeScript", "Node.js", "Express", "PostgreSQL"]
        
        return list(set(detected))  # Remove duplicates
    
    async def _generate_steps(
        self,
        instruction: str,
        analysis: Dict,
        context: Dict
    ) -> List[PlanStep]:
        """
        Generate specific execution steps
        """
        tech_stack = analysis.get("tech_stack", [])
        components = analysis.get("components", [])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Create a detailed execution plan for this project.

PROJECT: {instruction}
SUMMARY: {analysis['summary']}
TECH STACK: {', '.join(tech_stack)}
COMPONENTS: {', '.join(components)}

Generate steps covering:
1. Project setup and initialization
2. Architecture/design decisions
3. Core feature development (split into logical chunks)
4. Integration and testing
5. Deployment preparation

For each step specify:
- Which agent should handle it (planner, architect, coder, reviewer, deployer)
- What files/components it will produce
- Dependencies on previous steps

Respond with JSON:
{{
    "steps": [
        {{
            "description": "Specific action to take",
            "agent": "architect|coder|reviewer|deployer",
            "dependencies": [],
            "priority": 5,
            "expected_files": ["src/components/Example.tsx"]
        }}
    ]
}}
"""}
        ]
        
        result = await self.llm_json(messages)
        raw_steps = result.get("steps", [])
        
        # Convert to PlanStep objects
        steps = []
        for i, step_data in enumerate(raw_steps):
            # Map string agent names to enum
            agent_str = step_data.get("agent", "coder")
            try:
                agent_type = AgentType(agent_str)
            except ValueError:
                agent_type = AgentType.CODER  # Default
            
            step = PlanStep(
                description=step_data["description"],
                agent_type=agent_type,
                dependencies=step_data.get("dependencies", []),
                priority=step_data.get("priority", 3),
                expected_files=step_data.get("expected_files", [])
            )
            steps.append(step)
        
        return steps
    
    def _optimize_step_order(self, steps: List[PlanStep]) -> List[PlanStep]:
        """
        Ensure logical ordering and add implicit dependencies
        """
        # Ensure setup comes first
        for i, step in enumerate(steps):
            if any(word in step.description.lower() for word in ["setup", "init", "create project"]):
                step.priority = 10
                step.dependencies = []
        
        # Architect should come before coder
        architect_indices = [i for i, s in enumerate(steps) if s.agent_type == AgentType.ARCHITECT]
        coder_indices = [i for i, s in enumerate(steps) if s.agent_type == AgentType.CODER]
        
        for c_idx in coder_indices:
            # Add dependency on last architect step if exists
            if architect_indices and not steps[c_idx].dependencies:
                last_arch = architect_indices[-1]
                if last_arch < c_idx:
                    steps[c_idx].dependencies.append(steps[last_arch].id)
        
        # Reviewer comes after coder
        reviewer_indices = [i for i, s in enumerate(steps) if s.agent_type == AgentType.REVIEWER]
        for r_idx in reviewer_indices:
            # Find nearest preceding coder step
            preceding_coders = [i for i in coder_indices if i < r_idx]
            if preceding_coders and not steps[r_idx].dependencies:
                steps[r_idx].dependencies.append(steps[preceding_coders[-1]].id)
        
        # Sort by priority (descending) then by index
        indexed_steps = list(enumerate(steps))
        indexed_steps.sort(key=lambda x: (-x[1].priority, x[0]))
        
        return [s for _, s in indexed_steps]
    
    async def refine_plan(
        self,
        original_plan: ProjectPlan,
        feedback: str,
        context: Dict
    ) -> ProjectPlan:
        """
        Refine plan based on user feedback or execution results
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
Refine this project plan based on feedback.

ORIGINAL PLAN:
{original_plan.summary}
Steps: {len(original_plan.steps)}

FEEDBACK: {feedback}

Provide updated plan with necessary adjustments.
"""}
        ]
        
        result = await self.llm_json(messages)
        # Reuse generation logic...
        return await self._generate_steps(feedback, result, context)
