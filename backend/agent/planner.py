import os
from typing import List, Dict, Any
from pydantic import BaseModel
import openai # Added for future integration

class PlanStep(BaseModel):
    id: int
    action: str
    description: str
    dependencies: List[int] = []

class Planner:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key

    async def generate_plan(self, query: str) -> List[PlanStep]:
        # If API key exists, we could use an LLM to generate the plan
        # For now, we use a sophisticated heuristic-based template
        
        # Heuristic: if query contains 'why' or 'cause', focus on causal analysis
        # if query contains 'when' or 'time', focus on temporal reasoning
        
        is_causal = any(word in query.lower() for word in ["why", "cause", "reason", "impact"])
        is_temporal = any(word in query.lower() for word in ["when", "time", "history", "sequence", "before", "after"])
        
        steps = [
            PlanStep(id=1, action="RETRIEVE_CONTEXT", description=f"Extracting graph nodes related to '{query}'"),
        ]
        
        current_deps = [1]
        
        if is_causal:
            steps.append(PlanStep(id=2, action="CAUSAL_ANALYSIS", description="Mapping directional influence and root causes", dependencies=current_deps))
            current_deps = [2]
            
        if is_temporal:
            steps.append(PlanStep(id=3, action="TEMPORAL_REASONING", description="Sequencing events along the historical timeline", dependencies=[1]))
            current_deps.append(3)
            
        # Ensure we always have these if they weren't added by heuristics to keep the demo consistent
        if not is_causal:
             steps.append(PlanStep(id=2, action="CAUSAL_ANALYSIS", description="Analyzing general structural causality", dependencies=[1]))
        if not is_temporal:
             steps.append(PlanStep(id=3, action="TEMPORAL_REASONING", description="Reviewing event sequence", dependencies=[1]))

        steps.append(PlanStep(id=4, action="SYNTHESIZE", description="Aggregating intelligence into a high-confidence answer", dependencies=[2, 3]))
        
        return steps
