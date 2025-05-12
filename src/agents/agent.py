from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class PrimingType(Enum):
    TEAMWORK = "TEAMWORK"
    TAXATION = "TAXATION"
    COMMUNITY = "COMMUNITY"
    BASELINE = "BASELINE"
    TAX_8 = "TAX_8"    # 8% tax rate
    TAX_9 = "TAX_9"    # 9% tax rate
    TAX_10 = "TAX_10"  # 10% tax rate
    TAX_12 = "TAX_12"  # 12% tax rate
    TAX_15 = "TAX_15"  # 15% tax rate

class AgentState:
    def __init__(self, public_bio: str, private_bio: str, endowment: float, priming: PrimingType):
        self.public_bio = public_bio
        self.private_bio = private_bio
        self.endowment = endowment
        self.contribution = 0.0
        self.memory = []
        self.priming = priming
        self.total_earnings = 0.0
        self.punishment_points = 0
        self.reward_points = 0

class TaxRate:
    def __init__(self):
        self.base_rates = {
            'low': 0.08,
            'medium': 0.10,
            'high': 0.15
        }
        self.adjustments = {
            'inflation': 0.0,
            'economic_growth': 0.0,
            'special_circumstances': 0.0
        }
    
    def get_adjusted_rate(self, income_level: str) -> float:
        base_rate = self.base_rates[income_level]
        total_adjustment = sum(self.adjustments.values())
        return base_rate * (1 + total_adjustment)

class Agent:
    def __init__(self, name: str, state: AgentState):
        self.name = name
        self.state = state
        self.chat_history = []
    
    def decide_contribution(self, game_context: Dict) -> float:
        """Decide how much to contribute based on game context and agent state."""
        raise NotImplementedError
    
    def update_memory(self, event: Dict):
        """Add an event to the agent's memory."""
        self.state.memory.append(event)
    
    def add_to_chat(self, message: str):
        """Add a message to the agent's chat history."""
        self.chat_history.append(message)
    
    def get_chat_history(self) -> List[str]:
        """Get the agent's chat history."""
        return self.chat_history
    
    def communicate(self, current_contributions: Dict) -> str:
        raise NotImplementedError 