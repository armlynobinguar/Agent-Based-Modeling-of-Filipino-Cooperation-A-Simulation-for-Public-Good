from typing import List, Dict
from ..agents.agent import Agent

class PGGManager:
    def __init__(self, 
                 agents: List[Agent],
                 multiplier: float = 1.6,
                 rounds: int = 10):
        self.agents = agents
        self.multiplier = multiplier
        self.rounds = rounds
        self.current_round = 0
        self.history = []
    
    def play_round(self) -> Dict:
        """Execute one round of the Public Goods Game."""
        # Collect contributions
        contributions = {}
        for agent in self.agents:
            game_context = self._create_game_context()
            contribution = agent.decide_contribution(game_context)
            contributions[agent.name] = contribution
        
        # Calculate payouts
        total_contribution = sum(contributions.values())
        payout = (total_contribution * self.multiplier) / len(self.agents)
        
        # Update agent states and record history
        round_data = {
            'round': self.current_round,
            'contributions': contributions,
            'total_contribution': total_contribution,
            'payout': payout
        }
        
        self.history.append(round_data)
        self.current_round += 1
        
        return round_data
    
    def _create_game_context(self) -> Dict:
        """Create context information for agents to make decisions."""
        return {
            'round': self.current_round,
            'total_rounds': self.rounds,
            'multiplier': self.multiplier,
            'other_agents': [
                {
                    'name': agent.name,
                    'public_bio': agent.state.public_bio
                }
                for agent in self.agents
            ]
        } 