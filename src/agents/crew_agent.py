from typing import Dict, List
from .agent import AgentState, PrimingType, Agent

class CrewAgent(Agent):
    def __init__(self, name: str, role: str, state: AgentState):
        super().__init__(name, state)
        self.role = role
        self.communication_system = CommunicationSystem()
        self.social_network = SocialNetwork()
    
    def decide_contribution(self, game_context: Dict) -> float:
        # Get peer influence
        peer_influence = self.social_network.get_peer_influence(self.name)
        
        # Process any messages
        if 'messages' in game_context:
            for message in game_context['messages']:
                self.communication_system.process_message(
                    message['content'],
                    message['sender'],
                    message['channel']
                )
        
        # Calculate base contribution
        base_contribution = self.state.endowment * self._get_contribution_factor()
        
        # Apply peer influence
        contribution = base_contribution * peer_influence
        
        return min(max(0, contribution), self.state.endowment)
    
    def _get_contribution_factor(self) -> float:
        # Tax rates based on income brackets
        if self.state.endowment >= 80000:
            return 0.15  # 15% tax rate
        elif self.state.endowment >= 40000:
            return 0.12  # 12% tax rate
        elif self.state.endowment >= 20000:
            return 0.10  # 10% tax rate
        elif self.state.endowment >= 10000:
            return 0.09  # 9% tax rate
        else:
            return 0.08  # 8% tax rate

class CommunicationSystem:
    def __init__(self):
        self.channels = {
            'formal': [],  # Official communications
            'informal': [],  # Peer discussions
            'feedback': []   # System feedback
        }
        self.influence_factors = {
            'authority': 1.0,
            'trust': 1.0,
            'relevance': 1.0
        }
    
    def process_message(self, message: str, sender: str, channel: str):
        # Process and weight messages based on various factors
        weighted_message = self._apply_influence_factors(message, sender)
        self.channels[channel].append(weighted_message)
    
    def _apply_influence_factors(self, message: str, sender: str) -> str:
        # Apply influence factors to message
        weighted_message = message
        for factor, weight in self.influence_factors.items():
            if factor == 'authority' and 'Government' in sender:
                weighted_message = f"[AUTHORITY] {weighted_message}"
            elif factor == 'trust' and 'Business' in sender:
                weighted_message = f"[TRUSTED] {weighted_message}"
        return weighted_message

class SocialNetwork:
    def __init__(self):
        self.connections = {}
        self.influence_weights = {}
    
    def add_connection(self, agent1: str, agent2: str, weight: float):
        if agent1 not in self.connections:
            self.connections[agent1] = {}
        self.connections[agent1][agent2] = weight
    
    def get_peer_influence(self, agent: str) -> float:
        # Calculate peer influence on tax compliance
        if agent not in self.connections:
            return 1.0
        
        total_influence = sum(self.connections[agent].values())
        return total_influence / len(self.connections[agent]) 