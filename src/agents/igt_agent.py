from typing import Dict, List
import random
import numpy as np
from .agent import Agent, AgentState, PrimingType

class IGTAgent(Agent):
    def __init__(self, name: str, role: str, state: AgentState):
        super().__init__(name, state)
        self.role = role
        
        # Required attributes for compatibility
        self.compliance_history = [1.0]  # Initialize with default compliance rate
        self.chat_history = []
        self.trust_in_government = random.uniform(0.5, 1.5)
        self.social_pressure_sensitivity = random.uniform(0.5, 1.5)
        self.behavioral_model = {
            'contribution_range': (0.8, 1.2),
            'fatigue_sensitivity': 1.0,
            'reward_sensitivity': 1.0
        }
        self.fatigue_factor = 1.0
        self.budget_constraint = 1.0
        
        # IGT-specific attributes
        self.deck_preferences = {
            'A': {'wins': 100, 'losses': -250, 'probability': 0.5},  # Risky deck
            'B': {'wins': 100, 'losses': -1250, 'probability': 0.1},  # Very risky deck
            'C': {'wins': 50, 'losses': -50, 'probability': 0.5},    # Safe deck
            'D': {'wins': 50, 'losses': -250, 'probability': 0.1}     # Moderately safe deck
        }
        self.deck_history = {deck: [] for deck in self.deck_preferences.keys()}
        self.expectancy = {deck: 0.0 for deck in self.deck_preferences.keys()}
        self.learning_rate = 0.1
        self.risk_tolerance = random.uniform(0.5, 1.5)
        self.memory_window = 10
    
    def add_to_chat(self, message: str):
        """Add message to chat history."""
        self.chat_history.append(message)
    
    def update_compliance_history(self, compliance_rate: float):
        """Update the agent's compliance history."""
        self.compliance_history.append(compliance_rate)
    
    def communicate(self, contributions: Dict) -> str:
        """Generate communication based on IGT model."""
        if not self.deck_history['A']:  # If no history yet
            return f"{self.name}: I'm still learning about the best strategies."
        
        # Calculate average outcome
        total_outcomes = sum(sum(outcomes) for outcomes in self.deck_history.values())
        total_trials = sum(len(outcomes) for outcomes in self.deck_history.values())
        avg_outcome = total_outcomes / total_trials if total_trials > 0 else 0
        
        if avg_outcome > 0:
            return f"{self.name}: My risk-taking strategy is paying off!"
        else:
            return f"{self.name}: I might need to be more conservative."
    
    def _get_contribution_factor(self) -> float:
        """Get base contribution factor based on priming."""
        try:
            if isinstance(self.state.priming, PrimingType):
                return float(self.state.priming.value) / 100.0
            elif isinstance(self.state.priming, str):
                # Try to extract number from string (e.g., "TAX_15" -> 15)
                parts = self.state.priming.split('_')
                if len(parts) > 1:
                    tax_rate = float(parts[1])
                    return tax_rate / 100.0
            elif hasattr(self.state.priming, 'value'):
                # Handle case where priming is an object with a value attribute
                return float(self.state.priming.value) / 100.0
        except (AttributeError, ValueError, TypeError):
            pass
        
        # Default to 10% if any parsing fails
        return 0.1
    
    def decide_contribution(self, game_context: Dict) -> float:
        """Make contribution decision based on IGT model."""
        # Update deck expectancies based on history
        self._update_expectancies()
        
        # Calculate deck preferences
        deck_scores = self._calculate_deck_scores()
        
        # Select deck based on scores and risk tolerance
        selected_deck = self._select_deck(deck_scores)
        
        # Get outcome from selected deck
        outcome = self._get_deck_outcome(selected_deck)
        
        # Update deck history
        self.deck_history[selected_deck].append(outcome)
        
        # Calculate contribution based on outcome
        base_contribution = self.state.endowment * self._get_contribution_factor()
        contribution = base_contribution * (1 + outcome/1000)  # Scale outcome to contribution
        
        # Apply behavioral model adjustments
        contribution *= self.behavioral_model['contribution_range'][0] + random.random() * (
            self.behavioral_model['contribution_range'][1] - self.behavioral_model['contribution_range'][0]
        )
        
        # Apply fatigue factor
        contribution *= self.fatigue_factor
        
        # Apply budget constraint
        contribution *= self.budget_constraint
        
        # Ensure contribution stays within reasonable bounds
        contribution = max(0, min(contribution, self.state.endowment))
        
        # Update compliance history
        expected_contribution = self.state.endowment * self._get_contribution_factor()
        compliance_rate = contribution / expected_contribution if expected_contribution > 0 else 0
        self.compliance_history.append(compliance_rate)
        
        return contribution
    
    def _update_expectancies(self):
        """Update expectancies for each deck based on recent history."""
        for deck in self.deck_preferences.keys():
            recent_outcomes = self.deck_history[deck][-self.memory_window:]
            if recent_outcomes:
                # Calculate weighted average of recent outcomes
                weights = np.exp(-np.arange(len(recent_outcomes)) * 0.1)  # Exponential decay
                weights = weights / np.sum(weights)
                self.expectancy[deck] = np.sum(np.array(recent_outcomes) * weights)
    
    def _calculate_deck_scores(self) -> Dict[str, float]:
        """Calculate scores for each deck based on expectancy and risk."""
        scores = {}
        for deck, pref in self.deck_preferences.items():
            # Base score on expectancy
            score = self.expectancy[deck]
            
            # Adjust for risk tolerance
            if pref['losses'] < 0:  # If deck has potential losses
                risk_factor = abs(pref['losses']) / 1000  # Normalize risk
                if self.risk_tolerance > 1.0:
                    score *= (1 + (self.risk_tolerance - 1) * risk_factor)
                else:
                    score *= (1 - (1 - self.risk_tolerance) * risk_factor)
            
            scores[deck] = score
        
        return scores
    
    def _select_deck(self, deck_scores: Dict[str, float]) -> str:
        """Select a deck based on scores and exploration rate."""
        # Convert scores to probabilities using softmax
        scores = np.array(list(deck_scores.values()))
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Add some exploration
        exploration_rate = 0.1
        probabilities = (1 - exploration_rate) * probabilities + exploration_rate / len(probabilities)
        
        # Select deck
        return random.choices(list(deck_scores.keys()), probabilities)[0]
    
    def _get_deck_outcome(self, deck: str) -> float:
        """Get outcome from selected deck."""
        pref = self.deck_preferences[deck]
        if random.random() < pref['probability']:
            return pref['wins']
        return pref['losses']
    
    def update_behavior(self, outcome: Dict):
        """Update behavior based on outcomes."""
        # Update risk tolerance based on outcomes
        if outcome.get('success', False):
            self.risk_tolerance *= 1.05
        else:
            self.risk_tolerance *= 0.95
        
        # Update learning rate based on performance
        if len(self.deck_history['A']) > 10:
            recent_performance = sum(sum(outcomes) for outcomes in self.deck_history.values())
            if recent_performance > 0:
                self.learning_rate *= 1.1
            else:
                self.learning_rate *= 0.9
        
        # Update fatigue factor
        self.fatigue_factor *= 0.98  # Gradual fatigue
        
        # Reset fatigue if rewarded
        if outcome.get('rewarded', False):
            self.fatigue_factor = 1.0
        
        # Ensure parameters stay within reasonable bounds
        self.risk_tolerance = max(0.5, min(1.5, self.risk_tolerance))
        self.learning_rate = max(0.01, min(0.5, self.learning_rate))
        self.fatigue_factor = max(0.5, min(1.0, self.fatigue_factor))