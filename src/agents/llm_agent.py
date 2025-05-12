from openai import OpenAI
from typing import Dict, List
from .agent import Agent, AgentState, PrimingType
from ..config import OPENAI_API_KEY
import random

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

class LLMAgent(Agent):
    def __init__(self, name: str, role: str, state: AgentState, model: str = "gpt-3.5-turbo"):
        super().__init__(name, state)
        self.role = role
        self.model = model
        self.chat_history = []
        self.compliance_history = []
        self.risk_tolerance = random.uniform(0.5, 1.5)
        self.social_pressure_sensitivity = random.uniform(0.5, 1.5)
        self.trust_in_government = random.uniform(0.5, 1.5)
        self.behavioral_model = self._initialize_behavioral_model()
        self.fatigue_factor = 1.0
        self.budget_constraint = 1.0
        self.last_contribution = 0.0
        self.reward_history = []
    
    def _initialize_behavioral_model(self) -> Dict:
        """Initialize agent's behavioral model based on role and income."""
        models = {
            'risk_averse': {
                'contribution_range': (0.95, 1.05),
                'fatigue_sensitivity': 1.2,
                'reward_sensitivity': 0.8
            },
            'tax_avoidant': {
                'contribution_range': (0.7, 0.9),
                'fatigue_sensitivity': 0.8,
                'reward_sensitivity': 1.2
            },
            'rational_selfish': {
                'contribution_range': (0.8, 1.0),
                'fatigue_sensitivity': 1.0,
                'reward_sensitivity': 1.0
            },
            'community_focused': {
                'contribution_range': (0.9, 1.1),
                'fatigue_sensitivity': 1.1,
                'reward_sensitivity': 1.1
            }
        }
        
        # Assign behavioral model based on role and income
        if 'Government' in self.role or 'Religious' in self.role:
            return models['community_focused']
        elif self.state.endowment >= 80000:
            return models['rational_selfish']
        elif self.state.endowment <= 20000:
            return models['risk_averse']
        else:
            return models['tax_avoidant']
    
    def decide_contribution(self, game_context: Dict) -> float:
        # Get base factors
        factors = {
            'income_level': self._get_income_level_factor(),
            'previous_compliance': self._get_compliance_history(),
            'peer_pressure': self._get_peer_influence(),
            'economic_conditions': self._get_economic_context(),
            'personal_circumstances': self._get_personal_factors(),
            'risk_tolerance': self._get_risk_factor(),
            'trust_factor': self._get_trust_factor(),
            'social_pressure': self._get_social_pressure_factor(game_context),
            'fatigue': self._get_fatigue_factor(),
            'budget': self._get_budget_constraint()
        }
        
        # Calculate base contribution
        base_contribution = self.state.endowment * self._get_contribution_factor()
        
        # Apply factors with weights
        weighted_contribution = base_contribution
        for factor, weight in factors.items():
            weighted_contribution *= weight
        
        # Apply behavioral model constraints
        min_contribution = base_contribution * self.behavioral_model['contribution_range'][0]
        max_contribution = base_contribution * self.behavioral_model['contribution_range'][1]
        
        # Ensure contribution is within behavioral model constraints
        final_contribution = min(max(min_contribution, weighted_contribution), max_contribution)
        
        # Store contribution for next round
        self.last_contribution = final_contribution
        
        return final_contribution
    
    def _get_income_level_factor(self) -> float:
        # Return factor based on income level
        if self.state.endowment >= 80000:
            return 1.2  # High income bonus
        elif self.state.endowment >= 40000:
            return 1.1  # Middle income bonus
        else:
            return 1.0  # Base factor
    
    def _get_compliance_history(self) -> float:
        # Return factor based on previous compliance
        if not self.state.memory:
            return 1.0
        
        compliance_rates = [
            event.get('compliance_rate', 1.0)
            for event in self.state.memory
            if 'compliance_rate' in event
        ]
        
        if not compliance_rates:
            return 1.0
        
        return sum(compliance_rates) / len(compliance_rates)
    
    def _get_risk_factor(self) -> float:
        """Calculate risk factor based on agent's risk tolerance and history."""
        if not self.compliance_history:
            return 1.0
        
        # Calculate risk based on previous compliance and risk tolerance
        avg_compliance = sum(self.compliance_history) / len(self.compliance_history)
        risk_factor = 1.0 + (self.risk_tolerance - 1.0) * (1.0 - avg_compliance)
        
        # Add some randomness
        return risk_factor * random.uniform(0.9, 1.1)
    
    def _get_trust_factor(self) -> float:
        """Calculate trust factor based on agent's trust in government."""
        # Trust affects compliance
        trust_factor = 0.8 + (self.trust_in_government * 0.4)  # Range: 0.8-1.2
        
        # Trust can change based on recent events
        if self.state.memory:
            recent_events = self.state.memory[-3:]  # Look at last 3 events
            for event in recent_events:
                if event.get('type') == 'government_action':
                    trust_factor *= 1.1 if event.get('positive', True) else 0.9
        
        return trust_factor
    
    def _get_social_pressure_factor(self, game_context: Dict) -> float:
        """Calculate social pressure factor based on peer behavior."""
        if 'previous_contributions' not in game_context:
            return 1.0
        
        # Get average compliance of peers
        peer_contributions = [
            contrib for name, contrib in game_context['previous_contributions'].items()
            if name != self.name
        ]
        
        if not peer_contributions:
            return 1.0
        
        avg_peer_contribution = sum(peer_contributions) / len(peer_contributions)
        expected_contribution = self.state.endowment * self._get_contribution_factor()
        
        # Calculate pressure based on peer behavior
        if avg_peer_contribution < expected_contribution * 0.8:
            # Peers are under-contributing
            return 0.9 * self.social_pressure_sensitivity
        elif avg_peer_contribution > expected_contribution * 1.2:
            # Peers are over-contributing
            return 1.1 * self.social_pressure_sensitivity
        else:
            return 1.0
    
    def _get_peer_influence(self) -> float:
        """Enhanced peer influence with social learning."""
        if not self.chat_history:
            return 1.0
        
        # Analyze peer behavior
        peer_contributions = [
            msg for msg in self.chat_history
            if 'contribution' in msg.lower()
        ]
        
        # Calculate social learning factor
        social_learning = 1.0
        if peer_contributions:
            positive_contributions = sum(1 for msg in peer_contributions if 'high' in msg.lower() or 'increased' in msg.lower())
            negative_contributions = sum(1 for msg in peer_contributions if 'low' in msg.lower() or 'decreased' in msg.lower())
            
            if positive_contributions > negative_contributions:
                social_learning *= 1.1
            elif negative_contributions > positive_contributions:
                social_learning *= 0.9
        
        # Apply market confidence
        if hasattr(self, 'market_confidence'):
            social_learning *= (1 + self.market_confidence)
        
        return social_learning
    
    def _get_economic_context(self) -> float:
        # Return factor based on economic conditions
        # This would normally come from the EconomicContext class
        return 1.0
    
    def _get_personal_factors(self) -> float:
        # Return factor based on personal circumstances
        if 'Senior' in self.role or 'Student' in self.role:
            return 0.9  # Slight reduction for special cases
        return 1.0
    
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
    
    def communicate(self, current_contributions: Dict) -> str:
        # Tax-specific messages
        messages = {
            'TAX_8': [
                "Paying my fair share of taxes for community development.",
                "Contributing 8% of my income as required.",
                "Supporting public services through tax contributions."
            ],
            'TAX_9': [
                "Fulfilling my tax obligations for 9% of my income.",
                "Contributing to public infrastructure through taxes.",
                "Supporting government programs with my tax payments."
            ],
            'TAX_10': [
                "Meeting my 10% tax responsibility.",
                "Contributing to public services through taxation.",
                "Supporting community development with my tax payments."
            ],
            'TAX_12': [
                "Paying 12% in taxes as required for my income level.",
                "Contributing to public welfare through taxation.",
                "Supporting government initiatives with my tax payments."
            ],
            'TAX_15': [
                "Fulfilling my 15% tax obligation as a high earner.",
                "Contributing to public services through higher taxation.",
                "Supporting community development with my tax payments."
            ]
        }
        
        # Get messages for this agent's tax rate
        tax_messages = messages.get(self.state.priming.value, messages['TAX_8'])
        
        # Add role-specific messages
        role_messages = {
            'Government': "As a government official, I understand the importance of tax compliance.",
            'Business': "Businesses must contribute their fair share through taxes.",
            'Education': "Education funding relies on proper tax collection.",
            'Healthcare': "Healthcare services depend on tax contributions.",
            'Labor': "Taxes support worker protection programs.",
            'Religious': "Tax compliance supports community welfare programs.",
            'Youth': "Taxes fund education and youth programs.",
            'Senior': "Taxes support senior citizen benefits.",
            'OFW': "Taxes contribute to overseas worker protection."
        }
        
        # Combine messages
        all_messages = tax_messages + [role_messages.get(self.role.split()[0], "")]
        
        # Generate message
        message = f"{self.name} ({self.role}): {random.choice(all_messages)}"
        return message
    
    def apply_policy(self, policy: Dict):
        """Apply a policy change to the agent."""
        if 'tax_rate' in policy:
            self.state.priming = PrimingType(f"TAX_{int(policy['tax_rate']*100)}")
        if 'endowment_adjustment' in policy:
            self.state.endowment *= (1 + policy['endowment_adjustment'])

    def _create_system_prompt(self) -> str:
        priming_prompts = {
            PrimingType.TEAMWORK: "You are participating in a teamwork-focused game. Cooperation and mutual support are valued.",
            PrimingType.TAXATION: "You are participating in a taxation-like game. Strategic decision-making is important.",
            PrimingType.COMMUNITY: "You are participating in a community-building game. Bayanihan and shared responsibility are key values.",
            PrimingType.BASELINE: "You are participating in a Public Goods Game."
        }
        
        base_prompt = f"""You are {self.name}, a {self.role} participating in a Public Goods Game.
Your public biography: {self.state.public_bio}
Your private motivation: {self.state.private_bio}
Your current endowment: ₱{self.state.endowment}

{priming_prompts[self.state.priming]}

You must decide how much to contribute to the public pool.
Respond with only a number representing your contribution amount."""
        
        return base_prompt

    def _create_decision_prompt(self, game_context: Dict) -> str:
        prompt = f"""Round {game_context['round'] + 1}
Multiplier: {game_context['multiplier']}x

Other players:
{self._format_other_players(game_context['other_agents'])}"""

        if 'previous_contributions' in game_context:
            prompt += "\n\nPrevious round contributions:"
            for name, amount in game_context['previous_contributions'].items():
                prompt += f"\n{name}: ₱{amount:.2f}"

        if self.chat_history:
            prompt += "\n\nRecent chat messages:"
            for message in self.chat_history[-3:]:  # Show last 3 messages
                prompt += f"\n{message}"

        prompt += "\n\nHow much will you contribute? (Respond with only a number)"
        return prompt

    def _format_other_players(self, other_agents: List[Dict]) -> str:
        return "\n".join([
            f"- {agent['name']}: {agent['public_bio']}"
            for agent in other_agents
            if agent['name'] != self.name
        ])

    def update_compliance_history(self, compliance_rate: float):
        """Update agent's compliance history."""
        self.compliance_history.append(compliance_rate)
        if len(self.compliance_history) > 10:  # Keep last 10 records
            self.compliance_history.pop(0)
        
        # Update risk tolerance based on compliance history
        if compliance_rate < 0.8:
            self.risk_tolerance *= 1.1  # Increase risk tolerance
        elif compliance_rate > 0.95:
            self.risk_tolerance *= 0.9  # Decrease risk tolerance
        
        # Update trust in government
        if compliance_rate > 0.9:
            self.trust_in_government *= 1.05
        elif compliance_rate < 0.7:
            self.trust_in_government *= 0.95

    def _get_fatigue_factor(self) -> float:
        """Calculate fatigue factor based on previous contributions."""
        if not self.last_contribution:
            return 1.0
        
        # Calculate fatigue based on last contribution relative to expected
        expected = self.state.endowment * self._get_contribution_factor()
        last_ratio = self.last_contribution / expected
        
        # Apply fatigue if last contribution was high
        if last_ratio > 1.1:
            self.fatigue_factor *= 0.95
        elif last_ratio < 0.9:
            self.fatigue_factor *= 1.05
        
        # Ensure fatigue factor stays within reasonable bounds
        self.fatigue_factor = max(0.7, min(1.3, self.fatigue_factor))
        
        return self.fatigue_factor
    
    def _get_budget_constraint(self) -> float:
        """Calculate budget constraint based on recent contributions."""
        if not self.last_contribution:
            return 1.0
        
        # Calculate budget impact
        budget_impact = self.last_contribution / self.state.endowment
        
        # Adjust budget constraint
        if budget_impact > 0.2:  # If last contribution was more than 20% of income
            self.budget_constraint *= 0.95
        else:
            self.budget_constraint *= 1.05
        
        # Ensure budget constraint stays within reasonable bounds
        self.budget_constraint = max(0.8, min(1.2, self.budget_constraint))
        
        return self.budget_constraint
    
    def update_reward(self, reward: float):
        """Update agent's behavior based on received reward."""
        self.reward_history.append(reward)
        if len(self.reward_history) > 5:
            self.reward_history.pop(0)
        
        # Adjust behavioral model based on rewards
        avg_reward = sum(self.reward_history) / len(self.reward_history)
        if avg_reward > 1.1:
            # If rewards are good, become more community-focused
            self.behavioral_model['contribution_range'] = (
                max(0.9, self.behavioral_model['contribution_range'][0]),
                min(1.1, self.behavioral_model['contribution_range'][1])
            )
        elif avg_reward < 0.9:
            # If rewards are poor, become more selfish
            self.behavioral_model['contribution_range'] = (
                max(0.7, self.behavioral_model['contribution_range'][0] - 0.1),
                min(1.0, self.behavioral_model['contribution_range'][1] - 0.1)
            )

    def update_behavior(self, outcome: Dict):
        """Update behavior based on outcomes and social learning."""
        # Update trust based on outcomes
        if outcome.get('success', False):
            self.trust_in_government *= 1.05
        else:
            self.trust_in_government *= 0.95
        
        # Update risk tolerance based on peer behavior
        peer_compliance = outcome.get('peer_compliance', 1.0)
        if peer_compliance > 0.9:
            self.risk_tolerance *= 0.95  # Become more conservative
        elif peer_compliance < 0.7:
            self.risk_tolerance *= 1.05  # Become more risk-taking
        
        # Apply fatigue
        self.fatigue_factor *= 0.98  # Gradual fatigue
        
        # Reset fatigue if rewarded
        if outcome.get('rewarded', False):
            self.fatigue_factor = 1.0

class AdaptiveAgent(LLMAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = 0.1
        self.experience = []
    
    def update_behavior(self, outcome: Dict):
        # Learn from outcomes and adjust behavior
        self.experience.append(outcome)
        self._adjust_parameters(outcome)
    
    def _adjust_parameters(self, outcome: Dict):
        # Adjust decision parameters based on outcomes
        if outcome['success']:
            self.learning_rate *= 1.1
        else:
            self.learning_rate *= 0.9 