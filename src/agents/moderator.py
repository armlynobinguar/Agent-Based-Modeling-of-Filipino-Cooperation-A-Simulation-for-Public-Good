from typing import List, Dict
from .agent import Agent
import random
import json
from datetime import datetime
import os
from ..simulation_results import SimulationResults

class EconomicContext:
    def __init__(self):
        self.indicators = {
            'gdp_growth': 0.0,
            'inflation_rate': 0.0,
            'unemployment_rate': 0.0,
            'market_confidence': 0.0
        }
    
    def update_indicators(self, new_data: Dict):
        for key, value in new_data.items():
            self.indicators[key] = value
    
    def get_tax_adjustment(self) -> float:
        # Adjust tax rates based on economic indicators
        return sum(self.indicators.values()) / len(self.indicators)

class ComplianceMonitor:
    def __init__(self):
        self.compliance_history = {}
        self.risk_factors = {}
    
    def assess_compliance(self, agent: Agent, contribution: float) -> Dict:
        expected = agent.state.endowment * agent._get_contribution_factor()
        compliance_rate = contribution / expected
        
        risk_assessment = {
            'compliance_rate': compliance_rate,
            'risk_level': self._calculate_risk_level(compliance_rate),
            'recommended_actions': self._get_recommendations(compliance_rate)
        }
        
        return risk_assessment

class EventSystem:
    def __init__(self):
        self.events = {
            'tax_scandal': {
                'probability': 0.1,
                'impact': {
                    'trust_in_government': -0.2,
                    'compliance_rate': -0.15
                },
                'message': "News of a tax scandal has shaken public trust in the system."
            },
            'natural_disaster': {
                'probability': 0.05,
                'impact': {
                    'voluntary_contribution': 0.3,
                    'community_spirit': 0.4
                },
                'message': "A natural disaster has increased community solidarity."
            },
            'economic_boom': {
                'probability': 0.15,
                'impact': {
                    'market_confidence': 0.2,
                    'compliance_rate': 0.1
                },
                'message': "Economic growth is boosting market confidence."
            },
            'corruption_exposure': {
                'probability': 0.08,
                'impact': {
                    'trust_in_government': -0.25,
                    'risk_tolerance': 0.2
                },
                'message': "Exposure of corruption has increased risk-taking behavior."
            }
        }
        self.active_events = []
    
    def check_for_events(self) -> List[Dict]:
        """Check for and generate new events."""
        new_events = []
        for event_name, event_data in self.events.items():
            if random.random() < event_data['probability']:
                new_events.append({
                    'name': event_name,
                    'impact': event_data['impact'],
                    'message': event_data['message']
                })
        self.active_events.extend(new_events)
        return new_events
    
    def apply_event_impacts(self, agent: Agent):
        """Apply impacts of active events to an agent."""
        for event in self.active_events:
            for factor, impact in event['impact'].items():
                if factor == 'trust_in_government':
                    agent.trust_in_government *= (1 + impact)
                elif factor == 'risk_tolerance':
                    agent.risk_tolerance *= (1 + impact)
                elif factor == 'compliance_rate':
                    # Initialize compliance history if empty
                    if not agent.compliance_history:
                        agent.compliance_history = [1.0]  # Default compliance rate
                    # Update the last compliance rate
                    agent.compliance_history[-1] *= (1 + impact)
                elif factor == 'voluntary_contribution':
                    agent.behavioral_model['contribution_range'] = (
                        agent.behavioral_model['contribution_range'][0] * (1 + impact),
                        agent.behavioral_model['contribution_range'][1] * (1 + impact)
                    )

class Moderator:
    def __init__(self, agents: List[Agent], multiplier: float = 1.5):
        self.agents = agents
        self.multiplier = multiplier
        self.round_history = []
        self.communication_enabled = False
        self.transparency_enabled = False
        self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        self.results_dir = "simulation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.event_system = EventSystem()
        self.sanctions = {}
        self.rewards = {}
        self.results = SimulationResults(self.simulation_id)
        self.results.round_history = self.round_history  # Pass round history to results
    
    def start_round(self, round_num: int) -> Dict:
        # Check for new events
        new_events = self.event_system.check_for_events()
        
        # Collect contributions from all agents
        contributions = {}
        
        # First pass: collect initial contributions
        for agent in self.agents:
            # Initialize compliance history if empty
            if not agent.compliance_history:
                agent.compliance_history = [1.0]  # Default compliance rate
            
            # Apply event impacts
            self.event_system.apply_event_impacts(agent)
            
            # Apply sanctions/rewards
            self._apply_sanctions_and_rewards(agent)
            
            game_context = self._create_game_context(round_num, agent)
            contribution = agent.decide_contribution(game_context)
            contributions[agent.name] = contribution
        
        # If communication is enabled, allow agents to discuss and revise contributions
        if self.communication_enabled:
            for agent in self.agents:
                # Allow agent to communicate with others
                message = agent.communicate(contributions)
                if message:
                    agent.chat_history.append(message)
                
                # Allow agent to revise contribution based on communication
                game_context = self._create_game_context(round_num, agent)
                revised_contribution = agent.decide_contribution(game_context)
                contributions[agent.name] = revised_contribution
        
        # Update compliance history for each agent
        for agent in self.agents:
            expected = agent.state.endowment * agent._get_contribution_factor()
            actual = contributions[agent.name]
            compliance_rate = actual / expected
            agent.update_compliance_history(compliance_rate)
        
        # Calculate total contribution and individual payout
        total_contribution = sum(contributions.values())
        payout = (total_contribution * self.multiplier) / len(self.agents)
        
        # Record round data
        round_data = {
            'round': round_num,
            'contributions': contributions,
            'total_contribution': total_contribution,
            'payout': payout,
            'timestamp': datetime.now().isoformat(),
            'agent_states': self._capture_agent_states()
        }
        self.round_history.append(round_data)
        self.results.round_history = self.round_history  # Update round history in results
        
        # Save round data and update summary
        self.results.save_round(round_data)
        
        # Create and save summary after each round
        summary = {
            'simulation_id': self.simulation_id,
            'start_time': self.round_history[0]['timestamp'],
            'current_round': round_num,
            'total_rounds': round_num,
            'total_contribution': total_contribution,
            'average_payout': payout,
            'agent_count': len(self.agents),
            'final_contribution': total_contribution,
            'average_payout': sum(r['payout'] for r in self.round_history) / len(self.round_history)
        }
        
        # Generate agent summaries for the final report
        agent_summaries = self._generate_agent_summaries()
        self.results.agent_summaries = agent_summaries  # Pass agent summaries to results
        
        # Save summary and final report
        self.results.save_summary(summary)
        
        return round_data
    
    def _create_game_context(self, round_num: int, current_agent: Agent) -> Dict:
        other_agents = [
            {
                'name': agent.name,
                'public_bio': agent.state.public_bio,
                'role': agent.role
            }
            for agent in self.agents
            if agent != current_agent
        ]
        
        context = {
            'round': round_num,
            'multiplier': self.multiplier,
            'other_agents': other_agents
        }
        
        if self.transparency_enabled and self.round_history:
            context['previous_contributions'] = self.round_history[-1]['contributions']
        
        return context
    
    def _distribute_payouts(self, base_payout: float, policy_feedback: Dict) -> Dict:
        """Distribute payouts with policy feedback adjustments."""
        payouts = {}
        
        for agent in self.agents:
            # Start with base payout
            payout = base_payout
            
            # Apply policy feedback adjustments
            if policy_feedback['additional_benefits']:
                # Give more to lower income agents
                if agent.state.endowment < 20000:
                    payout *= 1.2  # 20% bonus for low income
                elif agent.state.endowment < 40000:
                    payout *= 1.1  # 10% bonus for middle income
            
            if policy_feedback['tax_incentives']:
                # Reward high compliance
                compliance_rate = agent.compliance_history[-1] if agent.compliance_history else 1.0
                if compliance_rate > 0.9:
                    payout *= 1.15  # 15% bonus for high compliance
            
            if policy_feedback['community_projects']:
                # Reward community-focused behavior
                if agent.behavioral_model['contribution_range'][0] > 0.9:
                    payout *= 1.1  # 10% bonus for community focus
            
            payouts[agent.name] = payout
            
            # Update agent's trust in government based on payout
            if payout > base_payout:
                agent.trust_in_government *= 1.05
            elif payout < base_payout:
                agent.trust_in_government *= 0.95
        
        return payouts
    
    def _facilitate_communication(self):
        """Enhanced communication facilitation with policy feedback."""
        # Simulate a group chat session
        for _ in range(3):  # Allow 3 messages per agent
            for agent in self.agents:
                # Get policy feedback for this agent
                policy_feedback = self._calculate_policy_feedback(
                    sum(c for c in self.round_history[-1]['contributions'].values())
                )
                
                # Generate message based on policy feedback
                message = self._generate_chat_message(agent, policy_feedback)
                
                # Add message to all agents' chat history
                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.add_to_chat(message)
        
        # Update social pressure sensitivity based on communication
        for agent in self.agents:
            recent_messages = agent.chat_history[-5:]
            positive_messages = sum(1 for msg in recent_messages if 'comply' in msg.lower() or 'fair' in msg.lower())
            negative_messages = sum(1 for msg in recent_messages if 'avoid' in msg.lower() or 'evade' in msg.lower())
            
            if positive_messages > negative_messages:
                agent.social_pressure_sensitivity *= 1.05
            elif negative_messages > positive_messages:
                agent.social_pressure_sensitivity *= 0.95

    def _apply_sanctions_and_rewards(self, agent: Agent):
        """Apply sanctions and rewards based on compliance history."""
        if not agent.compliance_history:
            return
        
        compliance_rate = agent.compliance_history[-1]
        
        # Apply sanctions for low compliance
        if compliance_rate < 0.7:
            if agent.name not in self.sanctions:
                self.sanctions[agent.name] = 0
            self.sanctions[agent.name] += 1
            
            # Increase penalties for repeat offenders
            penalty = 0.1 * self.sanctions[agent.name]
            agent.trust_in_government *= (1 - penalty)
            agent.behavioral_model['contribution_range'] = (
                agent.behavioral_model['contribution_range'][0] * (1 - penalty),
                agent.behavioral_model['contribution_range'][1] * (1 - penalty)
            )
        
        # Apply rewards for high compliance
        elif compliance_rate > 0.9:
            if agent.name not in self.rewards:
                self.rewards[agent.name] = 0
            self.rewards[agent.name] += 1
            
            # Increase rewards for consistent compliance
            reward = 0.05 * self.rewards[agent.name]
            agent.trust_in_government *= (1 + reward)
            agent.behavioral_model['contribution_range'] = (
                agent.behavioral_model['contribution_range'][0] * (1 + reward),
                agent.behavioral_model['contribution_range'][1] * (1 + reward)
            )
    
    def _generate_chat_message(self, agent: Agent, policy_feedback: Dict) -> str:
        """Enhanced chat message generation with sanctions and rewards."""
        messages = super()._generate_chat_message(agent, policy_feedback)
        
        # Add sanction messages
        if agent.name in self.sanctions and self.sanctions[agent.name] > 0:
            messages.append(f"Warning: {agent.name} has been sanctioned {self.sanctions[agent.name]} times for low compliance.")
        
        # Add reward messages
        if agent.name in self.rewards and self.rewards[agent.name] > 0:
            messages.append(f"Congratulations to {agent.name} for consistent high compliance!")
        
        # Add event messages
        for event in self.event_system.active_events:
            messages.append(event['message'])
        
        return random.choice(messages)
    
    def _capture_agent_states(self) -> Dict:
        """Capture the current state of all agents."""
        agent_states = {}
        for agent in self.agents:
            agent_states[agent.name] = {
                'role': agent.role,
                'endowment': agent.state.endowment,
                'compliance_history': agent.compliance_history,
                'risk_tolerance': agent.risk_tolerance,
                'trust_in_government': agent.trust_in_government,
                'social_pressure_sensitivity': agent.social_pressure_sensitivity,
                'behavioral_model': agent.behavioral_model,
                'fatigue_factor': agent.fatigue_factor,
                'budget_constraint': agent.budget_constraint
            }
        return agent_states
    
    def save_final_report(self):
        """Save a final report of the simulation."""
        if not self.round_history:
            return
        
        report = {
            'simulation_id': self.simulation_id,
            'start_time': self.round_history[0]['timestamp'],
            'end_time': self.round_history[-1]['timestamp'],
            'total_rounds': len(self.round_history),
            'final_contribution': self.round_history[-1]['total_contribution'],
            'average_contribution': sum(r['total_contribution'] for r in self.round_history) / len(self.round_history),
            'agent_summaries': self._generate_agent_summaries(),
            'round_summaries': self._generate_round_summaries()
        }
        
        self.results.save_final_report(report)
        self.results.analyze_simulation()
    
    def _generate_agent_summaries(self) -> Dict:
        """Generate summaries for each agent."""
        summaries = {}
        for agent in self.agents:
            agent_rounds = [r for r in self.round_history if agent.name in r['contributions']]
            summaries[agent.name] = {
                'role': agent.role,
                'average_contribution': sum(r['contributions'][agent.name] for r in agent_rounds) / len(agent_rounds),
                'compliance_rate': sum(agent.compliance_history) / len(agent.compliance_history) if agent.compliance_history else 0,
                'final_trust': agent.trust_in_government,
                'final_risk_tolerance': agent.risk_tolerance
            }
        return summaries
    
    def _generate_round_summaries(self) -> List[Dict]:
        """Generate summaries for each round."""
        return [{
            'round': r['round'],
            'total_contribution': r['total_contribution'],
            'payout': r['payout'],
            'timestamp': r['timestamp']
        } for r in self.round_history] 