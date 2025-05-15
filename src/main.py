from src.agents.llm_agent import LLMAgent, AdaptiveAgent
from src.agents.agent import AgentState, PrimingType, Agent
from src.agents.moderator import Moderator, EconomicContext, ComplianceMonitor
from src.agents.crew_agent import CommunicationSystem, SocialNetwork
from src.agents.igt_agent import IGTAgent
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import random
from datetime import datetime
import os
import json

def run_simulation(rounds: int = 10, 
                  communication: bool = True,
                  transparency: bool = True) -> List[Agent]:
    """Run the tax compliance simulation."""
    # Create simulation ID
    simulation_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create 18 agents with diverse backgrounds and their monthly salaries
    agents = [
        # Government Sector
        LLMAgent("Mayor Santos", "Government Official", AgentState(
            public_bio="Elected mayor with 15 years of public service",
            private_bio="Focused on community development and public welfare",
            endowment=50000.0,  # ₱50,000 monthly salary
            priming=PrimingType.TAX_15  # 15% tax rate
        )),
        LLMAgent("Atty. Reyes", "Public Attorney", AgentState(
            public_bio="Public attorney handling community cases",
            private_bio="Believes in justice and equal access to legal services",
            endowment=35000.0,  # ₱35,000 monthly salary
            priming=PrimingType.TAX_12  # 12% tax rate
        )),
        
        # Business Sector
        LLMAgent("Mr. Tan", "Business Tycoon", AgentState(
            public_bio="Owner of multiple businesses and properties",
            private_bio="Strategic investor focused on economic growth",
            endowment=150000.0,  # ₱150,000 monthly income
            priming=PrimingType.TAX_15  # 15% tax rate
        )),
        LLMAgent("Mrs. Garcia", "Restaurant Owner", AgentState(
            public_bio="Successful restaurant chain owner",
            private_bio="Entrepreneur supporting local suppliers",
            endowment=45000.0,  # ₱45,000 monthly income
            priming=PrimingType.TAX_12  # 12% tax rate
        )),
        LLMAgent("Mang Jose", "Sari-sari Store Owner", AgentState(
            public_bio="Neighborhood store owner for 20 years",
            private_bio="Community-focused small business owner",
            endowment=15000.0,  # ₱15,000 monthly income
            priming=PrimingType.TAX_8  # 8% tax rate
        )),
        
        # Education Sector
        LLMAgent("Dr. Cruz", "University Professor", AgentState(
            public_bio="PhD holder teaching at state university",
            private_bio="Advocate for educational reform",
            endowment=40000.0,  # ₱40,000 monthly salary
            priming=PrimingType.TAX_12  # 12% tax rate
        )),
        LLMAgent("Teacher Ana", "Public School Teacher", AgentState(
            public_bio="Dedicated public school teacher",
            private_bio="Passionate about student development",
            endowment=25000.0,  # ₱25,000 monthly salary
            priming=PrimingType.TAX_10  # 10% tax rate
        )),
        
        # Healthcare Sector
        LLMAgent("Dr. Lim", "Private Doctor", AgentState(
            public_bio="Medical practitioner with private clinic",
            private_bio="Provides healthcare to both rich and poor",
            endowment=80000.0,  # ₱80,000 monthly income
            priming=PrimingType.TAX_15  # 15% tax rate
        )),
        LLMAgent("Nurse Maria", "Public Hospital Nurse", AgentState(
            public_bio="Senior nurse at public hospital",
            private_bio="Dedicated to public healthcare",
            endowment=30000.0,  # ₱30,000 monthly salary
            priming=PrimingType.TAX_10  # 10% tax rate
        )),
        
        # Labor Sector
        LLMAgent("Mang Pedro", "Construction Foreman", AgentState(
            public_bio="Experienced construction worker",
            private_bio="Supports family through skilled labor",
            endowment=20000.0,  # ₱20,000 monthly income
            priming=PrimingType.TAX_9  # 9% tax rate
        )),
        LLMAgent("Aling Nena", "Street Vendor", AgentState(
            public_bio="Long-time street food vendor",
            private_bio="Hardworking mother of three",
            endowment=12000.0,  # ₱12,000 monthly income
            priming=PrimingType.TAX_8  # 8% tax rate
        )),
        
        # Religious Sector
        LLMAgent("Father Gomez", "Parish Priest", AgentState(
            public_bio="Community spiritual leader",
            private_bio="Promotes social justice and charity",
            endowment=25000.0,  # ₱25,000 monthly stipend
            priming=PrimingType.TAX_10  # 10% tax rate
        )),
        
        # Youth Sector
        LLMAgent("Sarah", "College Student", AgentState(
            public_bio="Scholarship student studying medicine",
            private_bio="Aspiring doctor from humble background",
            endowment=8000.0,  # ₱8,000 monthly allowance
            priming=PrimingType.TAX_8  # 8% tax rate
        )),
        
        # Senior Citizen
        LLMAgent("Lola Remedios", "Retired Teacher", AgentState(
            public_bio="Former public school teacher with pension",
            private_bio="Active in community programs",
            endowment=15000.0,  # ₱15,000 monthly pension
            priming=PrimingType.TAX_8  # 8% tax rate
        )),
        
        # OFW Family
        LLMAgent("Mrs. Santos", "OFW Wife", AgentState(
            public_bio="Manages family while husband works abroad",
            private_bio="Runs small business with remittance money",
            endowment=30000.0,  # ₱30,000 monthly remittance
            priming=PrimingType.TAX_10  # 10% tax rate
        )),
        
        # Add IGT agents
        IGTAgent("Risk Taker", "Entrepreneur", AgentState(
            public_bio="Serial entrepreneur with high risk tolerance",
            private_bio="Believes in high-risk, high-reward strategies",
            endowment=100000.0,
            priming=PrimingType.TAX_15
        )),
        IGTAgent("Risk Averse", "Pensioner", AgentState(
            public_bio="Retired with conservative investment strategy",
            private_bio="Prefers safe, steady returns",
            endowment=20000.0,
            priming=PrimingType.TAX_8
        )),
        IGTAgent("Balanced", "Financial Advisor", AgentState(
            public_bio="Professional financial advisor",
            private_bio="Balances risk and reward for clients",
            endowment=60000.0,
            priming=PrimingType.TAX_12
        ))
    ]
    
    # Initialize communication and social networks
    communication_system = CommunicationSystem()
    social_network = SocialNetwork()
    
    # Set up social connections
    for agent in agents:
        for other_agent in agents:
            if agent != other_agent:
                # Connect agents based on their roles
                if 'Government' in agent.role and 'Business' in other_agent.role:
                    social_network.add_connection(agent.name, other_agent.name, 1.2)
                elif 'Business' in agent.role and 'Labor' in other_agent.role:
                    social_network.add_connection(agent.name, other_agent.name, 1.1)
                else:
                    social_network.add_connection(agent.name, other_agent.name, 1.0)
    
    # Create moderator with economic context
    moderator = Moderator(agents, multiplier=1.5)
    moderator.communication_enabled = communication
    moderator.transparency_enabled = transparency
    
    # Initialize economic context
    economic_context = EconomicContext()
    economic_context.update_indicators({
        'gdp_growth': 0.05,
        'inflation_rate': 0.03,
        'unemployment_rate': 0.05,
        'market_confidence': 0.07
    })
    
    # Run simulation
    for round_num in range(1, rounds + 1):
        print(f"\n{'='*50}")
        print(f"=== Round {round_num} ===")
        print(f"{'='*50}")
        
        # Update economic context
        economic_context.update_indicators({
            'gdp_growth': random.uniform(0.03, 0.07),
            'inflation_rate': random.uniform(0.02, 0.04),
            'unemployment_rate': random.uniform(0.04, 0.06),
            'market_confidence': random.uniform(0.05, 0.09)
        })
        
        # Start round and get contributions
        round_data = moderator.start_round(round_num)
        
        # Process communications
        if communication:
            for agent in agents:
                message = agent.communicate(round_data['contributions'])
                if message:
                    communication_system.process_message(
                        message,
                        agent.name,
                        'informal'
                    )
        
        # Print round results with more detail
        print("\nContributions by Sector:")
        for sector in ['Government', 'Business', 'Education', 'Healthcare', 'Labor', 'Religious', 'Youth', 'Senior', 'OFW']:
            sector_agents = [a for a in agents if sector in a.role]
            if sector_agents:
                print(f"\n{sector} Sector:")
                for agent in sector_agents:
                    contribution = round_data['contributions'][agent.name]
                    tax_rate = agent._get_contribution_factor() * 100
                    peer_influence = social_network.get_peer_influence(agent.name)
                    print(f"  {agent.name} ({agent.role}):")
                    print(f"    Monthly Income: ₱{agent.state.endowment:,.2f}")
                    print(f"    Tax Rate: {tax_rate:.1f}%")
                    print(f"    Tax Contribution: ₱{contribution:,.2f}")
                    print(f"    Percentage of Income: {(contribution/agent.state.endowment)*100:.1f}%")
                    print(f"    Peer Influence Factor: {peer_influence:.2f}")
        
        # Print round summary
        print(f"\n{'='*50}")
        print("Round Summary:")
        print(f"Total Tax Collection: ₱{round_data['total_contribution']:,.2f}")
        print(f"Average Tax Rate: {(round_data['total_contribution'] * 100 / sum(a.state.endowment for a in agents)):.1f}%")
        print(f"Economic Indicators:")
        for indicator, value in economic_context.indicators.items():
            print(f"  {indicator}: {value:.2%}")
        
        # If communication is enabled, show some agent interactions
        if communication and round_num < rounds:
            print("\nAgent Communications:")
            for channel in communication_system.channels:
                if communication_system.channels[channel]:
                    print(f"\n{channel.upper()} Channel:")
                    for message in communication_system.channels[channel][-3:]:  # Show last 3 messages
                        print(f"  {message}")
        
        print(f"\n{'='*50}")
        
        # Save round data
        save_round_data(simulation_id, round_num, round_data)
    
    # Generate and save final report
    final_report = generate_final_report(agents, moderator)
    save_final_report(simulation_id, final_report)
    
    return agents

def analyze_results(history, agents):
    # Convert history to DataFrame
    df = pd.DataFrame(history)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(4, 2)
    
    # 1. Contributions Over Time by Sector
    ax1 = fig.add_subplot(gs[0, 0])
    for agent in agents:
        contributions = [round_data['contributions'][agent.name] for round_data in history]
        ax1.plot(range(len(contributions)), contributions, 
                label=f"{agent.name} ({agent.role})", 
                marker='o')
    
    ax1.set_title('Contributions Over Time by Sector')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Contribution Amount (₱)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # 2. Contribution as Percentage of Income
    ax2 = fig.add_subplot(gs[0, 1])
    for agent in agents:
        contributions = [round_data['contributions'][agent.name] for round_data in history]
        percentages = [c/agent.state.endowment * 100 for c in contributions]
        ax2.plot(range(len(percentages)), percentages, 
                label=f"{agent.name}", 
                marker='o')
    
    ax2.set_title('Contribution as Percentage of Income')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Percentage of Income (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    # 3. Income vs Contribution Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    income_data = []
    for agent in agents:
        avg_contribution = sum(round_data['contributions'][agent.name] 
                             for round_data in history) / len(history)
        income_data.append({
            'Agent': f"{agent.name} ({agent.role})",
            'Income': agent.state.endowment,
            'Avg Contribution': avg_contribution,
            'Contribution %': (avg_contribution/agent.state.endowment) * 100
        })
    
    income_df = pd.DataFrame(income_data)
    sns.scatterplot(data=income_df, x='Income', y='Avg Contribution', 
                   hue='Agent', size='Contribution %', ax=ax3)
    ax3.set_title('Income vs Contribution Analysis')
    ax3.set_xlabel('Income (₱)')
    ax3.set_ylabel('Average Contribution (₱)')
    
    # 4. Contributions by Sector and Priming
    ax4 = fig.add_subplot(gs[1, 1])
    sector_data = []
    for round_data in history:
        for agent in agents:
            sector = agent.role.split()[0]  # Get sector from role
            sector_data.append({
                'Sector': sector,
                'Priming': agent.state.priming.value,
                'Contribution': round_data['contributions'][agent.name],
                'Income': agent.state.endowment
            })
    
    sector_df = pd.DataFrame(sector_data)
    sns.boxplot(data=sector_df, x='Sector', y='Contribution', 
                hue='Priming', ax=ax4)
    ax4.set_title('Contributions by Sector and Priming')
    ax4.set_xlabel('Sector')
    ax4.set_ylabel('Contribution Amount (₱)')
    plt.xticks(rotation=45)
    
    # 5. Income Distribution by Sector
    ax5 = fig.add_subplot(gs[2, 0])
    income_dist = pd.DataFrame({
        'Agent': [agent.name for agent in agents],
        'Income': [agent.state.endowment for agent in agents],
        'Sector': [agent.role.split()[0] for agent in agents]
    })
    sns.barplot(data=income_dist, x='Sector', y='Income', ax=ax5)
    ax5.set_title('Income Distribution by Sector')
    ax5.set_xlabel('Sector')
    ax5.set_ylabel('Income (₱)')
    plt.xticks(rotation=45)
    
    # 6. Contribution Efficiency by Income Level
    ax6 = fig.add_subplot(gs[2, 1])
    efficiency_data = []
    for round_data in history:
        for agent in agents:
            contribution = round_data['contributions'][agent.name]
            efficiency = (contribution/agent.state.endowment) * 100
            efficiency_data.append({
                'Agent': agent.name,
                'Income Level': 'High' if agent.state.endowment >= 200 else 
                              'Medium' if agent.state.endowment >= 100 else 'Low',
                'Efficiency': efficiency,
                'Round': round_data['round']
            })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    sns.boxplot(data=efficiency_df, x='Income Level', y='Efficiency', ax=ax6)
    ax6.set_title('Contribution Efficiency by Income Level')
    ax6.set_xlabel('Income Level')
    ax6.set_ylabel('Contribution Efficiency (%)')
    
    # 7. Cumulative Contributions Over Time
    ax7 = fig.add_subplot(gs[3, :])
    cumulative_data = []
    for round_num, round_data in enumerate(history):
        total = sum(round_data['contributions'].values())
        cumulative_data.append({
            'Round': round_num + 1,
            'Cumulative': total
        })
    
    cumulative_df = pd.DataFrame(cumulative_data)
    cumulative_df['Cumulative'] = cumulative_df['Cumulative'].cumsum()
    sns.lineplot(data=cumulative_df, x='Round', y='Cumulative', ax=ax7)
    ax7.set_title('Cumulative Contributions Over Time')
    ax7.set_xlabel('Round')
    ax7.set_ylabel('Cumulative Contributions (₱)')
    ax7.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('game_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nAverage Contributions by Agent:")
    for agent in agents:
        agent_contributions = [round_data['contributions'][agent.name] 
                            for round_data in history]
        avg_contribution = sum(agent_contributions)/len(agent_contributions)
        print(f"{agent.name} ({agent.role}): ₱{avg_contribution:.2f}")
    
    print("\nAverage Contributions by Sector:")
    for sector in sector_df['Sector'].unique():
        sector_contributions = sector_df[sector_df['Sector'] == sector]['Contribution']
        print(f"{sector}: ₱{sector_contributions.mean():.2f}")
    
    print("\nAverage Contributions by Income Level:")
    for level in efficiency_df['Income Level'].unique():
        level_contributions = efficiency_df[efficiency_df['Income Level'] == level]['Efficiency']
        print(f"{level}: {level_contributions.mean():.1f}% of income")
    
    # Calculate total contributions and payouts
    total_contributions = sum(round_data['total_contribution'] for round_data in history)
    total_payouts = sum(round_data['payout'] for round_data in history)
    
    print("\nTotal Contributions:", f"₱{total_contributions:.2f}")
    print("Average Payout per Round:", f"₱{total_payouts/len(history):.2f}")
    print("Average Efficiency:", f"{sum(efficiency_df['Efficiency'])/len(efficiency_df):.1f}%")

def analyze_tax_contributions(agents, rounds=10):
    tax_rates = [PrimingType.TAX_8, PrimingType.TAX_9, PrimingType.TAX_10]
    results = []
    
    for tax_rate in tax_rates:
        # Create agents with tax priming
        tax_agents = [
            LLMAgent(agent.name, agent.role, AgentState(
                public_bio=agent.state.public_bio,
                private_bio=agent.state.private_bio,
                endowment=agent.state.endowment,
                priming=tax_rate
            ))
            for agent in agents
        ]
        
        # Run simulation
        moderator = Moderator(tax_agents, multiplier=1.5)
        moderator.communication_enabled = True
        moderator.transparency_enabled = True
        
        for round_num in range(rounds):
            round_data = moderator.start_round(round_num)
            results.append({
                'tax_rate': tax_rate.value,
                'round': round_num,
                'total_contribution': round_data['total_contribution'],
                'efficiency': (round_data['total_contribution'] * 100 / sum(a.state.endowment for a in tax_agents))
            })
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Total Contributions by Tax Rate
    plt.subplot(2, 1, 1)
    tax_df = pd.DataFrame(results)
    sns.boxplot(data=tax_df, x='tax_rate', y='total_contribution')
    plt.title('Total Contributions by Tax Rate')
    plt.xlabel('Tax Rate')
    plt.ylabel('Total Contribution (₱)')
    
    # Plot 2: Efficiency by Tax Rate
    plt.subplot(2, 1, 2)
    sns.boxplot(data=tax_df, x='tax_rate', y='efficiency')
    plt.title('Contribution Efficiency by Tax Rate')
    plt.xlabel('Tax Rate')
    plt.ylabel('Efficiency (%)')
    
    plt.tight_layout()
    plt.savefig('tax_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\n=== Tax Rate Analysis ===")
    for tax_rate in tax_rates:
        tax_data = tax_df[tax_df['tax_rate'] == tax_rate.value]
        print(f"\n{tax_rate.value}:")
        print(f"Average Contribution: ₱{tax_data['total_contribution'].mean():.2f}")
        print(f"Average Efficiency: {tax_data['efficiency'].mean():.1f}%")
        print(f"Contribution Range: ₱{tax_data['total_contribution'].min():.2f} - ₱{tax_data['total_contribution'].max():.2f}")

def analyze_tax_compliance(agents, rounds=10):
    results = []
    
    for round_num in range(rounds):
        for agent in agents:
            tax_rate = agent._get_contribution_factor()
            expected_tax = agent.state.endowment * tax_rate
            actual_contribution = agent.decide_contribution({})
            compliance_rate = (actual_contribution / expected_tax) * 100
            
            results.append({
                'agent': agent.name,
                'role': agent.role,
                'income': agent.state.endowment,
                'tax_rate': tax_rate * 100,
                'expected_tax': expected_tax,
                'actual_contribution': actual_contribution,
                'compliance_rate': compliance_rate,
                'round': round_num
            })
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Tax Compliance by Income Level
    plt.subplot(2, 1, 1)
    tax_df = pd.DataFrame(results)
    sns.boxplot(data=tax_df, x='tax_rate', y='compliance_rate')
    plt.title('Tax Compliance by Tax Rate')
    plt.xlabel('Tax Rate (%)')
    plt.ylabel('Compliance Rate (%)')
    
    # Plot 2: Actual vs Expected Tax Contributions
    plt.subplot(2, 1, 2)
    sns.scatterplot(data=tax_df, x='expected_tax', y='actual_contribution', 
                   hue='tax_rate', size='income')
    plt.title('Actual vs Expected Tax Contributions')
    plt.xlabel('Expected Tax (₱)')
    plt.ylabel('Actual Contribution (₱)')
    
    plt.tight_layout()
    plt.savefig('tax_compliance.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\n=== Tax Compliance Analysis ===")
    for tax_rate in sorted(tax_df['tax_rate'].unique()):
        tax_data = tax_df[tax_df['tax_rate'] == tax_rate]
        print(f"\n{tax_rate}% Tax Rate:")
        print(f"Average Compliance: {tax_data['compliance_rate'].mean():.1f}%")
        print(f"Total Tax Collected: ₱{tax_data['actual_contribution'].sum():.2f}")
        print(f"Expected Tax: ₱{tax_data['expected_tax'].sum():.2f}")
        print(f"Compliance Range: {tax_data['compliance_rate'].min():.1f}% - {tax_data['compliance_rate'].max():.1f}%")

def simulate_policy_change(policy: Dict, agents: List[Agent], rounds: int = 10):
    results = []
    for round_num in range(rounds):
        # Apply policy changes
        for agent in agents:
            agent.apply_policy(policy)
        
        # Run simulation round
        round_data = run_simulation_round(agents)
        results.append(round_data)
    
    return analyze_policy_results(results)

def run_simulation_round(agents: List[Agent]) -> Dict:
    """Run a single round of simulation."""
    moderator = Moderator(agents, multiplier=1.5)
    moderator.communication_enabled = True
    moderator.transparency_enabled = True
    return moderator.start_round(0)

def analyze_policy_results(results: List[Dict]) -> Dict:
    """Analyze the results of a policy change simulation."""
    total_contributions = sum(r['total_contribution'] for r in results)
    avg_contribution = total_contributions / len(results)
    
    return {
        'total_contributions': total_contributions,
        'average_contribution': avg_contribution,
        'rounds': len(results)
    }

def analyze_simulation_results(simulation_id: str):
    """Analyze results from a specific simulation."""
    from .simulation_results import SimulationResults
    
    results = SimulationResults(simulation_id)
    data = results.load_simulation(simulation_id)
    
    # Print summary statistics
    print("\n=== Simulation Summary ===")
    print(f"Total Rounds: {data['summary']['total_rounds']}")
    print(f"Final Contribution: ₱{data['summary']['final_contribution']:,.2f}")
    print(f"Average Payout: ₱{data['summary']['average_payout']:,.2f}")
    
    # Print agent summaries
    print("\n=== Agent Summaries ===")
    for agent_name, summary in data['final_report']['agent_summaries'].items():
        print(f"\n{agent_name} ({summary['role']}):")
        print(f"Average Contribution: ₱{summary['average_contribution']:,.2f}")
        print(f"Compliance Rate: {summary['compliance_rate']:.1%}")
        print(f"Final Trust: {summary['final_trust']:.2f}")
        print(f"Risk Tolerance: {summary['final_risk_tolerance']:.2f}")
    
    # Generate visualizations
    results.analyze_simulation()
    print(f"\nVisualizations saved to: simulation_results/simulation_{simulation_id}/")

def save_round_data(simulation_id: str, round_num: int, round_data: Dict):
    """Save data for a single round."""
    sim_dir = os.path.join("simulation_results", simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    
    round_path = os.path.join(sim_dir, f"round_{round_num}.json")
    with open(round_path, 'w') as f:
        json.dump(round_data, f, indent=2)

def save_final_report(simulation_id: str, final_report: Dict):
    """Save the final simulation report."""
    sim_dir = os.path.join("simulation_results", simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    
    # Save final report
    report_path = os.path.join(sim_dir, "final_report.json")
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(sim_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'total_rounds': len(final_report.get('rounds', {})),
            'final_contribution': final_report.get('total_contribution', 0),
            'average_payout': final_report.get('average_payout', 0)
        }, f, indent=2)

def generate_final_report(agents: List[Agent], moderator: Moderator) -> Dict:
    """Generate the final simulation report."""
    agent_summaries = {}
    for agent in agents:
        agent_summaries[agent.name] = {
            'role': agent.role,
            'average_contribution': sum(agent.compliance_history) / len(agent.compliance_history) if agent.compliance_history else 0,
            'compliance_rate': agent.compliance_history[-1] if agent.compliance_history else 0,
            'final_trust': getattr(agent, 'trust_in_government', 1.0),
            'final_risk_tolerance': getattr(agent, 'risk_tolerance', 1.0)
        }
    
    # Calculate total contribution from the last round
    total_contribution = sum(agent.compliance_history[-1] * agent.state.endowment for agent in agents if agent.compliance_history)
    
    # Get the last round's data from round_history
    last_round = moderator.round_history[-1] if moderator.round_history else {}
    average_payout = last_round.get('payout', 0) if last_round else 0
    
    return {
        'agent_summaries': agent_summaries,
        'total_contribution': total_contribution,
        'average_payout': average_payout,
        'rounds': moderator.round_history
    }

if __name__ == "__main__":
    print("Starting tax simulation with 15 diverse agents...")
    print("Running 20 rounds to analyze tax compliance...")
    
    # Run simulation
    agents = run_simulation(rounds=20, communication=True, transparency=True)
    
    # Get the latest simulation ID
    simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Analyze results
    print("\nAnalyzing simulation results...")
    analyze_simulation_results(simulation_id)
    
    print("\nSimulation complete! Check the simulation_results directory for detailed analysis.")
