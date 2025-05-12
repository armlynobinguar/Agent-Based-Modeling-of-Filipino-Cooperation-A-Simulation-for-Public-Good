import json
import os
from datetime import datetime
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SimulationResults:
    def __init__(self, simulation_id: str = None):
        self.results_dir = "simulation_results"
        if simulation_id:
            self.simulation_id = simulation_id
        else:
            self.simulation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sim_dir = os.path.join(self.results_dir, f"simulation_{self.simulation_id}")
        os.makedirs(self.sim_dir, exist_ok=True)
    
    def save_round(self, round_data: Dict):
        """Save round data to a JSON file."""
        round_file = os.path.join(self.sim_dir, f"round_{round_data['round']}.json")
        with open(round_file, 'w') as f:
            json.dump(round_data, f, indent=2)
    
    def save_summary(self, summary: Dict):
        """Save summary data to a JSON file."""
        summary_file = os.path.join(self.sim_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_final_report(self, report: Dict):
        """Save final report to a JSON file."""
        report_file = os.path.join(self.sim_dir, "final_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def load_simulation(self, simulation_id: str) -> Dict:
        """Load a complete simulation from its ID."""
        sim_dir = os.path.join(self.results_dir, f"simulation_{simulation_id}")
        if not os.path.exists(sim_dir):
            raise FileNotFoundError(f"Simulation {simulation_id} not found")
        
        # Load all round data
        rounds = []
        round_files = sorted([f for f in os.listdir(sim_dir) if f.startswith('round_')])
        for round_file in round_files:
            with open(os.path.join(sim_dir, round_file), 'r') as f:
                rounds.append(json.load(f))
        
        # Load summary and final report
        with open(os.path.join(sim_dir, "summary.json"), 'r') as f:
            summary = json.load(f)
        
        with open(os.path.join(sim_dir, "final_report.json"), 'r') as f:
            final_report = json.load(f)
        
        return {
            'rounds': rounds,
            'summary': summary,
            'final_report': final_report
        }
    
    def analyze_simulation(self, simulation_id: str = None):
        """Analyze a simulation and generate visualizations."""
        if simulation_id:
            data = self.load_simulation(simulation_id)
        else:
            data = self.load_simulation(self.simulation_id)
        
        # Convert round data to DataFrame
        rounds_df = pd.DataFrame([
            {
                'round': r['round'],
                'total_contribution': r['total_contribution'],
                'payout': r['payout'],
                'timestamp': r['timestamp']
            }
            for r in data['rounds']
        ])
        
        # Create visualizations
        self._create_contribution_plot(rounds_df)
        self._create_compliance_plot(data['rounds'])
        self._create_agent_behavior_plot(data['final_report']['agent_summaries'])
    
    def _create_contribution_plot(self, rounds_df: pd.DataFrame):
        """Create plot of contributions over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(rounds_df['round'], rounds_df['total_contribution'], marker='o')
        plt.title('Total Contributions Over Time')
        plt.xlabel('Round')
        plt.ylabel('Total Contribution (₱)')
        plt.grid(True)
        plt.savefig(os.path.join(self.sim_dir, 'contributions.png'))
        plt.close()
    
    def _create_compliance_plot(self, rounds: List[Dict]):
        """Create plot of compliance rates over time."""
        compliance_data = []
        for round_data in rounds:
            for agent_name, contribution in round_data['contributions'].items():
                agent_state = round_data['agent_states'][agent_name]
                expected = agent_state['endowment'] * 0.1  # Assuming 10% base rate
                compliance_rate = contribution / expected
                compliance_data.append({
                    'round': round_data['round'],
                    'agent': agent_name,
                    'compliance_rate': compliance_rate
                })
        
        compliance_df = pd.DataFrame(compliance_data)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=compliance_df, x='round', y='compliance_rate')
        plt.title('Compliance Rates Over Time')
        plt.xlabel('Round')
        plt.ylabel('Compliance Rate')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.sim_dir, 'compliance.png'))
        plt.close()
    
    def _create_agent_behavior_plot(self, agent_summaries: Dict):
        """Create plot of agent behaviors."""
        behavior_data = []
        for agent_name, summary in agent_summaries.items():
            behavior_data.append({
                'agent': agent_name,
                'role': summary['role'],
                'compliance_rate': summary['compliance_rate'],
                'trust': summary['final_trust'],
                'risk_tolerance': summary['final_risk_tolerance']
            })
        
        behavior_df = pd.DataFrame(behavior_data)
        
        # Create subplots with better sizing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Plot 1: Compliance by Role
        sns.boxplot(data=behavior_df, x='role', y='compliance_rate', ax=ax1)
        ax1.set_title('Compliance Rates by Role', fontsize=14, pad=20)
        ax1.set_xlabel('Role', fontsize=12)
        ax1.set_ylabel('Compliance Rate', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Trust vs Risk Tolerance
        scatter = sns.scatterplot(
            data=behavior_df,
            x='trust',
            y='risk_tolerance',
            hue='role',
            size='compliance_rate',
            sizes=(100, 400),  # Increased size range
            ax=ax2,
            alpha=0.7,
            palette='husl'  # Better color palette
        )
        
        # Add agent names as annotations with better positioning
        for idx, row in behavior_df.iterrows():
            # Calculate offset based on position to avoid overlap
            x_offset = 10 if row['trust'] < 1.0 else -10
            y_offset = 10 if row['risk_tolerance'] < 1.0 else -10
            
            ax2.annotate(
                row['agent'],
                (row['trust'], row['risk_tolerance']),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        ax2.set_title('Trust vs Risk Tolerance by Role', fontsize=14, pad=20)
        ax2.set_xlabel('Trust in Government', fontsize=12)
        ax2.set_ylabel('Risk Tolerance', fontsize=12)
        
        # Add quadrant lines
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
        
        # Add quadrant labels with better positioning
        ax2.text(0.25, 0.25, 'Low Trust\nLow Risk', transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        ax2.text(0.25, 0.75, 'Low Trust\nHigh Risk', transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        ax2.text(0.75, 0.25, 'High Trust\nLow Risk', transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        ax2.text(0.75, 0.75, 'High Trust\nHigh Risk', transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        
        # Improve legend
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            title='Role',
            title_fontsize=12,
            fontsize=10
        )
        
        # Add size legend
        size_legend = ax2.get_legend()
        if size_legend:
            size_legend.set_title('Compliance Rate', prop={'size': 12})
            for text in size_legend.get_texts():
                text.set_fontsize(10)
        
        # Set axis limits with some padding
        ax2.set_xlim(behavior_df['trust'].min() - 0.1, behavior_df['trust'].max() + 0.1)
        ax2.set_ylim(behavior_df['risk_tolerance'].min() - 0.1, behavior_df['risk_tolerance'].max() + 0.1)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the plot with high resolution
        plt.savefig(
            os.path.join(self.sim_dir, 'agent_behavior.png'),
            bbox_inches='tight',
            dpi=300,
            pad_inches=0.5
        )
        plt.close()

def save_simulation_results(simulation_id: str, data: Dict):
    """Save simulation results to a directory."""
    # Create simulation directory
    sim_dir = os.path.join("simulation_results", simulation_id)
    os.makedirs(sim_dir, exist_ok=True)
    
    # Save summary data
    summary_path = os.path.join(sim_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save round data
    for round_num, round_data in data.get('rounds', {}).items():
        round_path = os.path.join(sim_dir, f"round_{round_num}.json")
        with open(round_path, 'w') as f:
            json.dump(round_data, f, indent=2)
    
    # Save final report
    report_path = os.path.join(sim_dir, "final_report.json")
    with open(report_path, 'w') as f:
        json.dump(data.get('final_report', {}), f, indent=2)
    
    print(f"\nSimulation results saved to: {sim_dir}/")