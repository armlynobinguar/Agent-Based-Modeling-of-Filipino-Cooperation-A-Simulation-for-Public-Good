# 🧾 Tax Compliance Simulation

A multi-agent simulation system that models tax compliance behavior in the Filipino context using different agent types, including LLM-based agents and Iowa Gambling Task (IGT) agents.

---

## 📌 Overview

This simulation captures tax compliance behavior in a diverse population, using a blend of behavioral and AI-driven agent models:

- **Agent Types**:
  - LLM Agents (powered by GPT-3.5)
  - IGT Agents (based on Iowa Gambling Task)
  - Crew Agents (for simulating social influence and network effects)

- **Behavioral Factors**:
  - Tax rates (ranging from 8% to 15%)
  - Trust in government
  - Risk tolerance
  - Social pressure and communication networks
  - Economic context (inflation, GDP growth, unemployment, market confidence)

---

## 🌐 Agent Population

| Sector             | Example Agents                                 |
|--------------------|-------------------------------------------------|
| Government         | Mayor, Public Attorney                          |
| Business           | Business Tycoon, Restaurant Owner, Sari-sari Owner |
| Education          | Professor, Public School Teacher                |
| Healthcare         | Private Doctor, Public Nurse                    |
| Labor              | Construction Foreman, Street Vendor             |
| Religious          | Parish Priest                                   |
| Youth              | College Student                                 |
| Senior Citizen     | Retired Teacher                                 |
| OFW Family         | OFW Wife                                        |

---

## 🧠 Agent Behaviors

- **Risk Averse**
- **Tax Avoidant**
- **Rational Selfish**
- **Community Focused**

---

## 📊 Features

- Multi-agent simulation framework using Python
- GPT-3.5-based agents for decision-making
- Social graph and peer influence modeling
- Public Goods Game logic (via `pgg_manager`)
- Economic environment modifiers
- Visualization of:
  - Trust vs. Risk Tolerance
  - Compliance rates over time
  - Contribution patterns across sectors

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/armlynobinguar/Agent-Based-Modeling-of-Filipino-Cooperation-A-Simulation-for-Public-Good.git
cd Agent-Based-Modeling-of-Filipino-Cooperation-A-Simulation-for-Public-Good
```

### Create a Virtual Environment
```bash
python -m venv .venv
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Your Environment Variables
Create a .env file with your OpenAI key:

```bash
OPENAI_API_KEY=your-api-key-here
```

### Run the Simulation

```bash
python -m src.main
```

## 📁 Project Structure

tax-compliance-simulation/
├── src/
│   ├── agents/
│   │   ├── agent.py              # Base agent class
│   │   ├── llm_agent.py          # LLM-based agent logic
│   │   ├── igt_agent.py          # IGT-based decision-making
│   │   ├── crew_agent.py         # Social influence modeling
│   │   └── moderator.py          # Simulation moderator
│   ├── game/
│   │   └── pgg_manager.py        # Public Goods Game logic
│   ├── simulation_results.py     # Result analysis and plotting
│   └── main.py                   # Main simulation runner
├── requirements.txt              # Python dependencies
├── setup.sh                      # Unix/MacOS setup
├── setup.bat                     # Windows setup
├── .env                          # API key file (excluded from repo)
└── simulation_results/           # Folder for generated results (gitignored)

## 📄 License
This project is licensed under the Apache License 2.0.

## Author

Armielyn C. Obinguar
