# Artillery RL – Gymnasium (Q-Learning + SAC)

## Overview

This project implements an AI-powered Artillery Simulation using Reinforcement Learning techniques in the Gymnasium environment. The system trains autonomous agents to learn optimal targeting and firing strategies under dynamic environmental conditions such as wind effects and obstacles.

The project combines:

* Q-Learning
* Soft Actor-Critic (SAC)
* Custom Gymnasium environments
* Reward-based learning
* Autonomous strategy optimization

The repository demonstrates how Reinforcement Learning can be applied to strategic combat simulations and intelligent control systems.

---

## Project Structure

```text
.
├── README.md                 # Project documentation
├── env.py                    # Custom Gymnasium environment
├── qlearning.py              # Q-Learning implementation
├── train_sac.py              # SAC training pipeline
├── sac_agent_reward.py       # SAC reward and agent logic
├── test_qlearning.py         # Testing Q-Learning agent
├── wind_obs_wrapper.py       # Wind and obstacle wrapper
├── main_demo.py              # Demo execution script
├── requirements.txt          # Project dependencies
```

---

## Features

### Reinforcement Learning Algorithms

* Q-Learning implementation
* Soft Actor-Critic (SAC)
* Continuous and discrete control learning
* Reward optimization strategies

### Environment Simulation

* Custom artillery environment
* Wind-effect simulation
* Obstacle-aware targeting
* Dynamic state transitions

### AI Agent Capabilities

* Autonomous targeting
* Strategic firing decisions
* Adaptive learning behavior
* Environment interaction and optimization

### Training and Evaluation

* Agent training pipelines
* Reward tracking
* Performance evaluation
* Testing and demo scripts

---

## Technologies Used

* Python
* PyTorch
* Gymnasium
* NumPy
* Reinforcement Learning
* Deep Learning

---

## Workflow

### 1. Environment Setup

The custom Gymnasium artillery environment is initialized with obstacles and wind dynamics.

### 2. State Observation

The agent observes environmental states such as:

* Target position
* Wind conditions
* Projectile parameters
* Obstacles

### 3. Action Selection

The RL agent chooses firing parameters and targeting actions.

### 4. Reward Optimization

The agent receives rewards based on:

* Accuracy
* Successful targeting
* Efficient strategy
* Collision avoidance

### 5. Policy Learning

Q-Learning and SAC algorithms update policies to improve long-term performance.

### 6. Evaluation

Trained agents are tested for targeting efficiency and decision-making capability.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/artillery-rl.git
cd artillery-rl
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train SAC Agent

```bash
python train_sac.py
```

### Run Q-Learning

```bash
python qlearning.py
```

### Test Q-Learning Agent

```bash
python test_qlearning.py
```

### Run Demo

```bash
python main_demo.py
```

---

## Example Use Case

### Input

* Wind conditions
* Target location
* Environment obstacles

### Output

The RL agent predicts:

* Optimal firing angle
* Firing power
* Strategic targeting action

---

## Applications

* Autonomous defense systems
* Strategic simulation environments
* AI-based targeting systems
* Robotics control systems
* Reinforcement learning research
* Military simulation AI

---

## Future Improvements

Potential enhancements:

* Multi-agent reinforcement learning
* PPO and DDPG implementations
* Real-time visualization
* Advanced physics simulation
* Curriculum learning
* Distributed RL training

---

## Results

The project demonstrates how reinforcement learning agents can autonomously learn targeting and firing strategies in dynamic environments using reward-driven optimization and deep learning techniques.

---

## License

This project is intended for research and educational purposes.

---

## Author

Created by Manas.
