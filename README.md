# FrozenLake Reinforcement Learning

This repository contains implementations of various reinforcement learning (RL) approaches to solve the **FrozenLake-v1** environment from OpenAI Gym. The goal is to train agents capable of navigating a slippery, grid-based environment to reach a goal while avoiding obstacles and slippery surfaces.
![FrozenLake Demo](Demo/sample_demo.gif)

## Environment Details

The FrozenLake environment is a 4x4 grid with the following elements:

- **S**: Start point (e.g., state 0).
- **F**: Frozen surface (safe to walk on).
- **H**: Hole (fall into it = end of episode).
- **G**: Goal (reaching it = reward).

### Actions:
Agents can take one of four discrete actions:
1. **Left** (0)
2. **Down** (1)
3. **Right** (2)
4. **Up** (3)

### Rewards:
- **Goal (G)**: Reward of +1 for reaching the goal.
- **Holes (H)**: Reward of 0 for falling into a hole.
- **Frozen surface (F)**: Reward of 0 for stepping onto a frozen surface.

### Dynamics:
The environment is **slippery**, meaning actions may not always result in the intended movement.

## Implemented Agents

### 1. Random Agent
- Selects actions randomly without learning from the environment.

### 2. Q-Learning Agent
- Uses the Q-learning algorithm to update a Q-table based on rewards and state-action pairs.
- **Results**: Achieves an average reward of **0.83** after training.

### 3. Deep Q-Learning Agent
- Implements a Deep Q-Network (DQN) with the following architecture:
  - **Input**: One-hot encoded state representation.
  - **Hidden Layer**: Fully connected layer with 128 neurons and ReLU activation.
  - **Output**: Q-values for each action.
- **Policy**: Epsilon-greedy for balancing exploration and exploitation.
- **Optimization**:
  - Loss Function: Mean Squared Error (MSE)
  - Optimizer: Adam
  - Uses a target network to stabilize training.
- **Results**: Achieves an average reward of **0.77** after training.
- Saves a demo of the trained agent’s gameplay as a **GIF**.

## How to Use

### Prerequisites
- Python 3.8 or later
- Install dependencies:
  ```bash
  pip install gym numpy matplotlib

### Running the Agents
1. Random Agent:
  ```bash
  python random_agent.py
  ```
2. Q-Learning Agent:
  ```bash
  python q_learning_agent.py
  ```
3. Deep Q-Learning Agent:
  ```bash
  python deep_q_learning_agent.py
  ```
