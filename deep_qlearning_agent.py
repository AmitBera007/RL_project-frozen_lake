import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
from tqdm import tqdm
import random
import imageio

# Neural network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Epsilon-greedy policy
def epsilon_greedy_policy(network, state, epsilon, action_dim, device):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, action_dim)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        with torch.no_grad():
            q_values = network(state_tensor)
        return torch.argmax(q_values).item()

# Deep Q-Learning
def deep_q_learning(env, num_episodes, replay_buffer_size=1000, batch_size=64,
                    gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_min=0.1, 
                    epsilon_decay=0.995, target_update_freq=10, device='cpu'):
    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    # Initialize networks and optimizer
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)

    # Experience replay buffer
    replay_buffer = deque(maxlen=replay_buffer_size)

    def add_to_replay_buffer(transition):
        replay_buffer.append(transition)

    def sample_from_replay_buffer():
        return random.sample(replay_buffer, batch_size)

    epsilon = epsilon_start
    pbar = tqdm(total=num_episodes, dynamic_ncols=True)

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0

        while not done:
            # One-hot encode state
            state_one_hot = np.eye(env.observation_space.n)[state]

            # Select action
            action = epsilon_greedy_policy(q_network, state_one_hot, epsilon, action_dim, device)

            # Step environment
            next_state, reward, done, _, _ = env.step(action)
            next_state_one_hot = np.eye(env.observation_space.n)[next_state]

            # Add transition to replay buffer
            add_to_replay_buffer((state_one_hot, action, reward, next_state_one_hot, done))

            # Train network
            if len(replay_buffer) >= batch_size:
                transitions = sample_from_replay_buffer()
                states, actions, rewards, next_states, dones = zip(*transitions)
                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                q_values = q_network(states).gather(1, actions)
                next_q_values = target_network(next_states).max(1)[0].detach()
                targets = rewards + gamma * next_q_values * (1 - dones)
                targets = targets.unsqueeze(1)

                loss = nn.MSELoss()(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            episode_reward += reward

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Log progress
        pbar.update(1)
        if episode % 100 == 0:
            avg_reward = evaluate_policy(env, q_network, device, 10)
            pbar.set_description(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
    pbar.close()

    return q_network

# Evaluate policy
def evaluate_policy(env, q_network, device, num_episodes=10):
    total_reward = 0
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        while not done:
            state_one_hot = np.eye(env.observation_space.n)[state]
            state_tensor = torch.tensor(state_one_hot, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(q_network(state_tensor)).item()
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes

# Save demo as GIF
def demo_agent_save_gif(env, q_network, device, filename="frozenlake_dqn_demo.gif", num_episodes=1):
    """
    Runs the DQN agent and saves the entire sequence as a GIF.

    Parameters:
        env: Gym environment (with render_mode='rgb_array').
        q_network: The trained Q-network.
        device: Device for inference.
        filename: Name of the output GIF file.
        num_episodes: Number of episodes to record.
    """
    frames = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        while not done:
            # Render environment and store the frame
            frame = env.render()
            frames.append(frame)

            # Choose action based on policy
            state_one_hot = np.eye(env.observation_space.n)[state]
            state_tensor = torch.tensor(state_one_hot, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(q_network(state_tensor)).item()
            state, _, done, _, _ = env.step(action)

        # Add the last frame multiple times to show terminal state
        for _ in range(5):  # Repeat the last frame to simulate a pause
            frames.append(env.render())

    # Save the frames as a GIF
    imageio.mimsave(filename, frames, fps=5)
    print(f"Demo saved as {filename}")

def main():
    env = gym.make("FrozenLake-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000

    # Train the agent using DQN
    q_network = deep_q_learning(env, num_episodes, device=device)

    # Evaluate the trained policy
    avg_reward = evaluate_policy(env, q_network, device, num_episodes=100)
    print(f"Average reward after Deep Q-learning: {avg_reward:.2f}")

    # Save a demo of the agent as a GIF
    visual_env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    demo_agent_save_gif(visual_env, q_network, device, filename="Demo/deep_qlearning_agent.gif", num_episodes=1)

if __name__ == "__main__":
    main()
