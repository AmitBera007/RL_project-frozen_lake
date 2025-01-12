import imageio
import numpy as np
import gym
from qlearning import q_learning, evaluate_policy

def demo_agent_save_gif(env, Q, filename="Demo/qlearning_agent.gif", num_episodes=1):
    """
    Runs the agent and saves the output as a GIF.

    Parameters:
        env: The gym environment (with render_mode='rgb_array').
        Q: The Q-table from training.
        filename: Name of the output GIF file.
        num_episodes: Number of episodes to record.
    """
    policy = np.argmax(Q, axis=1)
    frames = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # Render environment and store the frame
            frame = env.render()
            frames.append(frame)
            
            # Choose action based on the policy
            action = policy[state]
            state, reward, done, _, _ = env.step(action)

        # Add the last frame multiple times to show terminal state
        for _ in range(5):  # Repeat the last frame to simulate a pause
            frames.append(env.render())

    # Save the frames as a GIF
    imageio.mimsave(filename, frames, fps=5)
    print(f"Demo saved as {filename}")

def main():
    # Train the agent using Q-learning
    env = gym.make("FrozenLake-v1")
    num_episodes = 20000
    Q = q_learning(env, num_episodes)

    # Evaluate the trained policy
    avg_reward = evaluate_policy(env, Q, num_episodes=100)
    print(f"Average reward after training: {avg_reward:.2f}")

    # Print the Q-table 
    print("\nQ-table:") 
    print(Q)

    # Save a demo of the agent as a GIF
    visual_env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    demo_agent_save_gif(visual_env, Q, num_episodes=3)

if __name__ == "__main__":
    main()
