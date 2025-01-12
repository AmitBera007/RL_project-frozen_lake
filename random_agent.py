import os
import gym
import imageio

def random_agent(env, observation):
    return env.action_space.sample()

def training(env, num_episodes):
    # Ensure the Results directory exists
    os.makedirs("Results", exist_ok=True)

    total_reward = 0

    # Open a file to log average reward after each episode
    with open("Results/random_agent_results.txt", "w") as f:
        f.write("Episode,Average_Reward\n")  # Header for the CSV file
        
        for i in range(1, num_episodes + 1):
            observation, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = random_agent(env, observation)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_reward += episode_reward
            avg_reward = total_reward / i  # Calculate average reward up to the ith episode
            
            # Write the average reward for the ith episode to the file
            f.write(f"{i},{avg_reward:.4f}\n")
            
            # Print the progress (optional)
            if i % 100 == 0:
                print(f"Episode {i}: Average Reward = {avg_reward:.4f}")

def demo_agent(env, num_episodes=1, gif_path="Demo/random_agent.gif"):
    # Ensure the Demo directory exists
    os.makedirs("Demo", exist_ok=True)

    frames = []  # List to store frames for GIF
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        print(f"\nEpisode {episode + 1}")
        
        while not done:
            # Render the environment and capture the frame
            frame = env.render()  # Render the environment
            frames.append(frame)  # Append the frame
            
            # Take an action
            action = random_agent(env, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Add the last frame multiple times to show terminal state
        for _ in range(10):  # Repeat the last frame to simulate a pause
            frames.append(env.render())
        print("Episode ended.")

    # Save frames as a GIF with a controlled frame rate
    imageio.mimsave(gif_path, frames, fps=5)  # Adjust FPS to slow down the GIF
    print(f"GIF saved to {gif_path}")


def main():
    # Create training environment
    env = gym.make("FrozenLake-v1", render_mode=None)
    num_episodes = 20000

    training(env, num_episodes)

    # Create visual environment for GIF generation
    visual_env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    demo_agent(visual_env, num_episodes=3, gif_path="Demo/random_agent.gif")

if __name__ == '__main__':
    main()
