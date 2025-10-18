from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import time


# Create the environment
# env = make_vec_env('LunarLander-v2', n_envs=16)
# obs = env.reset()

model = PPO.load("ppo-LunarLander-v2-2")
vis_env = gym.make("LunarLander-v2", render_mode='human')
    
def watch_agent_live(model, episodes=3):
    """
    Watch the agent play with visual rendering (opens a window)
    
    Args:
        model: Trained PPO model
        episodes: Number of episodes to show
    """
    print("\n🎬 Opening live visualization window...")
    print("Close the window when you're done watching!")
    
    # Create environment with human rendering (opens a window)
    
    try:
        for episode in range(episodes):
            print(f"\nStarting episode {episode + 1}...")
            
            obs, info = vis_env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done:
                # Use the trained model to predict actions
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = vis_env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Check if episode is done
                done = terminated or truncated
                
                # Add small delay for better visualization
                time.sleep(0.02)
            
            print(f"Episode {episode + 1}: {step_count} steps, reward: {episode_reward:.2f}")
            
            # Small pause between episodes
            time.sleep(1)
    
    finally:
        vis_env.close()
        print("Visualization window closed!")


watch_agent_live(model)