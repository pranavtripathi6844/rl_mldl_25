"""Script for evaluating the trained SAC model on the Hopper environment"""
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from env.custom_hopper import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Test SAC on Hopper environment')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to test (default: 10)')
    return parser.parse_args()

def evaluate_policy(model, env, n_eval_episodes=10):
    """
    Evaluate a trained policy
    
    Args:
        model: Trained SAC model
        env: Environment to evaluate on
        n_eval_episodes: Number of episodes to evaluate
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get deterministic action (no exploration)
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, info = env.step(action)
            
            total_reward += reward[0]  # [0] because env is vectorized
            steps += 1
            
            # Render the environment
            env.render()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}")
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Steps:  {steps}")
    
    return np.mean(episode_rewards), np.std(episode_rewards)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create and vectorize the environment
    env = gym.make('CustomHopper-source-v0')
    env = DummyVecEnv([lambda: env])
    
    try:
        # First try to load the best model
        try:
            model = SAC.load("best_model/best_model")
            print("Loaded best model from training!")
        except:
            # If best model not found, try loading the final model
            try:
                model = SAC.load("sac_hopper")
                print("Loaded final model from training!")
            except:
                raise Exception("No trained model found! Please train the model first.")
        
        # Evaluate the model
        print("\nStarting evaluation...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.episodes)
        
        # Print results
        print("\nEvaluation Results:")
        print("=" * 50)
        print(f"Average Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Close the environment
        env.close()

if __name__ == '__main__':
    main() 