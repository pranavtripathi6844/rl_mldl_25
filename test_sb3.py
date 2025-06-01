"""Script for evaluating the trained SAC model on the Hopper environment"""
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from env.custom_hopper import *
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test SAC on Hopper environment')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to test (default: 10)')
    parser.add_argument('--model', type=str, choices=['source', 'target'], default='source',
                      help='Which model to test (source or target)')
    parser.add_argument('--env', type=str, choices=['source', 'target'], default='source',
                      help='Which environment to test on (source or target)')
    return parser.parse_args()

def evaluate_policy(model, env, n_eval_episodes=10):
    """
    Evaluate a trained policy
    
    Args:
        model: Trained SAC model
        env: Environment to evaluate on
        n_eval_episodes: Number of episodes to evaluate
        
    Returns:
        mean_reward: Mean reward across episodes
        std_reward: Standard deviation of rewards
        all_rewards: List of all episode rewards
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
        print(f"Episode {episode + 1:3d} | Reward: {total_reward:8.2f} | Steps: {steps:4d}")
    
    return np.mean(episode_rewards), np.std(episode_rewards), episode_rewards

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create and vectorize the environment based on argument
    env_id = f'CustomHopper-{args.env}-v0'
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])
    
    try:
        # Set up model paths
        if args.model == 'source':
            model_path = os.path.join("best_model", "source_model.zip")
        else:  # target
            model_path = os.path.join("best_model_target", "best_model.zip")
            
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        try:
            model = SAC.load(model_path)
            print(f"\nSuccessfully loaded {args.model} model from: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
        # Print test configuration
        print("\nTest Configuration:")
        print("=" * 50)
        print(f"Model:    {args.model}")
        print(f"Env:      {args.env}")
        print(f"Episodes: {args.episodes}")
        print("=" * 50)
        print("\nStarting evaluation...\n")
        
        # Evaluate the model
        mean_reward, std_reward, all_rewards = evaluate_policy(
            model, env, n_eval_episodes=args.episodes
        )
        
        # Print detailed results
        print("\nEvaluation Results:")
        print("=" * 50)
        print(f"Model → Environment: {args.model} → {args.env}")
        print(f"Number of episodes: {args.episodes}")
        print(f"Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Min reward: {min(all_rewards):.2f}")
        print(f"Max reward: {max(all_rewards):.2f}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    finally:
        # Close the environment
        env.close()

if __name__ == '__main__':
    main() 