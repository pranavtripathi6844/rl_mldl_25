"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)
   This version trains on the target environment.
"""
import gym
import numpy as np
import argparse
import os
import json
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC agent on target environment using Stable Baselines3')
    parser.add_argument('--episodes', type=int, default=2000,
                      help='Number of training episodes (default: 2000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Learning rate (default: 0.0003)')
    parser.add_argument('--load_best_params', type=str, default=None,
                      help='Load best parameters from file (e.g., best_sac_params.json)')
    return parser.parse_args()

def load_best_params(params_file):
    """Load best parameters from file"""
    if not os.path.exists(params_file):
        print(f"Warning: {params_file} not found. Using default parameters.")
        return {}
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    print(f"Loaded optimized parameters from {params_file}:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    return params

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Auto-detect device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load optimized parameters if provided
    best_params = {}
    if args.load_best_params:
        best_params = load_best_params(args.load_best_params)
    
    # Create the training environment (using target environment)
    train_env = gym.make('CustomHopper-target-v0')
    train_env = DummyVecEnv([lambda: train_env])

    # Create evaluation environment (using target environment)
    eval_env = gym.make('CustomHopper-target-v0')
    eval_env = DummyVecEnv([lambda: eval_env])

    print('State space:', train_env.observation_space)
    print('Action space:', train_env.action_space)
    print('Dynamics parameters:', train_env.envs[0].get_parameters())

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_target",  # New directory for target model
        log_path="./logs_target/",                   # New directory for target logs
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Extract policy_kwargs if present in best_params
    policy_kwargs = {}
    if 'net_arch' in best_params:
        policy_kwargs['net_arch'] = best_params.pop('net_arch')
        print(f"Using optimized network architecture: {policy_kwargs['net_arch']}")

    # Initialize SAC agent with optimized parameters if available
    if best_params:
        # Use optimized parameters
        model = SAC(
            "MlpPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            device=device,
            **best_params
        )
        print("Using optimized hyperparameters")
    else:
        # Use default parameters
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        optimize_memory_usage=False,
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        tensorboard_log="./logs_target/",
            verbose=1,
            device=device
    )

    # Calculate total timesteps
    max_steps_per_episode = 500
    total_timesteps = args.episodes * max_steps_per_episode

    print(f"Training for {args.episodes} episodes ({total_timesteps} timesteps) on target environment")

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=1000
    )

    # Save the final model
    model_name = "./best_model_target/target_model"
    print(f"Saving final model as {model_name}...")
    model.save(model_name)
    print("Final model saved successfully!")

if __name__ == '__main__':
    main() 