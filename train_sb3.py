"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import numpy as np
import argparse
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC on Hopper environment')
    parser.add_argument('--episodes', type=int, default=60000,
                      help='Number of episodes to train (default: 60000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Learning rate (default: 0.0003)')
    parser.add_argument('--use_udr', action='store_true',
                      help='Enable Uniform Domain Randomization')
    parser.add_argument('--mass_variation', type=float, default=0.3,
                      help='Mass variation range for UDR (e.g., 0.3 for ±30%)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Set up mass ranges for UDR if enabled
        mass_ranges = None
        if args.use_udr:
            mass_ranges = {
                'thigh': (1-args.mass_variation, 1+args.mass_variation),
                'leg': (1-args.mass_variation, 1+args.mass_variation),
                'foot': (1-args.mass_variation, 1+args.mass_variation)
            }

        # Create the training environment with UDR if enabled
        train_env = gym.make('CustomHopper-source-v0',
                           use_udr=args.use_udr,
                           mass_ranges=mass_ranges)
        train_env = DummyVecEnv([lambda: train_env])

        # Create evaluation environment with same UDR settings
        eval_env = gym.make('CustomHopper-source-v0',
                          use_udr=args.use_udr,
                          mass_ranges=mass_ranges)
        eval_env = DummyVecEnv([lambda: eval_env])

        print('State space:', train_env.observation_space)  # state-space
        print('Action space:', train_env.action_space)  # action-space
        print('Dynamics parameters:', train_env.envs[0].get_parameters())  # masses of each link of the Hopper
        if args.use_udr:
            print('UDR enabled with mass variation: ±{:.0f}%'.format(args.mass_variation * 100))

        # Create model directory with UDR indication
        model_dir = "./best_model_udr" if args.use_udr else "./best_model"
        os.makedirs(model_dir, exist_ok=True)

        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path="./logs/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )

        # Initialize SAC agent
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
            tensorboard_log="./logs/",
            verbose=1
        )

        # Calculate total timesteps based on command line argument
        max_steps_per_episode = 500
        total_timesteps = args.episodes * max_steps_per_episode

        print(f"Training for {args.episodes} episodes ({total_timesteps} timesteps)")
        if args.use_udr:
            print("Training with UDR - masses will be randomized each episode")

        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=1000
        )

        # Save the final model with UDR indication
        model_name = os.path.join(model_dir, "udr_model" if args.use_udr else "source_model")
        print(f"Saving final model as {model_name}...")
        model.save(model_name)
        print("Final model saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Clean up
        train_env.close()
        eval_env.close()

if __name__ == '__main__':
    main()