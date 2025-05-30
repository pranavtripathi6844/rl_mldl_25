"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env.custom_hopper import *

def main():
    # Create the training environment
    train_env = gym.make('CustomHopper-source-v0')
    train_env = DummyVecEnv([lambda: train_env])

    # Create evaluation environment
    eval_env = gym.make('CustomHopper-source-v0')
    eval_env = DummyVecEnv([lambda: eval_env])

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.envs[0].get_parameters())  # masses of each link of the Hopper

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Initialize SAC agent
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
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

    # Calculate total timesteps for 60,000 episodes
    # Each episode has max 500 steps (defined in environment registration)
    n_episodes = 60000
    max_steps_per_episode = 500
    total_timesteps = n_episodes * max_steps_per_episode  # 30,000,000 timesteps

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=1000
    )

    # Save the final model
    model.save("sac_hopper")

if __name__ == '__main__':
    main()