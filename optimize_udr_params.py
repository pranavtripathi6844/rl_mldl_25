"""Optuna-based hyperparameter optimization for UDR (Uniform Domain Randomization) on Hopper environment"""
import optuna
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
    parser = argparse.ArgumentParser(description='Optimize UDR parameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=30,
                      help='Number of optimization trials (default: 30)')
    parser.add_argument('--optimization_episodes', type=int, default=300,
                      help='Number of episodes per trial for optimization (default: 300)')
    parser.add_argument('--timeout', type=int, default=3600,
                      help='Optimization timeout in seconds (default: 3600)')
    parser.add_argument('--output_file', type=str, default='best_udr_params.json',
                      help='Output file for best parameters (default: best_udr_params.json)')
    return parser.parse_args()

def objective(trial, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # UDR-specific parameters
    mass_variation = trial.suggest_float('mass_variation', 0.1, 0.5)  # e.g., 10% to 50%
    # Optionally, you could add per-link scaling if desired
    # thigh_scale = trial.suggest_float('thigh_scale', 0.7, 1.3)
    # leg_scale = trial.suggest_float('leg_scale', 0.7, 1.3)
    # foot_scale = trial.suggest_float('foot_scale', 0.7, 1.3)
    mass_ranges = {
        'thigh': (1-mass_variation, 1+mass_variation),
        'leg': (1-mass_variation, 1+mass_variation),
        'foot': (1-mass_variation, 1+mass_variation)
    }
    train_env = gym.make('CustomHopper-source-v0', use_udr=True, mass_ranges=mass_ranges)
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = gym.make('CustomHopper-source-v0', use_udr=True, mass_ranges=mass_ranges)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path=None,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
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
        tensorboard_log=None,
        verbose=0,
        device=device,
        policy_kwargs={"net_arch": [256, 256]}
    )
    max_steps_per_episode = 500
    total_timesteps = args.optimization_episodes * max_steps_per_episode
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=1000
    )
    # Evaluate
    rewards = []
    for _ in range(10):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_reward += reward[0]
        rewards.append(total_reward)
    mean_reward = np.mean(rewards)
    train_env.close()
    eval_env.close()
    return mean_reward

def main():
    args = parse_args()
    study = optuna.create_study(direction='maximize')
    print("Starting UDR parameter optimization with Optuna...")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, timeout=args.timeout)
    print("\nOptimization complete!")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best reward achieved: {study.best_trial.value:.2f}")
    print(f"Best parameters: {study.best_trial.params}")
    with open(args.output_file, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print(f"Best parameters saved to: {args.output_file}")

if __name__ == '__main__':
    main() 