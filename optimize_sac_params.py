"""SAC Hyperparameter Optimization using Optuna
   This script finds the best hyperparameters for SAC and saves them for use in training scripts.
"""
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
    parser = argparse.ArgumentParser(description='Optimize SAC hyperparameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=50,
                      help='Number of optimization trials (default: 50)')
    parser.add_argument('--optimization_episodes', type=int, default=200,
                      help='Episodes per optimization trial (default: 200)')
    parser.add_argument('--timeout', type=int, default=3600,
                      help='Optimization timeout in seconds (default: 3600)')
    parser.add_argument('--use_udr', action='store_true',
                      help='Enable Uniform Domain Randomization during optimization')
    parser.add_argument('--mass_variation', type=float, default=0.3,
                      help='Mass variation range for UDR (e.g., 0.3 for ±30%)')
    parser.add_argument('--env_type', type=str, choices=['source', 'target'], default='source',
                      help='Environment type to optimize for (default: source)')
    parser.add_argument('--output_file', type=str, default='best_sac_params.json',
                      help='Output file for best parameters (default: best_sac_params.json)')
    return parser.parse_args()

def objective(trial, args):
    """Objective function for Optuna optimization"""
    
    # Auto-detect device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define hyperparameter search space with enhanced ranges
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)  # Expanded range
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024, 2048])  # More options
    buffer_size = trial.suggest_categorical('buffer_size', [100000, 500000, 1000000, 2000000])
    tau = trial.suggest_float('tau', 0.001, 0.02)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    train_freq = trial.suggest_categorical('train_freq', [1, 2, 4])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 2, 4])
    learning_starts = trial.suggest_categorical('learning_starts', [500, 1000, 2000])
    target_update_interval = trial.suggest_categorical('target_update_interval', [1, 2, 4])
    
    # Add steps per episode parameter
    steps_per_episode = trial.suggest_categorical('steps_per_episode', [300, 500, 750, 1000])
    
    # Network architecture parameters
    net_arch = trial.suggest_categorical('net_arch', [
        [64, 64],
        [128, 128], 
        [256, 256],
        [64, 128, 64],
        [128, 256, 128],
        [256, 512, 256],  # Added larger network
        [512, 512]        # Added larger network
    ])
    
    # Entropy coefficient with more options
    ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    
    # Set up mass ranges for UDR if enabled
    mass_ranges = None
    if args.use_udr:
        mass_ranges = {
            'thigh': (1-args.mass_variation, 1+args.mass_variation),
            'leg': (1-args.mass_variation, 1+args.mass_variation),
            'foot': (1-args.mass_variation, 1+args.mass_variation)
        }
    
    # Create environment based on type
    if args.env_type == 'source':
        train_env = gym.make('CustomHopper-source-v0',
                           use_udr=args.use_udr,
                           mass_ranges=mass_ranges)
        eval_env = gym.make('CustomHopper-source-v0',
                          use_udr=args.use_udr,
                          mass_ranges=mass_ranges)
    else:  # target
        train_env = gym.make('CustomHopper-target-v0')
        eval_env = gym.make('CustomHopper-target-v0')
    
    train_env = DummyVecEnv([lambda: train_env])
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./optuna_trials/trial_{trial.number}",
        log_path=f"./optuna_logs/trial_{trial.number}",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    try:
        # Initialize SAC with trial parameters
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy='auto',
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            tensorboard_log=f"./optuna_tensorboard/trial_{trial.number}",
            verbose=0,
            device=device,
            policy_kwargs={"net_arch": net_arch}
        )
        
        # Train for evaluation using steps_per_episode parameter
        total_timesteps = args.optimization_episodes * steps_per_episode
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=1000
        )
        
        # Evaluate the model
        mean_reward = evaluate_model(model, eval_env, n_eval_episodes=20)
        
        # Report intermediate value for pruning
        trial.report(mean_reward, step=total_timesteps)
        
        return mean_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('-inf')
    
    finally:
        train_env.close()
        eval_env.close()

def evaluate_model(model, env, n_eval_episodes=20):
    """Evaluate model and return mean reward"""
    episode_rewards = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
        
        episode_rewards.append(total_reward)
    
    return np.mean(episode_rewards)

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs("./optuna_trials", exist_ok=True)
    os.makedirs("./optuna_logs", exist_ok=True)
    os.makedirs("./optuna_tensorboard", exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Print optimization settings
    print("="*60)
    print("SAC HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Environment: {args.env_type}")
    print(f"Trials: {args.n_trials}")
    print(f"Episodes per trial: {args.optimization_episodes}")
    print(f"Timeout: {args.timeout} seconds")
    if args.use_udr:
        print(f"UDR enabled with mass variation: ±{args.mass_variation*100:.0f}%")
    print(f"Output file: {args.output_file}")
    print("="*60)
    
    # Run optimization
    print(f"\nStarting optimization...")
    study.optimize(
        lambda trial: objective(trial, args), 
        n_trials=args.n_trials, 
        timeout=args.timeout
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best reward achieved: {study.best_trial.value:.2f}")
    
    print(f"\nBest hyperparameters:")
    for param, value in study.best_trial.params.items():
        print(f"  {param}: {value}")
    
    # Save best parameters
    with open(args.output_file, "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    
    print(f"\n✅ Best parameters saved to: {args.output_file}")
    
    # Print usage instructions
    print(f"\n📋 USAGE INSTRUCTIONS:")
    print(f"To use these optimized parameters in your training scripts:")
    print(f"")
    print(f"1. For source environment:")
    print(f"   python train_sb3.py --episodes 2000 --load_best_params {args.output_file}")
    print(f"")
    print(f"2. For target environment:")
    print(f"   python train_sb3_target.py --episodes 2000 --load_best_params {args.output_file}")
    print(f"")
    print(f"3. For UDR training:")
    print(f"   python train_sb3.py --episodes 2000 --use_udr --load_best_params {args.output_file}")
    
    # Print top 5 trials for reference
    print(f"\n🏆 TOP 5 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    for i, trial in enumerate(trials[:5]):
        print(f"{i+1}. Trial {trial.number}: {trial.value:.2f}")
    
    return study.best_trial.params

if __name__ == '__main__':
    main()
