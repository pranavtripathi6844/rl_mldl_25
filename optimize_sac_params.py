"""Optuna-based hyperparameter optimization for SAC on Hopper environment"""
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
    parser.add_argument('--optimization_episodes', type=int, default=300,
                      help='Number of episodes per trial for optimization (default: 300)')
    parser.add_argument('--env_type', type=str, choices=['source', 'target'], default='source',
                      help='Environment type to optimize for (default: source)')
    parser.add_argument('--timeout', type=int, default=3600,
                      help='Optimization timeout in seconds (default: 3600)')
    parser.add_argument('--use_udr', action='store_true',
                      help='Enable Uniform Domain Randomization during optimization')
    parser.add_argument('--mass_variation', type=float, default=0.3,
                      help='Mass variation range for UDR (default: 0.3)')
    parser.add_argument('--output_file', type=str, default='best_sac_params.json',
                      help='Output file for best parameters (default: best_sac_params.json)')
    return parser.parse_args()

def objective(trial, args):
    """Objective function for Optuna optimization"""
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Updated hyperparameter search spaces based on user request
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
    gamma = trial.suggest_categorical('gamma', [0.99, 0.995, 0.999])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 3e-3, 5e-3, 1e-3, 3e-4, 5e-4])
    buffer_size = trial.suggest_categorical('buffer_size', [100_000, 500_000, 1_000_000])
    tau = trial.suggest_categorical('tau', [0.001, 0.005, 0.01])
    train_freq = trial.suggest_categorical('train_freq', [1, 2, 4])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 2, 4])
    learning_starts = trial.suggest_categorical('learning_starts', [500, 1000, 2000])
    target_update_interval = trial.suggest_categorical('target_update_interval', [1, 2, 4])
    net_arch = trial.suggest_categorical('net_arch', [
        [64, 64],
        [128, 128], 
        [256, 256],
        [64, 128, 64],
        [128, 256, 128],
        [256, 512, 256],
        [512, 512]
    ])
    ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    
    # Fixed steps per episode (environment determines actual episode length)
    steps_per_episode = 500  # Fixed value, environment will determine actual length
    
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
    
    # Create directories for this trial
    trial_dir = f"./optuna_trials/trial_{trial.number}"
    log_dir = f"./optuna_logs/trial_{trial.number}"
    tensorboard_dir = f"./optuna_tensorboard/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=trial_dir,
        log_path=log_dir,
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
            tensorboard_log=tensorboard_dir,
            verbose=0,
            device=device,
            policy_kwargs={"net_arch": net_arch}
        )
        
        # Train for evaluation using steps_per_episode parameter
        total_timesteps = args.optimization_episodes * steps_per_episode
        
        # Report intermediate values more frequently for better pruning
        for step in range(10000, total_timesteps + 1, 10000):
            # Train for this step
        model.learn(
                total_timesteps=step,
            callback=eval_callback,
                log_interval=1000,
                reset_num_timesteps=False
        )
        
            # Evaluate and report
            mean_reward = evaluate_model(model, eval_env, n_eval_episodes=10)
            trial.report(mean_reward, step=step)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation with more episodes
        mean_reward = evaluate_model(model, eval_env, n_eval_episodes=20)
        
        # Save the final model for this trial
        final_model_path = os.path.join(trial_dir, f"final_model_trial_{trial.number}.zip")
        model.save(final_model_path)
        
        # Save trial parameters for reference
        trial_params = {
            'trial_number': trial.number,
            'mean_reward': mean_reward,
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'tau': tau,
            'gamma': gamma,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'learning_starts': learning_starts,
            'target_update_interval': target_update_interval,
            'net_arch': net_arch,
            'ent_coef': ent_coef,
            'model_path': final_model_path
        }
        
        with open(os.path.join(trial_dir, f"trial_{trial.number}_params.json"), 'w') as f:
            json.dump(trial_params, f, indent=4)
        
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
    
    # Create study with less aggressive pruner
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,    # Wait 10 trials before pruning (vs 5 default)
            n_warmup_steps=20000,   # Wait 20k steps before pruning (vs 0 default)
            interval_steps=10000    # Check every 10k steps (vs 1 default)
        )
    )
    
    # Print optimization settings
    print("="*60)
    print("SAC HYPERPARAMETER OPTIMIZATION (IMPROVED)")
    print("="*60)
    print(f"Environment: {args.env_type}")
    print(f"Trials: {args.n_trials}")
    print(f"Episodes per trial: {args.optimization_episodes}")
    print(f"Timeout: {args.timeout} seconds")
    if args.use_udr:
        print(f"UDR enabled with mass variation: ±{args.mass_variation*100:.0f}%")
    print(f"Output file: {args.output_file}")
    print("Pruner: MedianPruner (less aggressive)")
    print("  - Startup trials: 10")
    print("  - Warmup steps: 20,000")
    print("  - Interval steps: 10,000")
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
    
    # Copy best model to a standard location
    best_trial_dir = f"./optuna_trials/trial_{study.best_trial.number}"
    best_model_path = os.path.join(best_trial_dir, f"final_model_trial_{study.best_trial.number}.zip")
    if os.path.exists(best_model_path):
        import shutil
        best_model_standard = f"./best_model_optuna_{args.env_type}.zip"
        shutil.copy2(best_model_path, best_model_standard)
        print(f"✅ Best model copied to: {best_model_standard}")
    
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
    print(f"")
    print(f"4. Test the best model directly:")
    print(f"   python test_sb3.py --episodes 50 --model_path {best_model_standard}")
    
    # Print top 5 trials for reference
    print(f"\n🏆 TOP 5 TRIALS:")
    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    for i, trial in enumerate(trials[:5]):
        print(f"{i+1}. Trial {trial.number}: {trial.value:.2f}")
        if trial.number == study.best_trial.number:
            print(f"    ⭐ BEST TRIAL")
    
    # Save summary of all trials
    trials_summary = []
    for trial in study.trials:
        if trial.value is not None:  # Skip pruned trials
            trials_summary.append({
                'trial_number': trial.number,
                'reward': trial.value,
                'params': trial.params,
                'model_path': f"./optuna_trials/trial_{trial.number}/final_model_trial_{trial.number}.zip"
            })
    
    # Sort by reward
    trials_summary.sort(key=lambda x: x['reward'], reverse=True)
    
    with open(f"trials_summary_{args.env_type}.json", 'w') as f:
        json.dump(trials_summary, f, indent=4)
    
    print(f"\n📊 All trials summary saved to: trials_summary_{args.env_type}.json")
    print(f"📁 All trial models saved in: ./optuna_trials/")
    
    return study.best_trial.params

if __name__ == '__main__':
    main()
