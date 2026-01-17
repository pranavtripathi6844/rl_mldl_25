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


class PPOStylePruner(optuna.pruners.BasePruner):
    """
    Custom pruner based on PPO's TrialEvalCallback strategy.
    Uses dynamic tolerance and dual-condition pruning for robust trial management.
    
    Adapted from PPO optimization code to work with SAC's off-policy learning characteristics.
    """
    def __init__(self, initial_tolerance=50, tolerance_multiplier=0.1, patience=1, n_warmup_steps=2):
        """
        Args:
            initial_tolerance: Initial absolute tolerance for reward decline
            tolerance_multiplier: Factor to compute dynamic tolerance (e.g., 0.1 = 10% of previous reward)
            patience: Number of tolerance violations allowed before pruning (default: 1)
            n_warmup_steps: Number of initial evaluations before pruning starts
        """
        self.initial_tolerance = initial_tolerance
        self.tolerance_multiplier = tolerance_multiplier
        self.patience = patience
        self.n_warmup_steps = n_warmup_steps
        # Store violation counts per trial
        self.violation_counts = {}
    
    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """
        Determine whether to prune based on PPO-style dual-condition logic:
        1. Tolerance-based: Allow limited violations of dynamic tolerance
        2. Two-step decline: Prune if worse than both of last 2 evaluations
        """
        # Get intermediate values for this trial
        intermediate_values = trial.intermediate_values
        
        # Need at least n_warmup_steps + 1 evaluations before pruning
        if len(intermediate_values) <= self.n_warmup_steps:
            return False
        
        # Get sorted steps
        steps = sorted(intermediate_values.keys())
        
        # Need at least 3 evaluations for dual-condition check
        if len(steps) < 3:
            return False
        
        # Get current and previous values
        current_value = intermediate_values[steps[-1]]
        prev_value = intermediate_values[steps[-2]]
        even_prev_value = intermediate_values[steps[-3]] if len(steps) >= 3 else prev_value
        
        # Initialize violation counter for this trial if not exists
        trial_id = trial.number
        if trial_id not in self.violation_counts:
            self.violation_counts[trial_id] = 0
        
        # === Condition 1: Tolerance-based pruning (with patience) ===
        # Dynamic tolerance: max(initial_tolerance, tolerance_multiplier × prev_reward)
        dynamic_tolerance = max(self.initial_tolerance, self.tolerance_multiplier * abs(prev_value))
        
        if current_value < prev_value - dynamic_tolerance:
            self.violation_counts[trial_id] += 1
            
            if self.violation_counts[trial_id] > self.patience:
                print(f"  ✂️  Pruning trial {trial_id} (Tolerance violation #{self.violation_counts[trial_id]}): "
                      f"Current={current_value:.2f}, Prev={prev_value:.2f}, "
                      f"Decline={prev_value-current_value:.2f} > Tolerance={dynamic_tolerance:.2f}")
                return True
        
        # === Condition 2: Two-step decline ===
        # Prune if current is worse than BOTH of the last 2 evaluations
        if (current_value < even_prev_value) and (current_value < prev_value):
            print(f"  ✂️  Pruning trial {trial_id} (Two-step decline): "
                  f"Current={current_value:.2f} < Prev={prev_value:.2f} and "
                  f"EvenPrev={even_prev_value:.2f}")
            return True
        
        # Trial survives
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize SAC hyperparameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of optimization trials (default: 100)')
    # In this script, `optimization_episodes` represents the total number of
    # training timesteps per trial (e.g. 100_000), not Gym episodes.
    parser.add_argument('--optimization_episodes', type=int, default=100_000,
                      help='Number of training timesteps per trial (default: 100000)')
    parser.add_argument('--env_type', type=str, choices=['source', 'target'], default='target',
                      help='Environment type to optimize for (default: target)')
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
    
    # === Hyperparameter search space (reduced) ===
    # We only search over the most important SAC hyperparameters:
    #  - batch_size
    #  - buffer_size
    #  - gamma
    #  - learning_rate
    # All the remaining hyperparameters are kept at their SB3 defaults.
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_categorical('buffer_size', [100_000, 500_000, 1_000_000])
    gamma = trial.suggest_categorical('gamma', [0.99, 0.995, 0.999])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 3e-3, 5e-3, 1e-3, 3e-4, 5e-4])
    
    # In the updated procedure, `optimization_episodes` is interpreted as
    # the total number of *timesteps* per trial (e.g. 100_000), and we
    # evaluate every 10_000 timesteps for 50 evaluation episodes.
    total_timesteps = args.optimization_episodes
    eval_interval = 10_000
    
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
    
    try:
        # Initialize SAC with trial parameters. Only the searched
        # hyperparameters are overridden, everything else stays at SB3 defaults.
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            tensorboard_log=tensorboard_dir,
            verbose=0,
            device=device,
        )
        
        # === Training & evaluation schedule (matches report description) ===
        # - Train for `total_timesteps` (e.g. 100_000) per trial.
        # - Every `eval_interval` timesteps (e.g. 10_000), evaluate the policy
        #   for 50 episodes on the target environment and report to Optuna.
        # - Optuna's pruner can stop unpromising trials early.
        trained_steps = 0
        best_mean_reward = -float("inf")
        while trained_steps < total_timesteps:
            # Train for the next chunk of timesteps
            next_chunk = min(eval_interval, total_timesteps - trained_steps)
            model.learn(
                total_timesteps=next_chunk,
                log_interval=1000,
                reset_num_timesteps=False
            )
            trained_steps += next_chunk

            # Evaluate and report (50 episodes as in the description)
            mean_reward = evaluate_model(model, eval_env, n_eval_episodes=50)
            trial.report(mean_reward, step=trained_steps)

            # Simple improvement-based tracking (for analysis / logging)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward

            # Let Optuna decide whether to prune this trial
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Final evaluation (again 50 episodes for consistency)
        mean_reward = evaluate_model(model, eval_env, n_eval_episodes=50)
        
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
            'gamma': gamma,
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
    
    # Create study with PPO-style pruner (adapted for SAC)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=PPOStylePruner(
            initial_tolerance=50,      # Initial absolute tolerance
            tolerance_multiplier=0.1,  # Dynamic: 10% of previous reward
            patience=1,                 # Allow 1 violation before pruning
            n_warmup_steps=2           # Wait for 2 evaluations before pruning
        )
    )
    
    # Print optimization settings
    print("="*60)
    print("SAC HYPERPARAMETER OPTIMIZATION (PPO-STYLE PRUNING)")
    print("="*60)
    print(f"Environment: {args.env_type}")
    print(f"Trials: {args.n_trials}")
    print(f"Episodes per trial: {args.optimization_episodes}")
    print(f"Timeout: {args.timeout} seconds")
    if args.use_udr:
        print(f"UDR enabled with mass variation: ±{args.mass_variation*100:.0f}%")
    print(f"Output file: {args.output_file}")
    print("Pruner: PPOStylePruner (dual-condition with dynamic tolerance)")
    print("  - Initial tolerance: 50 (absolute)")
    print("  - Tolerance multiplier: 10% of previous reward")
    print("  - Patience: 1 violation allowed")
    print("  - Warmup steps: 2 evaluations")
    print("  - Two-step decline check: enabled")
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
