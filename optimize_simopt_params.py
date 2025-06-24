"""Optuna-based hyperparameter optimization for SimOpt (adaptive domain randomization) on Hopper environment"""
import optuna
import argparse
import os
import json
from train_simopt import train_with_params, evaluate_on_target
from simopt import SimOpt

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize SimOpt parameters using Optuna')
    parser.add_argument('--n_trials', type=int, default=30,
                      help='Number of optimization trials (default: 30)')
    parser.add_argument('--episodes', type=int, default=2000,
                      help='Number of training episodes per trial (default: 2000)')
    parser.add_argument('--eval_episodes', type=int, default=50,
                      help='Number of evaluation episodes (default: 50)')
    parser.add_argument('--timeout', type=int, default=3600,
                      help='Optimization timeout in seconds (default: 3600)')
    parser.add_argument('--output_file', type=str, default='best_simopt_params.json',
                      help='Output file for best parameters (default: best_simopt_params.json)')
    return parser.parse_args()

def objective(trial, args):
    # SimOpt-specific parameters
    n_initial_points = trial.suggest_int('n_initial_points', 3, 10)
    n_iterations = trial.suggest_int('n_iterations', 5, 30)
    mass_variation = trial.suggest_float('mass_variation', 0.1, 0.5)
    # Prepare mass_ranges for SimOpt
    mass_ranges = {
        'thigh': (1-mass_variation, 1+mass_variation),
        'leg': (1-mass_variation, 1+mass_variation),
        'foot': (1-mass_variation, 1+mass_variation)
    }
    # SimOpt instance
    simopt = SimOpt(
        param_ranges=mass_ranges,
        n_initial_points=n_initial_points,
        n_iterations=n_iterations,
        save_dir="./simopt_results_optuna_trial"
    )
    # Use train_with_params and evaluate_on_target from train_simopt.py
    best_params = simopt.optimize(
        train_fn=lambda params: train_with_params(params, args),
        eval_fn=lambda model: evaluate_on_target(model, args.eval_episodes)
    )
    # After optimization, train a final model and evaluate
    final_model = train_with_params(best_params, args)
    mean_reward = evaluate_on_target(final_model, args.eval_episodes)
    return mean_reward

def main():
    args = parse_args()
    print("Starting SimOpt parameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
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