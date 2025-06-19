"""Script for training a control policy using SimOpt for adaptive domain randomization."""
import gym
import numpy as np
import argparse
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from env.custom_hopper import *
from simopt import SimOpt

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC with SimOpt on Hopper environment')
    parser.add_argument('--episodes', type=int, default=10000,
                      help='Number of episodes per optimization iteration (default: 10000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Learning rate (default: 0.0003)')
    parser.add_argument('--n-initial-points', type=int, default=5,
                      help='Number of initial random points (default: 5)')
    parser.add_argument('--n-iterations', type=int, default=20,
                      help='Number of optimization iterations (default: 20)')
    parser.add_argument('--eval-episodes', type=int, default=50,
                      help='Number of episodes for evaluation (default: 50)')
    return parser.parse_args()

def train_with_params(params, args):
    """Train model with specific randomization parameters."""
    # Create training environment with specified parameters
    train_env = gym.make('CustomHopper-source-v0',
                        use_udr=True,
                        mass_ranges=params)
    train_env = DummyVecEnv([lambda: train_env])

    # Create evaluation environment
    eval_env = gym.make('CustomHopper-source-v0',
                       use_udr=True,
                       mass_ranges=params)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_simopt",
        log_path="./logs_simopt/",
        eval_freq=1000,
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
        tensorboard_log="./logs_simopt/",
        verbose=1,
        device="cuda"
    )

    # Calculate total timesteps
    max_steps_per_episode = 500
    total_timesteps = args.episodes * max_steps_per_episode

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=1000
    )

    return model

def evaluate_on_target(model, n_episodes=50):
    """Evaluate model on target environment."""
    # Create target environment
    env = gym.make('CustomHopper-target-v0')
    env = DummyVecEnv([lambda: env])

    # Evaluate
    episode_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        
        episode_rewards.append(total_reward)
    
    # Calculate mean reward
    mean_reward = np.mean(episode_rewards)
    return mean_reward

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create directories
    os.makedirs("./best_model_simopt", exist_ok=True)
    os.makedirs("./logs_simopt", exist_ok=True)
    
    # Define initial parameter ranges
    initial_ranges = {
        'thigh': (0.0, 0.3),  # ±30% variation
        'leg': (0.0, 0.3),
        'foot': (0.0, 0.3)
    }
    
    # Initialize SimOpt
    simopt = SimOpt(
        param_ranges=initial_ranges,
        n_initial_points=args.n_initial_points,
        n_iterations=args.n_iterations,
        save_dir="./simopt_results"
    )
    
    # Run optimization
    best_params = simopt.optimize(
        train_fn=lambda params: train_with_params(params, args),
        eval_fn=lambda model: evaluate_on_target(model, args.eval_episodes)
    )
    
    print("\nOptimization complete!")
    print(f"Best parameters found: {best_params}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = train_with_params(best_params, args)
    
    # Save final model
    final_model.save(os.path.join("./best_model_simopt", "final_model"))
    print("Final model saved!")

if __name__ == '__main__':
    main() 