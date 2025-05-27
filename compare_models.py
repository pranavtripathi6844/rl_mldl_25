"""Compare two RL agents on the OpenAI Gym Hopper environment with visualization"""
import argparse
import numpy as np
import torch
import gym
import time

from env.custom_hopper import *
from agent import Agent, Policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', default='model_no_baseline.mdl', type=str, help='Path to first model (no baseline)')
    parser.add_argument('--model2', default='model.mdl', type=str, help='Path to second model (with baseline)')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--episodes', default=3, type=int, help='Number of test episodes')
    parser.add_argument('--delay', default=2, type=int, help='Delay between models in seconds')

    return parser.parse_args()

def evaluate_model(model_path, env, device, episodes, render=True):
    policy = Policy(env.observation_space.shape[-1], env.action_space.shape[-1])
    policy.load_state_dict(torch.load(model_path), strict=True)
    agent = Agent(policy, device=device)
    
    returns = []
    
    for episode in range(episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            
            if render:
                env.render()
                
            test_reward += reward
            
        returns.append(test_reward)
        print(f"Model: {model_path} | Episode: {episode} | Return: {test_reward}")
    
    return returns

def main():
    args = parse_args()
    
    env = gym.make('CustomHopper-source-v0')
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())
    
    print("\nEvaluating model without baseline...")
    returns_no_baseline = evaluate_model(args.model1, env, args.device, args.episodes)
    
    print(f"\nWaiting {args.delay} seconds before showing the next model...")
    time.sleep(args.delay)
    
    print("\nEvaluating model with baseline...")
    returns_with_baseline = evaluate_model(args.model2, env, args.device, args.episodes)
    
    # Calculate and print statistics
    print("\nComparison Results:")
    print("-" * 50)
    print(f"Model without baseline:")
    print(f"  Mean return: {np.mean(returns_no_baseline):.2f}")
    print(f"  Std return: {np.std(returns_no_baseline):.2f}")
    print(f"  Min return: {np.min(returns_no_baseline):.2f}")
    print(f"  Max return: {np.max(returns_no_baseline):.2f}")
    
    print(f"\nModel with baseline:")
    print(f"  Mean return: {np.mean(returns_with_baseline):.2f}")
    print(f"  Std return: {np.std(returns_with_baseline):.2f}")
    print(f"  Min return: {np.min(returns_with_baseline):.2f}")
    print(f"  Max return: {np.max(returns_with_baseline):.2f}")

if __name__ == '__main__':
    main() 