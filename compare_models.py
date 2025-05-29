"""Compare three RL agents on the OpenAI Gym Hopper environment with visualization"""
import argparse
import numpy as np
import torch
import gym
import time

from env.custom_hopper import *
from agent import Agent, Policy
from agent_actor_critic import Agent as ActorCriticAgent, Policy as Value

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', default='model_no_baseline.mdl', type=str, help='Path to first model (no baseline)')
    parser.add_argument('--model2', default='model_with_baseline.mdl', type=str, help='Path to second model (with baseline)')
    parser.add_argument('--model3', default='actor_critic.mdl', type=str, help='Path to third model (actor-critic)')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--episodes', default=3, type=int, help='Number of test episodes')
    parser.add_argument('--delay', default=2, type=int, help='Delay between models in seconds')
    parser.add_argument('--render', action='store_true', help='Enable rendering')

    return parser.parse_args()

def evaluate_model(model_path, env, device, episodes, render=True, method='baseline'):
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    if method == 'actor_critic':
        policy = Value(observation_space_dim, action_space_dim)
        policy.load_state_dict(torch.load(model_path), strict=True)
        agent = ActorCriticAgent(policy, device=device)
    else:
        policy = Policy(observation_space_dim, action_space_dim)
        policy.load_state_dict(torch.load(model_path), strict=True)
        baseline = 20.0 if method == 'baseline' else 0.0
        agent = Agent(policy, device=device, baseline=baseline)
    
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
    returns_no_baseline = evaluate_model(args.model1, env, args.device, args.episodes, args.render, 'no_baseline')
    
    print(f"\nWaiting {args.delay} seconds before showing the next model...")
    time.sleep(args.delay)
    
    print("\nEvaluating model with baseline...")
    returns_with_baseline = evaluate_model(args.model2, env, args.device, args.episodes, args.render, 'baseline')
    
    print(f"\nWaiting {args.delay} seconds before showing the next model...")
    time.sleep(args.delay)
    
    print("\nEvaluating actor-critic model...")
    returns_actor_critic = evaluate_model(args.model3, env, args.device, args.episodes, args.render, 'actor_critic')
    
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
    
    print(f"\nActor-Critic model:")
    print(f"  Mean return: {np.mean(returns_actor_critic):.2f}")
    print(f"  Std return: {np.std(returns_actor_critic):.2f}")
    print(f"  Min return: {np.min(returns_actor_critic):.2f}")
    print(f"  Max return: {np.max(returns_actor_critic):.2f}")

if __name__ == '__main__':
    main() 