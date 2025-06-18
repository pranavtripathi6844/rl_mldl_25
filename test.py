"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym
import numpy as np

from env.custom_hopper import *
from agent import Agent, Policy
from agent_actor_critic import Agent as ActorCriticAgent, Policy as Value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')
    parser.add_argument('--method', default='baseline', type=str, choices=['no_baseline', 'baseline', 'actor_critic'], 
                      help='Method to use: no_baseline, baseline, or actor_critic')

    return parser.parse_args()

args = parse_args()


def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    if args.method == 'actor_critic':
        policy = Value(observation_space_dim, action_space_dim)
        policy.load_state_dict(torch.load(args.model), strict=True)
        agent = ActorCriticAgent(policy, device=args.device)
    else:
        policy = Policy(observation_space_dim, action_space_dim)
        policy.load_state_dict(torch.load(args.model), strict=True)
        baseline = 20.0 if args.method == 'baseline' else 0.0
        agent = Agent(policy, device=args.device, baseline=baseline)

    episode_returns = []  # Store returns for all episodes

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        episode_returns.append(test_reward)
        print(f"Episode: {episode} | Return: {test_reward}")

    # Print summary statistics
    print("\nSummary of Results:")
    print("=" * 40)
    print(f"Average reward: {np.mean(episode_returns):.2f}")
    print(f"Std deviation:  {np.std(episode_returns):.2f}")
    print(f"Min reward:     {np.min(episode_returns):.2f}")
    print(f"Max reward:     {np.max(episode_returns):.2f}")
    print(f"Best episode:   {np.argmax(episode_returns)} (Reward: {np.max(episode_returns):.2f})")
    print("=" * 40)


if __name__ == '__main__':
    main()