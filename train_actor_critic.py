"""Train an RL agent on the OpenAI Gym Hopper environment using Actor-Critic"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import ActorCriticAgent, Policy, Value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())
    print("Training Actor-Critic")

    """
        Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    value = Value(observation_space_dim)
    agent = ActorCriticAgent(policy, value, device=args.device)

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            train_reward += reward
        
        # Update policy and value after each episode
        policy_loss, value_loss = agent.update()
        
        if (episode+1)%args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)
            print('Policy loss:', policy_loss)
            print('Value loss:', value_loss)

    torch.save(agent.policy.state_dict(), "model_actor_critic.mdl")

if __name__ == '__main__':
    main()