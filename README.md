# Sim-to-Real Reinforcement Learning with Hopper

This project explores the challenge of transferring reinforcement learning (RL) policies from simulation to the real world, using the MuJoCo Hopper environment as a testbed. The core objective is to understand how different RL algorithms and training strategies affect an agent’s ability to generalize when faced with changes in environment dynamics—mimicking the "sim-to-real gap" encountered in robotics.

We implement and compare several RL methods, including REINFORCE (with and without baseline), Actor-Critic, Soft Actor-Critic (SAC), Uniform Domain Randomization (UDR), and SimOpt (Sim-to-Real Optimization). Each algorithm is trained in a simulated "source" environment and evaluated in a "target" environment with altered physical parameters, such as mass and friction, to simulate real-world variability.

The project investigates how domain randomization (UDR) and adaptive simulation parameter optimization (SimOpt) can improve policy robustness and transferability. We also employ Optuna for automated hyperparameter tuning, aiming to maximize performance in the target domain.

Comprehensive experiments are conducted to compare the effectiveness of each approach, with results analyzed in terms of average return, stability, and adaptability. The findings provide insights into which RL strategies are most effective for sim-to-real transfer, and highlight the trade-offs between robustness, sample efficiency, and final performance.

This work serves as a practical guide for researchers and practitioners interested in deploying RL agents in real-world scenarios, where simulation-to-reality discrepancies are a major challenge.
