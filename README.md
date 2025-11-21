## Reinforcement Learning Agents (Value Iteration and Q-Learning)
This project implements reinforcement learning agents for Markov Decision Process (MDP) environments. It includes Value Iteration (model-based planning) and Q-Learning (model-free learning with exploration), along with analysis of parameter choices, convergence, and policy behavior.
## Features
Value Iteration Agent: Computes optimal state values through iterative Bellman updates and extracts the optimal policy.
Q-Learning Agent: Learns action-value estimates using temporal-difference updates with epsilon-greedy exploration.
Analysis File: Explains parameter effects, convergence patterns, and expected behaviors.
Gridworld Support: Compatible with the Gridworld MDP using the included utility files.
## Techniques Used
Markov Decision Processes (MDPs)
Value Iteration
Q-Learning
Epsilon-greedy exploration
Reward-based policy evaluation
Policy extraction and behavior analysis
## Included Files
valueIterationAgents.py — Value Iteration implementation
qlearningAgents.py — Q-Learning implementation
analysis.py — RL behavior explanations
mdp.py — MDP interface
environment.py — Agent-environment logic
gridworld.py — Gridworld environment
util.py — Utility functions
textGridworldDisplay.py and graphicsGridworldDisplay.py — Optional visualizers
## How to Run
Value Iteration
python gridworld.py -a value -i 100
## Q-Learning
python gridworld.py -a q
## Purpose
This repository demonstrates practical implementations of reinforcement learning algorithms and showcases understanding of policy computation, exploration strategies, and agent evaluation.
