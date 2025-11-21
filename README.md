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
## Screenshots
<img width="1207" height="815" alt="Screenshot 2025-11-20 at 7 10 24 PM" src="https://github.com/user-attachments/assets/20148e95-d9d7-4ed9-a6bc-e3c3685eaa1b" />
<img width="669" height="626" alt="Screenshot 2025-11-20 at 7 10 30 PM" src="https://github.com/user-attachments/assets/f3adcefd-ccbe-40ce-a6c8-e8e7adbf152e" />

