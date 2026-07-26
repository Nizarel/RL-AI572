# Programming Assignment 3: Deep Q-Network for Acrobot

## Objective

Implement a DQN agent for the Acrobot environment, evaluate its learning behavior, and compare it with the other reinforcement-learning algorithms explored for Acrobot.

## Method

- Use a PyTorch multilayer perceptron to approximate the Q-function.
- Use experience replay and a target network to stabilize learning.
- Train using epsilon-greedy exploration on Acrobot-v1.

## Metrics

- Episode reward
- Moving-average reward
- Episode length
- Evaluation success rate
- Training loss trend
- Epsilon exploration schedule

## Results

The implemented DQN script trains a PyTorch-based agent for Acrobot-v1, records per-episode rewards, episode lengths, checkpoint success rates, and evaluation statistics, and exports them as CSV and PNG artifacts. In the latest verified run, the agent was trained for 1200 episodes and reached a best 50-episode moving-average reward of **-119.62**. The greedy evaluation summary reported a mean reward of **-489.10**, a standard deviation of **46.38**, a success rate of **10.0%**, and a mean episode length of **489.2** steps. These results show that the agent learned a clearer reward trend over time, but it did not yet produce a fully reliable Acrobot-solving policy within the tested budget.

## Comparison

The saved Actor-Critic and Q-Actor-Critic results remain strong reference points for comparison. Actor-Critic achieved a mean reward of **-87.06**, 100% success, and a mean episode length of **88.1** steps, while Q-Actor-Critic achieved **-160.39**, 100% success, and a mean episode length of **161.39** steps. Based on these reported results, **Actor-Critic demonstrates the best overall performance** among the explored methods because it achieved the highest reward quality, the highest success rate, and the shortest episode lengths. The DQN implementation learned a visible reward-improvement trend, but its final evaluation performance was still weaker than the actor-critic baselines and therefore should be described as promising rather than fully competitive in this assignment's comparison.
