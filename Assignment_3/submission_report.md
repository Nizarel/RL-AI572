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

The documented notebook is `DQN_Acrobot_Documented.ipynb`. It includes environment inspection, DQN training, greedy evaluation, reward and episode-length curves, checkpoint success rates, loss and epsilon plots, descriptive statistics, and exported CSV/PNG artifacts.

The documented 200-episode DQN run improves its training reward moving average from the `-500` time limit toward approximately `-300`, with a temporary 80% checkpoint success rate. However, the separate 30-episode greedy evaluation reports **-500.00** mean reward, **0%** success, and **500** steps. This shows that the compact run has not produced a reliable final policy; the training curves alone should not be interpreted as convergence.

## Comparison

The saved Actor-Critic evaluation reports a mean reward of **-87.06**, 100% success, and a mean episode length of 88.1 steps. The saved Q-Actor-Critic evaluation reports **-160.39**, 100% success, and a mean episode length of 161.39 steps. Thus, Actor-Critic is the strongest directly comparable saved result because its reward is closest to zero and its episodes are shortest among those evaluation summaries.

The notebook's comparison section reports all three algorithms using the fresh 30-episode greedy DQN evaluation and the saved evaluation summaries for the two actor-critic methods. On these results, **Actor-Critic demonstrates superior performance**: its mean reward is closest to zero (`-87.06`), success is 100%, and mean episode length is 88.1 steps. The DQN needs more episodes, hyperparameter tuning, or a stronger stabilization strategy before a fair claim of competitiveness can be made.
