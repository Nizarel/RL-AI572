# Programming Assignment 3: Deep Q-Network for Acrobot

## Objective

Implement a DQN agent for the Acrobot environment, evaluate its learning behavior, and compare it with the other reinforcement-learning algorithms explored for Acrobot.

## Method

- Use a PyTorch multilayer perceptron (2 x 128 hidden units, ReLU) to approximate the Q-function.
- Use experience replay and a target network to stabilize learning.
- Train using epsilon-greedy exploration on Acrobot-v1.

### Hyperparameter Refinement

An initial configuration failed to learn (greedy evaluation of -500.0 with 0% success). Diagnosis identified one correctness defect and four hyperparameter problems, all of which were corrected:

| Aspect | Initial | Refined | Rationale |
| --- | --- | --- | --- |
| Time-out handling | `done = terminated or truncated` | `done = terminated` only | The 500-step limit is a time-out, not a terminal state. Masking it forced the bootstrap target to 0 for every unsolved episode, destroying the return estimate. |
| Exploration schedule | decay 0.995, floor 0.05 | decay 0.97, floor 0.02 | Epsilon was still ~0.37 at episode 200, so almost no greedy experience was collected. The refined schedule reaches ~0.05 by episode 100. |
| Target network | hard copy every 10 episodes | Polyak soft update, tau = 0.005, every step | Removes several thousand steps of target staleness between syncs. |
| Target estimate | vanilla max over target network | Double DQN (online net selects, target net evaluates) | Removes the maximisation bias that inflates Q under a dense -1 reward. |
| Optimisation | MSE, lr 1e-3, batch 64, no warm-up | Huber loss, lr 5e-4, batch 128, gradient clip 10, 1000-step warm-up | Bounds the update size when TD errors are large and lowers gradient variance. |

Remaining settings: gamma = 0.99, replay capacity 100,000, 300 training episodes, 500-step episode cap.

## Metrics

- Episode reward
- Moving-average reward
- Episode length
- Evaluation success rate (greedy, no exploration)
- Sample efficiency (first episode whose moving-average reward exceeds -200)
- Training loss trend
- Epsilon exploration schedule

## Results

The refined DQN solves Acrobot-v1 reliably. Over 300 training episodes (242 s on CPU) the agent reached a best 25-episode moving-average reward of **-76.52**, and the moving average first crossed the -200 threshold at **episode 42**. The final greedy evaluation over 30 exploration-free episodes reported a mean reward of **-78.57**, a standard deviation of **11.62**, a success rate of **100.0%**, and a mean episode length of **79.6** steps. Checkpoint evaluations reached and held 100% success from episode 125 onward.

The notebook implementation reproduces this result independently: **-79.13** mean reward, **100%** success, and **80.1** mean episode length over its own 30 greedy episodes. The close agreement between the two runs indicates the outcome is a property of the refined configuration rather than a favourable seed.

Artifacts: `results.csv`, `checkpoint_success.csv`, `summary_metrics.csv`, and `figures/` from the script; `notebook_outputs/` from the notebook.

## Comparison

All three algorithms are compared on the same basis: mean reward, success rate, and mean episode length from exploration-free evaluation episodes. Reward closer to zero is better because Acrobot assigns -1 per step.

| Algorithm | Mean reward | Success rate | Mean episode length |
| --- | --- | --- | --- |
| **DQN** | **-78.57** | 100% | **79.6** |
| Actor-Critic | -87.06 | 100% | 88.1 |
| Q-Actor-Critic | -160.39 | 100% | 161.39 |

**DQN demonstrates the superior performance** among the three explored algorithms. It achieves the highest (least negative) mean reward, beating Actor-Critic by 8.49 reward and reaching the goal roughly 8 steps sooner per episode; it outperforms Q-Actor-Critic by a wide margin of 81.82 reward.

Because all three methods reach 100% success, success rate alone does not separate them, so mean reward and episode length are the metrics that distinguish policy efficiency. The ranking is therefore driven by how quickly each policy swings the lower link to the target height: DQN is the most efficient, Actor-Critic is close behind, and Q-Actor-Critic takes roughly twice as many steps.

One caveat: the DQN result depends materially on the refinements described above. The same architecture with the initial configuration scored -500.0 with 0% success, so this comparison reflects a tuned DQN against the previously reported actor-critic baselines.
