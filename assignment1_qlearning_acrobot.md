# Programming Assignment 1: Q-Learning for Acrobot

## 1. What the assignment is asking you to do

You need to submit a documented solution for the Acrobot control problem using Q-learning in Python. Based on the rubric in the assignment screenshot, your submission should cover five areas:

1. A working Python implementation of the Q-learning algorithm.
2. Performance metrics that can also be reused later for other reinforcement learning algorithms.
3. Clear execution instructions.
4. Visualizations that show whether learning is improving.
5. Written documentation that explains your approach and results.

The key technical detail is that `Acrobot-v1` has a continuous observation space and a discrete action space. Standard tabular Q-learning does not work directly on continuous states, so you need a state discretization step before storing values in a Q-table.

## 2. Problem summary: Acrobot in plain terms

Acrobot is a two-link pendulum where only one joint is actuated. The goal is to swing the end of the lower link high enough to reach the target height.

Typical environment setup:

- Environment: `Acrobot-v1`
- Library: Gymnasium
- Observation: 6 values
- Actions: 3 discrete torques
- Reward: usually `-1` every step until the goal is reached
- Objective: reach the goal in as few steps as possible

Because the reward is `-1` per step, better policies usually have:

- higher episode return, meaning less negative total reward
- fewer steps before termination
- higher success rate during evaluation

## 3. Recommended method

Use **tabular Q-learning with discretized states**.

Why this is appropriate:

- The action space is small and discrete.
- Q-learning is a standard first assignment algorithm.
- It is easy to explain, visualize, and compare with future algorithms.

Why discretization is required:

- The environment state is continuous.
- A Q-table needs discrete indices.
- You will convert each observation dimension into a bin index, then combine the bin indices into one discrete state representation.

## 4. Step-by-step implementation plan

### Step 1: Install the required packages

Use a small dependency set:

```bash
pip install gymnasium[classic-control] numpy matplotlib pandas
```

Optional:

- `seaborn` for nicer plots
- `jupyter` if you prefer a notebook submission

### Step 2: Create the environment and inspect it

Start by confirming the basic environment properties.

```python
import gymnasium as gym

env = gym.make("Acrobot-v1")
print(env.observation_space)
print(env.action_space)
```

You should verify:

- the observation has 6 components
- the action space has 3 actions

### Step 3: Define state bins for discretization

The Acrobot observation is usually:

```python
[cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]
```

Suggested handling:

- For the four cosine and sine terms, values are already in `[-1, 1]`.
- For angular velocities, clip to practical ranges before binning.

Suggested starting bins:

```python
num_bins = [8, 8, 8, 8, 12, 12]
```

Suggested bounds:

```python
state_low = [-1.0, -1.0, -1.0, -1.0, -12.57, -28.27]
state_high = [1.0, 1.0, 1.0, 1.0, 12.57, 28.27]
```

Implementation idea:

1. Clip each state value into its valid range.
2. Convert each value to a bin index.
3. Return the six bin indices as a tuple.

Example discretizer:

```python
import numpy as np

state_low = np.array([-1.0, -1.0, -1.0, -1.0, -12.57, -28.27])
state_high = np.array([1.0, 1.0, 1.0, 1.0, 12.57, 28.27])
num_bins = np.array([8, 8, 8, 8, 12, 12])

def discretize_state(state):
    state = np.clip(state, state_low, state_high)
    ratios = (state - state_low) / (state_high - state_low)
    indices = (ratios * (num_bins - 1)).astype(int)
    return tuple(indices)
```

### Step 4: Initialize the Q-table

The Q-table shape should be:

```python
(*num_bins, action_size)
```

Example:

```python
action_size = env.action_space.n
q_table = np.zeros(tuple(num_bins) + (action_size,))
```

This means each discretized state stores one Q-value per action.

### Step 5: Choose initial hyperparameters

Use a simple baseline first:

```python
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
num_episodes = 5000
max_steps = 500
```

Interpretation:

- `alpha`: learning rate
- `gamma`: how much future reward matters
- `epsilon`: exploration rate
- `epsilon_decay`: how quickly the agent becomes less random

These values are not guaranteed to be optimal. They are reasonable starting points for your first experiment.

### Step 6: Implement epsilon-greedy action selection

Your policy during training should explore sometimes and exploit learned Q-values otherwise.

```python
def choose_action(state_idx, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(q_table[state_idx]))
```

### Step 7: Implement the Q-learning update rule

The update equation is:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Bigr]
$$

In code:

```python
def update_q_value(state_idx, action, reward, next_state_idx, done):
    best_next = 0.0 if done else np.max(q_table[next_state_idx])
    td_target = reward + gamma * best_next
    td_error = td_target - q_table[state_idx][action]
    q_table[state_idx][action] += alpha * td_error
```

### Step 8: Write the training loop

For each episode:

1. Reset the environment.
2. Discretize the initial state.
3. Repeatedly choose an action using epsilon-greedy.
4. Step the environment.
5. Discretize the next state.
6. Update the Q-table.
7. Track reward, steps, and success.
8. Decay epsilon after the episode.

Training skeleton:

```python
episode_rewards = []
episode_lengths = []
success_flags = []

for episode in range(num_episodes):
    state, _ = env.reset()
    state_idx = discretize_state(state)
    total_reward = 0
    success = 0

    for step in range(max_steps):
        action = choose_action(state_idx, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state_idx = discretize_state(next_state)
        update_q_value(state_idx, action, reward, next_state_idx, done)

        state_idx = next_state_idx
        total_reward += reward

        if terminated:
            success = 1

        if done:
            episode_lengths.append(step + 1)
            break

    episode_rewards.append(total_reward)
    success_flags.append(success)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

### Step 9: Add an evaluation phase

Training performance and evaluation performance should be reported separately.

Use a greedy policy during evaluation:

```python
def evaluate_policy(env, q_table, episodes=100, max_steps=500):
    rewards = []
    lengths = []
    successes = []

    for _ in range(episodes):
        state, _ = env.reset()
        state_idx = discretize_state(state)
        total_reward = 0

        for step in range(max_steps):
            action = int(np.argmax(q_table[state_idx]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_idx = discretize_state(next_state)
            total_reward += reward

            if done:
                rewards.append(total_reward)
                lengths.append(step + 1)
                successes.append(1 if terminated else 0)
                break

    return rewards, lengths, successes
```

This is important because an assignment report should show what the final learned policy can do, not only what happened during exploration.

### Step 10: Save results for plots and the report

Save at least:

- episode reward
- episode length
- success flag
- epsilon per episode
- final evaluation summary

Saving results into a CSV makes your report easier to produce later.

## 5. Metrics you should report

Since your next assignments will likely compare several RL algorithms on the same problem, use metrics that stay consistent across methods.

### Core metrics

1. **Episode return**  
   Sum of rewards per episode. For Acrobot, less negative is better.

2. **Moving average return**  
   Use a window such as 50 or 100 episodes to show the learning trend more clearly.

3. **Episode length**  
   Number of steps before the episode ends. Fewer steps usually means better control.

4. **Success rate**  
   Percentage of evaluation episodes that reach the goal.

5. **Average steps to goal**  
   Compute this over successful evaluation episodes only.

6. **Training time**  
   Total wall-clock time for training.

7. **Sample efficiency**  
   Number of episodes required to reach a target moving average performance or success rate.

### Recommended summary table

Include one final table with:

- total training episodes
- final epsilon
- best moving average reward
- final evaluation mean reward
- final evaluation success rate
- final evaluation mean episode length
- training time

## 6. Visualizations to include

These align well with the rubric and make future comparisons easier.

### Required plots

1. **Episode reward vs. episode**  
   Show both raw rewards and a moving average curve.

2. **Episode length vs. episode**  
   This often becomes more informative than reward for Acrobot.

3. **Success rate across checkpoints**  
   Evaluate the learned policy every fixed number of episodes, such as every 100 or 250 episodes.

4. **Epsilon decay curve**  
   This shows how exploration changed during training.

### Optional plots

1. Histogram of evaluation episode lengths.
2. Comparison plot for multiple hyperparameter settings.
3. Heatmap of state-action visitation counts.

## 7. Suggested report structure for the PDF submission

You can use the following structure directly in your report.

### Title

`Programming Assignment 1: Q-Learning for Acrobot`

### 1. Introduction

Briefly describe:

- the Acrobot problem
- why reinforcement learning is suitable
- why Q-learning was chosen for this assignment

### 2. Environment description

Explain:

- state representation
- action space
- reward structure
- termination condition

### 3. Methodology

Explain:

- Q-learning update rule
- why discretization is necessary
- how you discretized the state space
- chosen hyperparameters
- exploration strategy

### 4. Implementation details

Describe:

- libraries used
- training loop design
- evaluation procedure
- data logging approach

### 5. Metrics

List the metrics you tracked and explain why they matter.

### 6. Results and visualizations

Insert plots and interpret them.

Example questions to answer:

- Did the learning curve improve over time?
- Did the agent solve the task consistently?
- Which metric best reflected progress?

### 7. Discussion

Discuss:

- what worked well
- where the agent struggled
- possible improvements such as more bins, slower epsilon decay, or longer training

### 8. Conclusion

Summarize the final performance and what you learned from the assignment.

## 8. Common mistakes to avoid

1. **Using raw continuous states directly in a Q-table**  
   This is the most common mistake for this environment.

2. **Confusing `terminated` and `truncated`**  
   In Gymnasium, both should contribute to `done = terminated or truncated`.

3. **Reporting only training reward**  
   You also need a separate evaluation result with little or no exploration.

4. **Using too few episodes**  
   Acrobot often needs several thousand episodes for a clear learning trend.

5. **Using only one metric**  
   Reward alone can hide important behavior. Include episode length and success rate too.

6. **Not fixing a random seed when comparing settings**  
   If you compare hyperparameters, use a seed for fairer comparisons.

## 9. Minimal implementation checklist

Use this as your build order.

1. Create the environment.
2. Print the observation and action spaces.
3. Write `discretize_state`.
4. Initialize the Q-table.
5. Write epsilon-greedy action selection.
6. Write the Q-learning update function.
7. Write the training loop.
8. Store reward, episode length, success, and epsilon.
9. Add a separate evaluation function.
10. Generate plots.
11. Write the PDF report using those results.

## 10. Suggested folder structure

If you want to organize your work cleanly, use something like this:

```text
RL-AI572/
|-- README.md
|-- assignment1_qlearning_acrobot.md
|-- q_learning_acrobot.py
|-- results.csv
|-- figures/
|   |-- reward_curve.png
|   |-- episode_length_curve.png
|   |-- success_rate.png
|   `-- epsilon_decay.png
`-- report.pdf
```

## 11. Short example of what to say in the documentation

You can reuse language like this in your report:

> The Acrobot environment has a continuous observation space and a discrete action space. Because tabular Q-learning requires discrete states, the six-dimensional observation vector was discretized into fixed bins. The agent was trained using an epsilon-greedy exploration strategy, and the Q-table was updated with the standard temporal-difference target. Performance was measured using episode return, episode length, success rate, and training time. These metrics were selected so that the same evaluation framework can be reused in later assignments for other reinforcement learning algorithms.

## 12. Final submission checklist

Before submitting, verify that you have:

- a working `.py` or `.ipynb` file
- a PDF report
- explanation of the environment and algorithm
- the Q-learning update equation
- a clear discretization strategy
- at least three meaningful plots
- a final metrics table
- a short discussion of strengths, limits, and possible improvements

If you want, the next practical step is to turn this guide into an actual `q_learning_acrobot.py` training script.