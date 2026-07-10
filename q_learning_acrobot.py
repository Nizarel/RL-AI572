"""
Programming Assignment 1 — Q-Learning for Acrobot
==================================================

Install dependencies:
    pip install gymnasium[classic-control] numpy matplotlib pandas

Run:
    python q_learning_acrobot.py

Output artifacts:
    results.csv          — per-episode training history
    summary_metrics.csv  — final metric table for comparisons
    checkpoint_success.csv — periodic greedy-evaluation success rates
    figures/             — four PNG plots
    Console summary      — key metrics

Performance metrics tracked:
    episode return, moving-average return, episode length, success flag,
    checkpoint success rate, final evaluation mean/std reward, final
    evaluation success rate, final evaluation mean episode length, training
    time, and sample-efficiency threshold crossing.

Visualizations generated:
    reward curve, episode-length curve, checkpoint success-rate curve,
    and epsilon-decay curve.
"""

import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG — all hyperparameters in one place
# ---------------------------------------------------------------------------
CONFIG = {
    "seed": 42,
    # Discretization
    "state_low":  np.array([-1.0, -1.0, -1.0, -1.0, -12.57, -28.27]),
    "state_high": np.array([ 1.0,  1.0,  1.0,  1.0,  12.57,  28.27]),
    "num_bins":   np.array([8, 8, 8, 8, 12, 12]),
    # Q-learning
    "alpha":         0.1,    # learning rate
    "gamma":         0.99,   # discount factor
    "epsilon":       1.0,    # initial exploration rate
    "epsilon_min":   0.05,
    "epsilon_decay": 0.995,
    # Training
    "num_episodes":  5000,
    "max_steps":     500,
    "eval_interval": 500,    # evaluate every N training episodes
    # Evaluation
    "eval_episodes": 100,
    # Plotting
    "ma_window":     100,    # moving-average window
    # Sample-efficiency threshold: first episode where 100-ep MA exceeds this
    "efficiency_threshold": -200.0,
}


# ---------------------------------------------------------------------------
# Phase 2 — Environment & discretization
# ---------------------------------------------------------------------------

def make_env(seed: int) -> gym.Env:
    env = gym.make("Acrobot-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    return env


def discretize_state(state: np.ndarray, cfg: dict) -> tuple:
    """Convert a continuous 6-D observation to a discrete tuple index."""
    clipped = np.clip(state, cfg["state_low"], cfg["state_high"])
    ratios  = (clipped - cfg["state_low"]) / (cfg["state_high"] - cfg["state_low"])
    indices = np.floor(ratios * cfg["num_bins"]).astype(int)
    indices = np.clip(indices, 0, cfg["num_bins"] - 1)
    return tuple(indices)


def _sanity_check_discretizer(env: gym.Env, cfg: dict) -> None:
    obs, _ = env.reset(seed=cfg["seed"])
    idx = discretize_state(obs, cfg)
    assert isinstance(idx, tuple) and len(idx) == 6, "discretize_state must return a 6-tuple"
    assert all(isinstance(i, (int, np.integer)) for i in idx), "all indices must be integers"


# ---------------------------------------------------------------------------
# Phase 3 — Q-table & policy
# ---------------------------------------------------------------------------

def init_q_table(cfg: dict, action_size: int) -> np.ndarray:
    shape = tuple(cfg["num_bins"]) + (action_size,)
    return np.zeros(shape)


def choose_action(rng: np.random.Generator, q_table: np.ndarray, state_idx: tuple,
                  epsilon: float, action_size: int) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(action_size))
    return int(np.argmax(q_table[state_idx]))


def update_q(q_table: np.ndarray, state_idx: tuple, action: int,
             reward: float, next_state_idx: tuple, done: bool,
             alpha: float, gamma: float) -> None:
    best_next = 0.0 if done else float(np.max(q_table[next_state_idx]))
    td_target = reward + gamma * best_next
    td_error  = td_target - q_table[state_idx][action]
    q_table[state_idx][action] += alpha * td_error


# ---------------------------------------------------------------------------
# Phase 5 — Evaluation (defined before training so it can be called inline)
# ---------------------------------------------------------------------------

def evaluate(env: gym.Env, q_table: np.ndarray, cfg: dict) -> dict:
    """
    Run `eval_episodes` greedy episodes.
    Returns dict with keys: rewards, lengths, successes.
    """
    eval_episodes = cfg["eval_episodes"]
    max_steps = cfg["max_steps"]
    rewards, lengths, successes = [], [], []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        state_idx = discretize_state(obs, cfg)
        total_reward = 0.0
        success = 0
        for step in range(max_steps):
            action = int(np.argmax(q_table[state_idx]))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state_idx = discretize_state(obs, cfg)
            total_reward += reward
            if terminated:
                success = 1
            if done:
                break
        rewards.append(total_reward)
        lengths.append(step + 1)
        successes.append(success)
    return {"rewards": rewards, "lengths": lengths, "successes": successes}


# ---------------------------------------------------------------------------
# Phase 4 — Training loop
# ---------------------------------------------------------------------------

def train(env: gym.Env, eval_env: gym.Env, q_table: np.ndarray,
          cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """
    Train the agent. Returns (updated q_table, history).
    history keys: rewards, lengths, successes, epsilons, checkpoint_episodes,
                  checkpoint_success_rates, training_time.
    """
    epsilon = cfg["epsilon"]
    action_size = env.action_space.n
    alpha = cfg["alpha"]
    gamma = cfg["gamma"]
    max_steps = cfg["max_steps"]
    ma_window = cfg["ma_window"]
    eval_interval = cfg["eval_interval"]
    num_episodes = cfg["num_episodes"]
    history = {
        "rewards":                  [],
        "lengths":                  [],
        "successes":                [],
        "epsilons":                 [],
        "checkpoint_episodes":      [],
        "checkpoint_success_rates": [],
    }

    t_start = time.time()

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state_idx = discretize_state(obs, cfg)
        total_reward = 0.0
        success = 0
        ep_length = max_steps

        for step in range(max_steps):
            action = choose_action(rng, q_table, state_idx, epsilon, action_size)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_idx = discretize_state(obs, cfg)

            update_q(q_table, state_idx, action, reward, next_state_idx, done,
                     alpha, gamma)

            state_idx = next_state_idx
            total_reward += reward

            if terminated:
                success = 1
            if done:
                ep_length = step + 1
                break

        history["rewards"].append(total_reward)
        history["lengths"].append(ep_length)
        history["successes"].append(success)
        history["epsilons"].append(epsilon)

        # Decay epsilon
        epsilon = max(cfg["epsilon_min"], epsilon * cfg["epsilon_decay"])

        # Periodic evaluation checkpoint
        if episode % eval_interval == 0:
            ckpt = evaluate(eval_env, q_table, cfg)
            success_rate = 100.0 * sum(ckpt["successes"]) / len(ckpt["successes"])
            history["checkpoint_episodes"].append(episode)
            history["checkpoint_success_rates"].append(success_rate)

            # Progress print
            recent_rewards = history["rewards"][-ma_window:]
            ma = np.mean(recent_rewards)
            print(
                f"Episode {episode:5d}/{num_episodes} | "
                f"MA Reward(100): {ma:8.1f} | "
                f"Epsilon: {epsilon:.4f} | "
                f"Eval success: {success_rate:.1f}%"
            )

    history["training_time"] = time.time() - t_start
    return q_table, history


# ---------------------------------------------------------------------------
# Phase 6 — Metrics & CSV
# ---------------------------------------------------------------------------

def compute_metrics(history: dict, final_eval: dict, cfg: dict) -> dict:
    rewards = np.array(history["rewards"])
    w = cfg["ma_window"]

    # Moving average (full series)
    ma_rewards = np.convolve(rewards, np.ones(w) / w, mode="valid")
    best_ma = float(np.max(ma_rewards)) if len(ma_rewards) > 0 else float("nan")

    # Sample efficiency
    threshold = cfg["efficiency_threshold"]
    crossing = next(
        (i + w for i, v in enumerate(ma_rewards) if v >= threshold),
        None
    )

    eval_rewards    = np.array(final_eval["rewards"])
    eval_lengths    = np.array(final_eval["lengths"])
    eval_successes  = np.array(final_eval["successes"])

    return {
        "total_episodes":          cfg["num_episodes"],
        "final_epsilon":           round(history["epsilons"][-1], 5),
        "best_ma_reward":          round(best_ma, 2),
        "eval_mean_reward":        round(float(np.mean(eval_rewards)), 2),
        "eval_std_reward":         round(float(np.std(eval_rewards)),  2),
        "eval_success_rate_pct":   round(100.0 * float(np.mean(eval_successes)), 1),
        "eval_mean_ep_length":     round(float(np.mean(eval_lengths)), 1),
        "training_time_sec":       round(history["training_time"], 1),
        "sample_efficiency_ep":    crossing,
    }


def save_csv(history: dict, path: str = "results.csv") -> None:
    df = pd.DataFrame({
        "episode": range(1, len(history["rewards"]) + 1),
        "reward":  history["rewards"],
        "length":  history["lengths"],
        "success": history["successes"],
        "epsilon": history["epsilons"],
    })
    df.to_csv(path, index=False)
    print(f"Saved training history → {path}")


def save_summary_csv(metrics: dict, path: str = "summary_metrics.csv") -> None:
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"Saved summary metrics → {path}")


def save_checkpoint_csv(history: dict, path: str = "checkpoint_success.csv") -> None:
    df = pd.DataFrame({
        "episode": history["checkpoint_episodes"],
        "success_rate_pct": history["checkpoint_success_rates"],
    })
    df.to_csv(path, index=False)
    print(f"Saved checkpoint success rates → {path}")


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 52)
    print("  SUMMARY TABLE")
    print("=" * 52)
    rows = [
        ("Total training episodes",      metrics["total_episodes"]),
        ("Final epsilon",                metrics["final_epsilon"]),
        ("Best 100-ep moving avg reward",metrics["best_ma_reward"]),
        ("Eval mean reward",             metrics["eval_mean_reward"]),
        ("Eval std reward",              metrics["eval_std_reward"]),
        ("Eval success rate (%)",        metrics["eval_success_rate_pct"]),
        ("Eval mean episode length",     metrics["eval_mean_ep_length"]),
        ("Training time (s)",            metrics["training_time_sec"]),
        ("Sample efficiency (episode)",
         metrics["sample_efficiency_ep"] if metrics["sample_efficiency_ep"] else "Not reached"),
    ]
    for label, value in rows:
        print(f"  {label:<34} {value}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Phase 7 — Visualizations
# ---------------------------------------------------------------------------

def _moving_average(data: list, window: int) -> np.ndarray:
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_reward_curve(history: dict, cfg: dict, out_dir: str) -> None:
    rewards = history["rewards"]
    ma = _moving_average(rewards, cfg["ma_window"])
    episodes = range(1, len(rewards) + 1)
    ma_episodes = range(cfg["ma_window"], len(rewards) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, rewards, alpha=0.25, color="steelblue", label="Episode reward")
    plt.plot(ma_episodes, ma,   color="steelblue",  linewidth=2,
             label=f"{cfg['ma_window']}-ep moving average")
    plt.axhline(-500, color="red", linestyle="--", linewidth=0.8, label="Worst possible")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-Learning — Episode Reward (Acrobot-v1)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "reward_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot → {path}")


def plot_length_curve(history: dict, cfg: dict, out_dir: str) -> None:
    lengths = history["lengths"]
    ma = _moving_average(lengths, cfg["ma_window"])
    episodes = range(1, len(lengths) + 1)
    ma_episodes = range(cfg["ma_window"], len(lengths) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, lengths, alpha=0.25, color="darkorange", label="Episode length")
    plt.plot(ma_episodes, ma,   color="darkorange", linewidth=2,
             label=f"{cfg['ma_window']}-ep moving average")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Q-Learning — Episode Length (Acrobot-v1)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "episode_length_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot → {path}")


def plot_success_rate(history: dict, out_dir: str) -> None:
    ckpt_eps   = history["checkpoint_episodes"]
    ckpt_rates = history["checkpoint_success_rates"]

    plt.figure(figsize=(8, 4))
    plt.plot(ckpt_eps, ckpt_rates, marker="o", color="seagreen", linewidth=2)
    plt.xlabel("Training episode")
    plt.ylabel("Success rate (%)")
    plt.title("Q-Learning — Checkpoint Success Rate (Acrobot-v1)")
    plt.ylim(0, 105)
    plt.tight_layout()
    path = os.path.join(out_dir, "success_rate.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot → {path}")


def plot_epsilon_decay(history: dict, out_dir: str) -> None:
    plt.figure(figsize=(8, 3))
    plt.plot(range(1, len(history["epsilons"]) + 1), history["epsilons"],
             color="mediumpurple", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Q-Learning — Epsilon Decay (Acrobot-v1)")
    plt.tight_layout()
    path = os.path.join(out_dir, "epsilon_decay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved plot → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(CONFIG["seed"])
    os.makedirs("figures", exist_ok=True)

    print("\n=== Q-Learning for Acrobot-v1 ===\n")

    # Phase 2
    env = make_env(CONFIG["seed"])
    eval_env = gym.make("Acrobot-v1")
    eval_env.reset(seed=CONFIG["seed"] + 1)
    _sanity_check_discretizer(env, CONFIG)

    # Phase 3
    q_table = init_q_table(CONFIG, env.action_space.n)
    print(f"Q-table shape: {q_table.shape}  "
          f"({q_table.size:,} entries)\n")

    # Phase 4 + 5 (training + periodic evaluation)
    print("Training ...\n")
    q_table, history = train(env, eval_env, q_table, CONFIG, rng)

    # Phase 5 — final evaluation
    print("\nRunning final evaluation ...")
    final_eval = evaluate(eval_env, q_table, CONFIG)

    # Phase 6
    metrics = compute_metrics(history, final_eval, CONFIG)
    save_csv(history)
    save_summary_csv(metrics)
    save_checkpoint_csv(history)
    print_summary(metrics)

    # Phase 7
    print("Generating plots ...")
    plot_reward_curve(history, CONFIG, "figures")
    plot_length_curve(history, CONFIG, "figures")
    plot_success_rate(history, "figures")
    plot_epsilon_decay(history, "figures")

    print("\nDone. See figures/ for plots, results.csv for training data, ")
    print("summary_metrics.csv for final metrics, and checkpoint_success.csv for checkpoint results.")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
