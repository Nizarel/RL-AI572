"""
Programming Assignment 3 — Deep Q-Network (DQN) for Acrobot
==========================================================

This script trains a simple DQN agent for Acrobot-v1 using PyTorch.
It saves training history, summary metrics, evaluation checkpoints,
and figure outputs under the Assignment_3 folder.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


CONFIG = {
    "seed": 42,
    "num_episodes": 400,
    "max_steps": 500,
    "eval_interval": 50,
    "eval_episodes": 40,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "replay_capacity": 20000,
    "target_update": 10,
    "hidden_dim": 64,
    "ma_window": 50,
    "efficiency_threshold": -150.0,
}


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def make_env(seed: int) -> gym.Env:
    env = gym.make("Acrobot-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")
    return env


def preprocess_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 12.566371, 28.274334], dtype=np.float32)
    return state / scale


def choose_action(policy_net: DQN, state: np.ndarray, epsilon: float, action_size: int, device: torch.device) -> int:
    if np.random.random() < epsilon:
        return int(np.random.randint(action_size))
    with torch.no_grad():
        state_tensor = torch.tensor(preprocess_state(state), dtype=torch.float32, device=device).unsqueeze(0)
        q_values = policy_net(state_tensor)
    return int(q_values.argmax().item())


def evaluate_agent(env: gym.Env, policy_net: DQN, cfg: Dict[str, float], device: torch.device) -> Dict[str, List[float]]:
    rewards: List[float] = []
    lengths: List[float] = []
    successes: List[float] = []
    for _ in range(cfg["eval_episodes"]):
        obs, _ = env.reset()
        total_reward = 0.0
        ep_length = 0
        success = 0
        for _ in range(cfg["max_steps"]):
            action = choose_action(policy_net, obs, epsilon=0.0, action_size=env.action_space.n, device=device)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            ep_length += 1
            if terminated:
                success = 1
                break
            if truncated:
                break
        rewards.append(total_reward)
        lengths.append(ep_length)
        successes.append(success)
    return {"rewards": rewards, "lengths": lengths, "successes": successes}


def train(env: gym.Env, eval_env: gym.Env, cfg: Dict[str, float]) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    device = torch.device("cpu")
    set_seed(cfg["seed"])

    state_dim = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = DQN(state_dim, int(cfg["hidden_dim"]), action_size).to(device)
    target_net = DQN(state_dim, int(cfg["hidden_dim"]), action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=cfg["learning_rate"])
    replay_buffer = ReplayBuffer(int(cfg["replay_capacity"]))

    history: Dict[str, List[float]] = {
        "rewards": [],
        "lengths": [],
        "successes": [],
        "epsilons": [],
        "checkpoint_episodes": [],
        "checkpoint_success_rates": [],
        "losses": [],
    }

    epsilon = cfg["epsilon_start"]
    start_time = time.time()

    for episode in range(1, int(cfg["num_episodes"]) + 1):
        obs, _ = env.reset()
        obs = preprocess_state(obs)
        total_reward = 0.0
        success = 0
        ep_length = 0
        for _ in range(int(cfg["max_steps"])):
            action = choose_action(policy_net, obs, epsilon, action_size, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            shaped_reward = float(reward)
            if not done:
                shaped_reward += -0.01 * np.linalg.norm(next_obs[:4])
            replay_buffer.push(obs, action, shaped_reward, preprocess_state(next_obs), done)
            obs = preprocess_state(next_obs)
            total_reward += shaped_reward
            ep_length += 1

            if len(replay_buffer) >= int(cfg["batch_size"]):
                states, actions, rewards_batch, next_states, dones = replay_buffer.sample(int(cfg["batch_size"]))
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1).values
                    target_q_values = rewards_batch + cfg["gamma"] * next_q_values * (1.0 - dones)
                loss = F.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
                optimizer.step()
                history["losses"].append(float(loss.item()))

            if terminated:
                success = 1
                break
            if truncated:
                break

        history["rewards"].append(total_reward)
        history["lengths"].append(ep_length)
        history["successes"].append(success)
        history["epsilons"].append(epsilon)

        epsilon = max(cfg["epsilon_min"], epsilon * cfg["epsilon_decay"])

        if episode % int(cfg["eval_interval"]) == 0:
            eval_result = evaluate_agent(eval_env, policy_net, cfg, device)
            checkpoint_success_rate = 100.0 * float(np.mean(eval_result["successes"]))
            history["checkpoint_episodes"].append(episode)
            history["checkpoint_success_rates"].append(checkpoint_success_rate)
            print(
                f"Episode {episode:4d}/{int(cfg['num_episodes'])} | "
                f"Reward: {total_reward:7.1f} | "
                f"Length: {ep_length:3d} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Eval success: {checkpoint_success_rate:5.1f}%"
            )

        if episode % int(cfg["target_update"]) == 0:
            target_net.load_state_dict(policy_net.state_dict())

    training_time = time.time() - start_time
    history["training_time"] = [training_time]

    final_eval = evaluate_agent(eval_env, policy_net, cfg, device)
    rewards = np.array(history["rewards"], dtype=float)
    ma_window = int(cfg["ma_window"])
    ma_rewards = np.convolve(rewards, np.ones(ma_window) / ma_window, mode="valid")
    best_ma = float(np.max(ma_rewards)) if len(ma_rewards) > 0 else float("nan")
    threshold = float(cfg["efficiency_threshold"])
    sample_efficiency_ep = int(next((i + ma_window for i, value in enumerate(ma_rewards) if value >= threshold), -1)) if len(ma_rewards) > 0 else -1

    eval_rewards = np.array(final_eval["rewards"], dtype=float)
    eval_lengths = np.array(final_eval["lengths"], dtype=float)
    eval_successes = np.array(final_eval["successes"], dtype=float)

    return history, {
        "total_episodes": int(cfg["num_episodes"]),
        "training_time_sec": round(float(training_time), 2),
        "final_epsilon": round(float(epsilon), 5),
        "best_ma_reward": round(best_ma, 2),
        "eval_mean_reward": round(float(np.mean(eval_rewards)), 2),
        "eval_std_reward": round(float(np.std(eval_rewards)), 2),
        "eval_success_rate_pct": round(100.0 * float(np.mean(eval_successes)), 1),
        "eval_mean_ep_length": round(float(np.mean(eval_lengths)), 1),
        "sample_efficiency_ep": sample_efficiency_ep,
    }


def save_histories(history: Dict[str, List[float]], metrics: Dict[str, float], base_dir: Path) -> None:
    results_df = pd.DataFrame({
        "episode": range(1, len(history["rewards"]) + 1),
        "reward": history["rewards"],
        "length": history["lengths"],
        "success": history["successes"],
        "epsilon": history["epsilons"],
    })
    results_df.to_csv(base_dir / "results.csv", index=False)

    checkpoint_df = pd.DataFrame({
        "episode": history["checkpoint_episodes"],
        "success_rate_pct": history["checkpoint_success_rates"],
    })
    checkpoint_df.to_csv(base_dir / "checkpoint_success.csv", index=False)

    summary_df = pd.DataFrame([metrics])
    summary_df.to_csv(base_dir / "summary_metrics.csv", index=False)


def moving_average(values: List[float], window: int) -> List[float]:
    if len(values) < window:
        return []
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def plot_results(history: Dict[str, List[float]], base_dir: Path, cfg: Dict[str, float]) -> None:
    rewards = history["rewards"]
    lengths = history["lengths"]
    ma_rewards = moving_average(rewards, int(cfg["ma_window"]))
    ma_lengths = moving_average(lengths, int(cfg["ma_window"]))

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(rewards) + 1), rewards, alpha=0.25, color="steelblue", label="Episode reward")
    plt.plot(range(int(cfg["ma_window"]), len(rewards) + 1), ma_rewards, color="steelblue", linewidth=2, label=f"{cfg['ma_window']}-ep moving average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DQN — Episode Reward (Acrobot-v1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "figures" / "reward_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(lengths) + 1), lengths, alpha=0.25, color="darkorange", label="Episode length")
    plt.plot(range(int(cfg["ma_window"]), len(lengths) + 1), ma_lengths, color="darkorange", linewidth=2, label=f"{cfg['ma_window']}-ep moving average")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("DQN — Episode Length (Acrobot-v1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / "figures" / "episode_length_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["checkpoint_episodes"], history["checkpoint_success_rates"], marker="o", color="seagreen", linewidth=2)
    plt.xlabel("Training episode")
    plt.ylabel("Success rate (%)")
    plt.title("DQN — Checkpoint Success Rate")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(base_dir / "figures" / "success_rate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(range(1, len(history["epsilons"]) + 1), history["epsilons"], color="mediumpurple", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("DQN — Epsilon Decay")
    plt.tight_layout()
    plt.savefig(base_dir / "figures" / "epsilon_decay.png", dpi=150)
    plt.close()

    if history["losses"]:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(history["losses"]) + 1), history["losses"], color="tomato", linewidth=1.0)
        plt.xlabel("Optimization step")
        plt.ylabel("Loss")
        plt.title("DQN — Training Loss")
        plt.tight_layout()
        plt.savefig(base_dir / "figures" / "training_loss.png", dpi=150)
        plt.close()


def print_summary(metrics: Dict[str, float], history: Dict[str, List[float]]) -> None:
    rewards = np.array(history["rewards"])
    ma_window = int(CONFIG["ma_window"])
    ma_rewards = np.convolve(rewards, np.ones(ma_window) / ma_window, mode="valid")
    best_ma = float(np.max(ma_rewards)) if len(ma_rewards) > 0 else float("nan")

    print("\n" + "=" * 56)
    print("  DQN SUMMARY")
    print("=" * 56)
    print(f"  Total episodes            : {metrics['total_episodes']}")
    print(f"  Best {ma_window}-ep MA reward : {best_ma:.2f}")
    print(f"  Eval mean reward          : {metrics['eval_mean_reward']:.2f}")
    print(f"  Eval std reward           : {metrics['eval_std_reward']:.2f}")
    print(f"  Eval success rate (%)     : {metrics['eval_success_rate_pct']:.1f}")
    print(f"  Eval mean ep length       : {metrics['eval_mean_ep_length']:.1f}")
    print(f"  Final epsilon             : {metrics['final_epsilon']:.5f}")
    print(f"  Training time (s)         : {metrics['training_time_sec']:.2f}")
    print(f"  Sample efficiency ep      : {metrics['sample_efficiency_ep'] if metrics['sample_efficiency_ep'] >= 0 else 'not reached'}")
    print("=" * 56)


def main() -> None:
    print("=== DQN for Acrobot-v1 ===")
    env = make_env(CONFIG["seed"])
    eval_env = gym.make("Acrobot-v1")
    eval_env.reset(seed=CONFIG["seed"] + 1)

    history, metrics = train(env, eval_env, CONFIG)
    save_histories(history, metrics, BASE_DIR)
    plot_results(history, BASE_DIR, CONFIG)
    print_summary(metrics, history)

    env.close()
    eval_env.close()
    print(f"\nArtifacts written to {BASE_DIR}")


if __name__ == "__main__":
    main()
