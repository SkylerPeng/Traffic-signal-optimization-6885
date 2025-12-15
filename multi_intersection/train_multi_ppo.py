import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from cityflow_env_multi import CityFlowMultiEnv
from multi_agent_ppo import MultiAgentPPO


def train_multi_agent_ppo(
        config_path='config.json',
        reward_type='mixed',  # 'pressure', 'waiting', or 'mixed'
        num_episodes=1000,
        max_steps_per_episode=3600,
        update_interval=1024,  # Update more frequently
        save_interval=50,
        eval_interval=20,  # Evaluate every N episodes
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='logs',
        model_dir='models/checkpoints'
):
    """Train Multi-Agent PPO"""

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 70)
    print("Multi-Agent PPO Training - Optimized Version")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Reward Type: {reward_type}")
    print(f"Config: {config_path}")

    # Initialize environment
    env = CityFlowMultiEnv(config_path, steps_per_action=10, reward_type=reward_type)

    num_agents = env.num_agents
    state_dim = 2  # [waiting_vehicles, total_vehicles]
    action_dims = [env.action_space.nvec[i] for i in range(num_agents)]

    print(f"Number of agents: {num_agents}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimensions: {action_dims}")
    print("=" * 70)

    # Initialize PPO with optimized hyperparameters
    ppo = MultiAgentPPO(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dims=action_dims,
        lr=5e-4,
        gamma=0.98,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.05,
        max_grad_norm=0.5,
        ppo_epochs=4,
        mini_batch_size=128,
        device=device
    )

    # Metrics
    episode_rewards = []
    episode_waiting_times = []
    episode_pressures = []
    training_losses = []

    best_reward = -float('inf')
    best_waiting = float('inf')
    total_steps = 0

    # Training loop
    for episode in range(num_episodes):
        states, _ = env.reset()
        episode_reward = 0
        episode_waiting = []
        episode_pressure = []
        steps = 0

        while steps < max_steps_per_episode:
            # Select actions
            actions, log_probs, values = ppo.select_actions(states)

            # Step
            next_states, rewards, done, truncated, info = env.step(actions)

            # Store
            ppo.store_transition(states, actions, log_probs, rewards, values, done)

            episode_reward += np.sum(rewards)
            episode_waiting.append(info['total_waiting'])
            episode_pressure.append(info['avg_pressure'])

            states = next_states
            steps += env.steps_per_action
            total_steps += env.steps_per_action

            # Update
            if total_steps % update_interval == 0:
                metrics = ppo.update(next_states)
                training_losses.append(metrics)

            if done or truncated:
                break

        # Episode stats
        avg_reward = episode_reward / num_agents
        avg_waiting = np.mean(episode_waiting)
        avg_pressure = np.mean(episode_pressure)

        episode_rewards.append(avg_reward)
        episode_waiting_times.append(avg_waiting)
        episode_pressures.append(avg_pressure)

        # Logging
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Steps: {steps} | Total Steps: {total_steps}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Waiting: {avg_waiting:.2f}")
        print(f"  Avg Pressure: {avg_pressure:.2f}")

        if len(training_losses) > 0:
            recent_loss = training_losses[-1]
            print(f"  Loss: {recent_loss['total_loss']:.4f} | "
                  f"Actor: {recent_loss['actor_loss']:.4f} | "
                  f"Critic: {recent_loss['critic_loss']:.4f} | "
                  f"Entropy: {recent_loss['entropy']:.4f}")

        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_waiting = avg_waiting
            ppo.save(os.path.join(model_dir, 'best_model.pth'))
            print(f"  🌟 New best reward: {best_reward:.2f} (waiting: {best_waiting:.2f})")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            ppo.save(os.path.join(model_dir, f'checkpoint_ep{episode + 1}.pth'))

            # Save metrics
            save_training_metrics(
                log_dir, episode_rewards, episode_waiting_times,
                episode_pressures, training_losses
            )

        print("-" * 70)

    # Final save
    ppo.save(os.path.join(model_dir, 'final_model.pth'))

    # Plot results
    plot_training_results(
        log_dir, episode_rewards, episode_waiting_times,
        episode_pressures, training_losses
    )

    print("\n" + "=" * 70)
    print("Training Completed!")
    print(f"Best Reward: {best_reward:.2f}")
    print(f"Best Waiting Time: {best_waiting:.2f}")
    print("=" * 70)

    return ppo


def save_training_metrics(log_dir, rewards, waiting, pressures, losses):
    """Save training metrics to JSON"""
    metrics = {
        'episode_rewards': rewards,
        'episode_waiting_times': waiting,
        'episode_pressures': pressures,
        'training_losses': losses,
        'best_reward': max(rewards) if rewards else None,
        'best_waiting': min(waiting) if waiting else None
    }

    with open(os.path.join(log_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def plot_training_results(log_dir, rewards, waiting, pressures, losses):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Reward')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Waiting times
    axes[0, 1].plot(waiting)
    axes[0, 1].set_title('Average Waiting Vehicles')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Vehicles')
    axes[0, 1].grid(True, alpha=0.3)

    # Pressures
    axes[0, 2].plot(pressures)
    axes[0, 2].set_title('Average Pressure')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Pressure')
    axes[0, 2].grid(True, alpha=0.3)

    # Smoothed rewards
    if len(rewards) >= 20:
        window = 20
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[1, 0].plot(smoothed)
        axes[1, 0].set_title(f'Smoothed Reward (window={window})')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)

    # Losses
    if losses:
        total_losses = [l['total_loss'] for l in losses]
        axes[1, 1].plot(total_losses)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)

    # Entropy
    if losses:
        entropies = [l['entropy'] for l in losses]
        axes[1, 2].plot(entropies)
        axes[1, 2].set_title('Policy Entropy')
        axes[1, 2].set_xlabel('Update Step')
        axes[1, 2].set_ylabel('Entropy')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300)
    print(f"✅ Training curves saved to {log_dir}/training_curves.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--reward', type=str, default='mixed',
                        choices=['pressure', 'waiting', 'mixed'])
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"\n🚀 Starting training with:")
    print(f"   Config: {args.config}")
    print(f"   Reward: {args.reward}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Device: {device}\n")

    train_multi_agent_ppo(
        config_path=args.config,
        reward_type=args.reward,
        num_episodes=args.episodes,
        device=device
    )
