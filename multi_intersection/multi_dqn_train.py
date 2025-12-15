import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import json
import os
import matplotlib.pyplot as plt
from cityflow_env_multi import CityFlowMultiEnv


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class MultiAgentDQN:
    def __init__(self, env, config):
        self.env = env
        self.num_agents = env.num_agents
        self.intersection_ids = env.intersection_ids
        
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 10)
        self.buffer_size = config.get('buffer_size', 10000)
        
        self.state_dim = 2
        self.action_dims = [env.phase_dict[iid] for iid in env.intersection_ids]
        
        self.q_networks = []
        self.target_networks = []
        self.optimizers = []
        
        for action_dim in self.action_dims:
            q_net = DQNNetwork(self.state_dim, action_dim)
            target_net = DQNNetwork(self.state_dim, action_dim)
            target_net.load_state_dict(q_net.state_dict())
            
            self.q_networks.append(q_net)
            self.target_networks.append(target_net)
            self.optimizers.append(optim.Adam(q_net.parameters(), lr=self.learning_rate))
        
        self.replay_buffers = [ReplayBuffer(self.buffer_size) for _ in range(self.num_agents)]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        for i in range(self.num_agents):
            self.q_networks[i].to(self.device)
            self.target_networks[i].to(self.device)
    
    def select_actions(self, states, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        
        actions = []
        for i in range(self.num_agents):
            if random.random() < epsilon:
                action = random.randint(0, self.action_dims[i] - 1)
            else:
                state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_networks[i](state_tensor)
                    action = q_values.argmax().item()
            actions.append(action)
        
        return actions
    
    def store_transition(self, states, actions, rewards, next_states, done):
        for i in range(self.num_agents):
            self.replay_buffers[i].push(states[i], actions[i], rewards[i], next_states[i], done)
    
    def train(self):
        if len(self.replay_buffers[0]) < self.batch_size:
            return None
        
        total_loss = 0
        count = 0
        for i in range(self.num_agents):
            loss = self._train_agent(i)
            if loss is not None:
                total_loss += loss
                count += 1
        
        return total_loss / count if count > 0 else None
    
    def _train_agent(self, agent_idx):
        if len(self.replay_buffers[agent_idx]) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffers[agent_idx].sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.q_networks[agent_idx](states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_networks[agent_idx](next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizers[agent_idx].zero_grad()
        loss.backward()
        self.optimizers[agent_idx].step()
        
        return loss.item()
    
    def update_target_networks(self):
        for i in range(self.num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, iid in enumerate(self.intersection_ids):
            torch.save(
                self.q_networks[i].state_dict(),
                os.path.join(save_dir, f"dqn_{iid}.pth")
            )
        print(f"Models saved to {save_dir}")
    
    def load(self, save_dir):
        for i, iid in enumerate(self.intersection_ids):
            path = os.path.join(save_dir, f"dqn_{iid}.pth")
            if os.path.exists(path):
                self.q_networks[i].load_state_dict(torch.load(path))
                self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        print(f"Models loaded from {save_dir}")


def plot_training(rewards, losses, waitings, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    # Moving average helper
    def moving_avg(x, w=20):
        return np.convolve(x, np.ones(w)/w, mode='same')

    # Rewards
    plt.figure(figsize=(10,5))
    plt.plot(rewards, label="Episode Reward")
    plt.plot(moving_avg(rewards), label="Moving Avg (20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rewards.png")
    plt.show()

    # Loss
    plt.figure(figsize=(10,5))
    plt.plot(losses, label="Loss")
    plt.plot(moving_avg(losses), label="Moving Avg (20)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss.png")
    plt.show()

    # Waiting
    plt.figure(figsize=(10,5))
    plt.plot(waitings, label="Average Waiting")
    plt.plot(moving_avg(waitings), label="Moving Avg (20)")
    plt.xlabel("Episode")
    plt.ylabel("Waiting Vehicles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/waiting.png")
    plt.show()


def train_multi_dqn(config_path, num_episodes=200, save_dir="models/multi_dqn"):
    config = {
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'learning_rate': 0.001,
        'batch_size': 64,
        'target_update_freq': 10,
        'buffer_size': 10000
    }
    
    print("Initializing Multi-Agent Environment...")
    env = CityFlowMultiEnv(config_path, steps_per_action=10, reward_type='pressure')
    
    print("Creating Multi-Agent DQN...")
    agent = MultiAgentDQN(env, config)
    
    episode_rewards = []
    episode_losses = []
    episode_waitings = []
    
    best_reward = -1e18
    best_model_dir = os.path.join(save_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    
    print("Starting Training")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_loss = 0
        step_count = 0
        
        while not done:
            actions = agent.select_actions(state)
            next_state, rewards, done, truncated, info = env.step(actions)
            
            agent.store_transition(state, actions, rewards, next_state, done)
            
            loss = agent.train()
            if loss is not None:
                ep_loss += loss
            
            state = next_state
            ep_reward += np.sum(rewards)
            step_count += 1
        
        avg_loss = ep_loss / step_count if step_count > 0 else 0
        
        episode_rewards.append(ep_reward)
        episode_losses.append(avg_loss)
        episode_waitings.append(info["total_waiting"])
        
        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target_networks()
        
        agent.decay_epsilon()
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(best_model_dir)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}  Reward={ep_reward:.2f}  Loss={avg_loss:.3f}")
    
    agent.save(save_dir)
    
    plot_training(episode_rewards, episode_losses, episode_waitings)
    
    print("Training completed.")
    print(f"Best model saved in {best_model_dir}")
    
    return agent, episode_rewards


def test_multi_dqn(config_path, model_dir="models/multi_dqn/best_model", num_episodes=10):
    print("Testing Multi-Agent DQN")
    
    env = CityFlowMultiEnv(config_path, steps_per_action=10, reward_type='pressure')
    config = {'epsilon_start': 0.0}
    
    agent = MultiAgentDQN(env, config)
    agent.load(model_dir)
    
    test_rewards = []
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            actions = agent.select_actions(state, epsilon=0.0)
            next_state, rewards, done, truncated, info = env.step(actions)
            state = next_state
            total_reward += np.sum(rewards)
        
        print(f"Test Episode {ep+1}: Reward={total_reward}")
        test_rewards.append(total_reward)
    
    print("Test Average Reward:", np.mean(test_rewards))
    return test_rewards


if __name__ == "__main__":
    config_path = "examples/config.json"

    print("Starting training...")
    agent, rewards = train_multi_dqn(config_path)

    print("Starting testing...")
    test_multi_dqn(config_path)
