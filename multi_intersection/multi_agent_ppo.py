import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    """Actor-Critic network with improved architecture"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared feature extractor (deeper network)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def act(self, state):
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate(self, states, actions):
        action_logits, values = self.forward(states)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


class MultiAgentPPO:
    """
    Multi-Agent PPO with optimized hyperparameters
    """

    def __init__(
            self,
            num_agents,
            state_dim,
            action_dims,
            lr=5e-4,  # Increased learning rate
            gamma=0.98,  # Slightly lower gamma
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.05,  # Increased entropy for more exploration
            max_grad_norm=0.5,
            ppo_epochs=4,  # Fewer epochs per update
            mini_batch_size=128,  # Larger batch size
            device='cpu'
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device

        # Create policies
        self.policies = []
        self.optimizers = []

        for i in range(num_agents):
            policy = ActorCritic(
                state_dim=state_dim,
                action_dim=action_dims[i],
                hidden_dim=256
            ).to(device)

            # Use Adam with weight decay
            optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)

            self.policies.append(policy)
            self.optimizers.append(optimizer)

        # Schedulers for learning rate decay
        self.schedulers = [
            optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.95)
            for opt in self.optimizers
        ]

        self.reset_buffers()

    def reset_buffers(self):
        self.buffers = [
            {
                'states': [],
                'actions': [],
                'log_probs': [],
                'rewards': [],
                'values': [],
                'dones': []
            }
            for _ in range(self.num_agents)
        ]

    def select_actions(self, states):
        """Select actions with exploration"""
        actions = []
        log_probs = []
        values = []

        for i in range(self.num_agents):
            # States is now (num_agents, 2), extract features for agent i
            state = torch.FloatTensor(states[i]).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.policies[i].act(state)

            actions.append(action)
            log_probs.append(log_prob.item())
            values.append(value.item())

        return np.array(actions), log_probs, values

    def store_transition(self, states, actions, log_probs, rewards, values, done):
        for i in range(self.num_agents):
            self.buffers[i]['states'].append(states[i])
            self.buffers[i]['actions'].append(actions[i])
            self.buffers[i]['log_probs'].append(log_probs[i])
            self.buffers[i]['rewards'].append(rewards[i])
            self.buffers[i]['values'].append(values[i])
            self.buffers[i]['dones'].append(done)

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, next_states):
        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for agent_idx in range(self.num_agents):
            buffer = self.buffers[agent_idx]

            if len(buffer['states']) == 0:
                continue

            # Get next value
            next_state = torch.FloatTensor(next_states[agent_idx]).to(self.device)
            with torch.no_grad():
                _, next_value = self.policies[agent_idx](next_state)
                next_value = next_value.item()

            # Compute GAE
            advantages, returns = self.compute_gae(
                buffer['rewards'],
                buffer['values'],
                buffer['dones'],
                next_value
            )

            # Convert to tensors
            states = torch.FloatTensor(buffer['states']).to(self.device)
            actions = torch.LongTensor(buffer['actions']).to(self.device)
            old_log_probs = torch.FloatTensor(buffer['log_probs']).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            dataset_size = len(states)

            for epoch in range(self.ppo_epochs):
                indices = np.random.permutation(dataset_size)

                for start in range(0, dataset_size, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, dataset_size)
                    batch_indices = indices[start:end]

                    if len(batch_indices) < 2:
                        continue

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    # Evaluate
                    log_probs, values, entropy = self.policies[agent_idx].evaluate(
                        batch_states, batch_actions
                    )

                    # PPO loss
                    ratios = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                    entropy_loss = -entropy.mean()

                    loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                    # Optimize
                    self.optimizers[agent_idx].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policies[agent_idx].parameters(), self.max_grad_norm)
                    self.optimizers[agent_idx].step()

                    total_loss += loss.item()
                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    total_entropy += entropy.mean().item()

            # Step scheduler
            self.schedulers[agent_idx].step()

        self.reset_buffers()

        metrics = {
            'total_loss': total_loss / self.num_agents,
            'actor_loss': total_actor_loss / self.num_agents,
            'critic_loss': total_critic_loss / self.num_agents,
            'entropy': total_entropy / self.num_agents
        }

        return metrics

    def save(self, path):
        checkpoint = {
            'num_agents': self.num_agents,
            'policies': [policy.state_dict() for policy in self.policies],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'schedulers': [sched.state_dict() for sched in self.schedulers]
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for i in range(self.num_agents):
            self.policies[i].load_state_dict(checkpoint['policies'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizers'][i])
            if 'schedulers' in checkpoint:
                self.schedulers[i].load_state_dict(checkpoint['schedulers'][i])
