import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from cityflow_env import CityFlowEnv  # environment
import matplotlib.pyplot as plt
import os

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# DQN
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # main and target network
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Epsilon-greedy params
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # reply buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        
    def select_action(self, state, training=True):
        #choose actions(epsilon-greedy)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def update(self):
        #update the internet
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # trainsform
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # calculate current q 
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # calculate target q
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # calculate loss and update
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        #update target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        #decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# train
def train(config_path="examples/config.json", episodes=500, max_steps=360, 
          steps_per_action=10, update_target_every=10):
    
    print("create environment...")
    env = CityFlowEnv(config_path=config_path, steps_per_action=steps_per_action)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"state dim: {state_dim}, action dim: {action_dim}")
    print(f"Actual simulation steps per turn: {max_steps * steps_per_action}")
    
    agent = DQNAgent(state_dim, action_dim)
    
    # record training data
    episode_rewards = []
    episode_losses = []
    episode_avg_waiting = []  # record average waiting vehicles numbers
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # choose actions
            action = agent.select_action(state)
            
            next_state, reward, done, truncated, _ = env.step(action)
            
            # save experience
            agent.replay_buffer.push(state, action, reward, next_state, done or truncated)
            
            # update
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        # update target network
        if (episode + 1) % update_target_every == 0:
            agent.update_target_network()
        
        # decay epsilon
        agent.decay_epsilon()
        
        # record
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        episode_losses.append(avg_loss)
        
        # calculate average waiting vehicles
        avg_waiting = -episode_reward / max_steps if max_steps > 0 else 0
        episode_avg_waiting.append(avg_waiting)
        
        # print
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_waiting_10 = np.mean(episode_avg_waiting[-10:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (last 10): {avg_reward:.2f} | "
                  f"Avg Waiting: {avg_waiting:.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # save models
    os.makedirs("models", exist_ok=True)
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'config_path': config_path,
        'state_dim': state_dim,
        'action_dim': action_dim
    }, "models/dqn_cityflow.pth")
    print("\nmodel saved models/dqn_cityflow.pth")
    
    # plot
    plot_training_curves(episode_rewards, episode_losses, episode_avg_waiting)
    
    return agent, env

# plot
def plot_training_curves(rewards, losses, avg_waiting):
    plt.figure(figsize=(15, 5))
    
    # reward curve
    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.6, label='Episode Reward', color='blue')

    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                label=f'Moving Avg ({window})', linewidth=2, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # loss curve
    plt.subplot(1, 3, 2)
    plt.plot(losses, alpha=0.6, label='Loss', color='green')
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), moving_avg, 
                label=f'Moving Avg ({window})', linewidth=2, color='darkgreen')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # avg waiting vehicles curve
    plt.subplot(1, 3, 3)
    plt.plot(avg_waiting, alpha=0.6, label='Avg Waiting Vehicles', color='orange')
    if len(avg_waiting) >= window:
        moving_avg = np.convolve(avg_waiting, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(avg_waiting)), moving_avg, 
                label=f'Moving Avg ({window})', linewidth=2, color='darkorange')
    plt.xlabel('Episode')
    plt.ylabel('Avg Waiting Vehicles')
    plt.title('Average Waiting Vehicles per Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("train curve saved in training_curves.png")
    plt.close()

# test function
def test(agent, config_path="examples/config.json", episodes=10, 
         steps_per_action=10, max_steps=360):
    #test trained agent
    env = CityFlowEnv(config_path=config_path, steps_per_action=steps_per_action)
    test_rewards = []
    test_avg_waiting = []
    
    print("\n" + "="*60)
    print("start test...")
    print("="*60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # greedy policy
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break
        
        avg_waiting = -episode_reward / max_steps if max_steps > 0 else 0
        test_rewards.append(episode_reward)
        test_avg_waiting.append(avg_waiting)
        
        print(f"Test Episode {episode+1}: "
              f"Reward = {episode_reward:.2f}, "
              f"Avg Waiting = {avg_waiting:.2f}")
    
    print("\n" + "="*60)
    print(f"test outcome:")
    print(f"  average rewards: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  average waiting vehicles: {np.mean(test_avg_waiting):.2f} ± {np.std(test_avg_waiting):.2f}")
    print("="*60)
    
    return test_rewards, test_avg_waiting

# load trained model
def load_model(model_path="models/dqn_cityflow.pth"):
   
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dim = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(checkpoint['policy_net'])
    agent.target_net.load_state_dict(checkpoint['target_net'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"load {model_path} ")
    print(f"state dim: {state_dim}, action dim: {action_dim}")
    
    return agent, checkpoint['config_path']

if __name__ == "__main__":
    print("=" * 60)
    print("Start training")
    print("=" * 60)
    
    # train params
    CONFIG_PATH = "examples/config.json"  # CityFlow
    EPISODES = 500                         # episodes
    MAX_STEPS = 360                        
    STEPS_PER_ACTION = 10                 
    
    # train
    agent, env = train(
        config_path=CONFIG_PATH,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        steps_per_action=STEPS_PER_ACTION,
        update_target_every=10
    )
    
    # test
    test(agent, config_path=CONFIG_PATH, episodes=10, 
         steps_per_action=STEPS_PER_ACTION, max_steps=MAX_STEPS)
    
    print("\nFinish train and test!")
    print("- Model saved: models/dqn_cityflow.pth")
    print("- Train curves saved: training_curves.png")