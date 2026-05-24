import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class _PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class QLearning:
    """Tabular Q-Learning agent for discrete state/action environments.

    Learns a Q-table Q[state, action] estimating expected future rewards via
    the Bellman update. Uses epsilon-greedy exploration with exponential decay.

    Parameters
    ----------
    alpha         : float, default=0.8   — learning rate
    gamma         : float, default=0.99  — discount factor for future rewards
    epsilon       : float, default=1.0   — initial exploration probability
    epsilon_decay : float, default=0.995 — multiplicative decay per episode
    epsilon_min   : float, default=0.01  — floor on exploration probability

    Attributes
    ----------
    Q             : ndarray  — learned Q-table (available after train)
    train_rewards : list     — total reward per training episode

    Example
    -------
    >>> agent  = QLearning().train(env, episodes=5_000)
    >>> report = agent.evaluate(env, episodes=500)
    >>> print(np.mean(agent.train_rewards[-100:]))
    """
    def __init__(self, alpha=0.8, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.Q             = None
        self.train_rewards = None

    def train(self, env, episodes=5_000):
        """Run Q-learning episodes and populate the Q-table.

        Parameters
        ----------
        env      : gymnasium.Env  — discrete state/action Gymnasium environment
        episodes : int, default=5000  — number of training episodes

        Returns
        -------
        self
        """
        alpha, gamma = self.alpha, self.gamma
        epsilon = self.epsilon
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        rewards_per_ep = []

        for _ in range(episodes):
            state, _ = env.reset()
            total_r  = 0
            done     = False
            while not done:
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                Q[state, action] += alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action]
                )
                state    = next_state
                total_r += reward
            rewards_per_ep.append(total_r)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)

        self.Q             = Q
        self.train_rewards = rewards_per_ep
        return self

    def predict(self, state):
        """Select the greedy action for a given state.

        Parameters
        ----------
        state : int  — current environment state index

        Returns
        -------
        action : int
        """
        return int(np.argmax(self.Q[state]))

    def evaluate(self, env, episodes=500):
        """Run greedy episodes and report performance metrics.

        Parameters
        ----------
        env      : gymnasium.Env
        episodes : int, default=500

        Returns
        -------
        dict with keys:
            win_rate          — fraction of episodes with positive reward
            avg_reward        — mean total reward per episode
            cumulative_reward — total reward across all evaluation episodes
        """
        wins, total_reward = 0, 0
        for _ in range(episodes):
            state, _ = env.reset()
            done      = False
            ep_reward = 0
            while not done:
                action = self.predict(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done      = terminated or truncated
                ep_reward += reward
            wins         += int(ep_reward > 0)
            total_reward += ep_reward
        return {
            "win_rate":          wins / episodes,
            "avg_reward":        total_reward / episodes,
            "cumulative_reward": total_reward,
        }


class DQNAgent:
    """Deep Q-Network (DQN) agent for continuous state / discrete action environments.

    Approximates Q(state, action) with a neural network. Uses experience replay
    and a separate target network for stable training.

    Parameters
    ----------
    gamma         : float, default=0.99    — discount factor
    lr            : float, default=1e-3    — Adam learning rate
    batch_size    : int,   default=64      — replay buffer sample size
    memory_size   : int,   default=10000   — maximum replay buffer capacity
    epsilon       : float, default=1.0     — initial exploration probability
    epsilon_decay : float, default=0.995   — per-episode decay multiplier
    epsilon_min   : float, default=0.01    — floor on exploration probability

    Attributes
    ----------
    policy        : _DQN   — trained policy network (available after train)
    train_rewards : list   — total reward per training episode

    Example
    -------
    >>> agent  = DQNAgent().train(env, episodes=300)
    >>> report = agent.evaluate(env, episodes=100)
    """
    def __init__(self, gamma=0.99, lr=1e-3, batch_size=64, memory_size=10_000,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.gamma         = gamma
        self.lr            = lr
        self.batch_size    = batch_size
        self.memory_size   = memory_size
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.policy        = None
        self.train_rewards = None

    def train(self, env, episodes=300):
        """Train the DQN agent.

        Parameters
        ----------
        env      : gymnasium.Env  — continuous state, discrete action environment
        episodes : int, default=300

        Returns
        -------
        self
        """
        s_dim  = env.observation_space.shape[0]
        a_dim  = env.action_space.n
        policy = _DQN(s_dim, a_dim)
        target = _DQN(s_dim, a_dim)
        target.load_state_dict(policy.state_dict())
        optimizer   = optim.Adam(policy.parameters(), lr=self.lr)
        criterion   = nn.MSELoss()
        memory      = deque(maxlen=self.memory_size)
        epsilon     = self.epsilon_start
        rewards_log = []

        for ep in range(episodes):
            state, _ = env.reset()
            total_r  = 0
            done     = False
            while not done:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy(
                            torch.tensor(state, dtype=torch.float32)
                        ).argmax().item()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                memory.append((state, action, reward, next_state, float(done)))
                state    = next_state
                total_r += reward

                if len(memory) >= self.batch_size:
                    batch     = random.sample(memory, self.batch_size)
                    states, actions, rewards, nexts, dones = zip(*batch)
                    states_t  = torch.tensor(states,  dtype=torch.float32)
                    nexts_t   = torch.tensor(nexts,   dtype=torch.float32)
                    rewards_t = torch.tensor(rewards, dtype=torch.float32)
                    dones_t   = torch.tensor(dones,   dtype=torch.float32)
                    actions_t = torch.tensor(actions, dtype=torch.long)
                    q_vals    = policy(states_t).gather(
                        1, actions_t.unsqueeze(1)).squeeze()
                    with torch.no_grad():
                        q_next = target(nexts_t).max(1).values
                    targets_t = rewards_t + self.gamma * q_next * (1 - dones_t)
                    loss      = criterion(q_vals, targets_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rewards_log.append(total_r)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            if (ep + 1) % 50 == 0:
                target.load_state_dict(policy.state_dict())
                print(f"Episode {ep+1}  "
                      f"avg_reward={np.mean(rewards_log[-50:]):.1f}  "
                      f"eps={epsilon:.3f}")

        self.policy        = policy
        self.train_rewards = rewards_log
        return self

    def predict(self, state):
        """Select the greedy action for a given state.

        Parameters
        ----------
        state : array-like  — current environment observation

        Returns
        -------
        action : int
        """
        self.policy.eval()
        with torch.no_grad():
            return self.policy(
                torch.tensor(state, dtype=torch.float32)
            ).argmax().item()

    def evaluate(self, env, episodes=100):
        """Run greedy episodes and report performance metrics.

        Parameters
        ----------
        env      : gymnasium.Env
        episodes : int, default=100

        Returns
        -------
        dict with keys:
            avg_reward        — mean total reward per episode
            cumulative_reward — total reward across all evaluation episodes
            avg_steps         — mean steps per episode
        """
        total_reward, total_steps = 0, 0
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.predict(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done          = terminated or truncated
                total_reward += reward
                total_steps  += 1
        return {
            "avg_reward":        total_reward / episodes,
            "cumulative_reward": total_reward,
            "avg_steps":         total_steps / episodes,
        }


class PolicyGradientAgent:
    """REINFORCE policy gradient agent for discrete action environments.

    Directly parameterises a stochastic policy with a neural network and
    optimises it by maximising expected return via Monte Carlo gradient estimates.

    Parameters
    ----------
    gamma : float, default=0.99  — discount factor for computing returns
    lr    : float, default=1e-3  — Adam learning rate

    Attributes
    ----------
    policy        : _PolicyNet  — trained policy network (available after train)
    train_rewards : list        — total reward per training episode

    Example
    -------
    >>> agent  = PolicyGradientAgent().train(env, episodes=500)
    >>> report = agent.evaluate(env, episodes=100)
    """
    def __init__(self, gamma=0.99, lr=1e-3):
        self.gamma         = gamma
        self.lr            = lr
        self.policy        = None
        self.train_rewards = None

    def train(self, env, episodes=500):
        """Train the policy network using the REINFORCE algorithm.

        Parameters
        ----------
        env      : gymnasium.Env  — discrete action environment
        episodes : int, default=500

        Returns
        -------
        self
        """
        s_dim     = env.observation_space.shape[0]
        a_dim     = env.action_space.n
        policy    = _PolicyNet(s_dim, a_dim)
        optimizer = optim.Adam(policy.parameters(), lr=self.lr)
        rewards_log = []

        for ep in range(episodes):
            state, _   = env.reset()
            log_probs  = []
            ep_rewards = []
            done       = False

            while not done:
                state_t = torch.tensor(state, dtype=torch.float32)
                probs   = policy(state_t)
                dist    = torch.distributions.Categorical(probs)
                action  = dist.sample()
                log_probs.append(dist.log_prob(action))
                state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                ep_rewards.append(reward)

            # Compute normalised discounted returns
            returns = []
            G = 0
            for r in reversed(ep_rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns_t = torch.tensor(returns, dtype=torch.float32)
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            loss = -torch.stack(log_probs).dot(returns_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            rewards_log.append(sum(ep_rewards))
            if (ep + 1) % 100 == 0:
                print(f"Episode {ep+1}  "
                      f"avg_reward={np.mean(rewards_log[-100:]):.1f}")

        self.policy        = policy
        self.train_rewards = rewards_log
        return self

    def predict(self, state):
        """Select the most probable action for a given state.

        Parameters
        ----------
        state : array-like  — current environment observation

        Returns
        -------
        action : int
        """
        self.policy.eval()
        with torch.no_grad():
            probs = self.policy(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(probs).item()

    def evaluate(self, env, episodes=100):
        """Run greedy episodes and report performance metrics.

        Parameters
        ----------
        env      : gymnasium.Env
        episodes : int, default=100

        Returns
        -------
        dict with keys:
            avg_reward        — mean total reward per episode
            cumulative_reward — total reward across all evaluation episodes
        """
        total_reward = 0
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.predict(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done          = terminated or truncated
                total_reward += reward
        return {
            "avg_reward":        total_reward / episodes,
            "cumulative_reward": total_reward,
        }
