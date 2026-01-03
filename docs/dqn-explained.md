# Deep Q-Learning Explained

This guide provides a comprehensive introduction to Deep Q-Learning (DQN), the algorithm that revolutionized reinforcement learning by combining neural networks with Q-learning. By the end, you'll understand how DQN works and why it's particularly powerful for game playing.

## Table of Contents

1. [From Tabular Q-Learning to Deep Q-Learning](#from-tabular-q-learning-to-deep-q-learning)
2. [Neural Network Architecture for Game Playing](#neural-network-architecture-for-game-playing)
3. [Experience Replay](#experience-replay)
4. [Target Networks](#target-networks)
5. [The DQN Algorithm Step by Step](#the-dqn-algorithm-step-by-step)
6. [Key Hyperparameters](#key-hyperparameters)

---

## From Tabular Q-Learning to Deep Q-Learning

### Recap: Tabular Q-Learning

In tabular Q-learning, we maintain a Q-table that maps every (state, action) pair to an expected future reward:

```
Q-Table Structure:
+------------------+----------+----------+----------+
|      State       | Action 0 | Action 1 | Action 2 |
+------------------+----------+----------+----------+
| State A          |   0.5    |   0.8    |   0.2    |
| State B          |   0.3    |   0.6    |   0.9    |
| State C          |   0.7    |   0.1    |   0.4    |
| ...              |   ...    |   ...    |   ...    |
+------------------+----------+----------+----------+
```

The Q-value update rule is:

```
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a))
```

Where:
- `s` is the current state
- `a` is the action taken
- `s'` is the next state
- `alpha` is the learning rate
- `gamma` is the discount factor

**The Problem:** This works beautifully for small state spaces (like tic-tac-toe with ~5,000 states), but what happens when you have millions or trillions of possible states? You simply cannot store a table that large.

### The Deep Learning Solution

Instead of storing Q-values in a table, we use a neural network to **approximate** the Q-function:

```
Tabular:     Q-Table[state][action] -> Q-value
Deep:        NeuralNetwork(state) -> [Q-value for each action]
```

```
                    Tabular Q-Learning              Deep Q-Learning

                    +-------------+                 +-----------+
    State  -------> |   Q-Table   | -> Q-values     |  Neural   |
                    | (explicit)  |                 |  Network  | -> Q-values
                    +-------------+                 +-----------+

    Memory:         O(|S| x |A|)                    O(parameters)
    Lookup:         Exact                           Approximation
    Generalization: None                            Yes!
```

**Key Insight:** The neural network can generalize from states it has seen to similar states it hasn't seen. This is crucial for large state spaces where we can never visit every possible state.

---

## Neural Network Architecture for Game Playing

For board games like Connect Four, we need a network that can:
1. Take a board state as input
2. Output Q-values for each possible action (column to drop a piece)

### Simple Feedforward Architecture

```python
import torch
import torch.nn as nn

class SimpleDQN(nn.Module):
    def __init__(self, board_size=42, num_actions=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, state):
        return self.network(state)
```

```
Input Layer (42 neurons - one per board cell)
        |
        v
    [Hidden Layer 1: 128 neurons + ReLU]
        |
        v
    [Hidden Layer 2: 128 neurons + ReLU]
        |
        v
    Output Layer (7 neurons - one Q-value per column)
```

### Convolutional Architecture (Better for Board Games)

Convolutional Neural Networks (CNNs) excel at detecting spatial patterns - exactly what we need for board games:

```python
class ConvDQN(nn.Module):
    def __init__(self, num_actions=7):
        super().__init__()

        # Convolutional layers for pattern detection
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, padding=1),  # 2 channels: player pieces, opponent pieces
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Fully connected layers for decision making
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, state):
        features = self.conv_layers(state)
        return self.fc_layers(features)
```

```
Input: 2 x 6 x 7 board representation
     (channels x height x width)
        |
        v
[Conv2D: 64 filters, 4x4 kernel] -> Detects small patterns
        |
        v
[Conv2D: 64 filters, 3x3 kernel] -> Combines into larger patterns
        |
        v
[Conv2D: 64 filters, 3x3 kernel] -> Higher-level features
        |
        v
[Flatten + Dense layers] -> Maps features to action Q-values
        |
        v
Output: 7 Q-values (one per column)
```

---

## Experience Replay

### The Problem: Correlated Samples

When learning online (updating after each step), consecutive experiences are highly correlated:

```
Episode 1, Step 1: State_A -> Action -> State_B
Episode 1, Step 2: State_B -> Action -> State_C
Episode 1, Step 3: State_C -> Action -> State_D
...
```

These correlated samples cause problems:
1. **Inefficient learning:** The network overfits to recent experiences
2. **Catastrophic forgetting:** Learning new things erases old knowledge
3. **Unstable training:** Correlated updates can cause oscillations

### The Solution: Experience Replay Buffer

Store experiences in a buffer and sample randomly for training:

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
```

### How It Works

```
During Play:                          During Training:

+-------------------+                 +-------------------+
| Experience 1      | ----+           | Sample randomly:  |
| (s, a, r, s', d)  |     |           |                   |
+-------------------+     |           | Experience 847    |
| Experience 2      |     |           | Experience 12     |
| (s, a, r, s', d)  |     +---------> | Experience 3391   |
+-------------------+     |           | Experience 156    |
| Experience 3      |     |           | ...               |
| (s, a, r, s', d)  | ----+           |                   |
+-------------------+                 +-------------------+
| ...               |                        |
+-------------------+                        v
                                      Train on this batch
```

### Benefits of Experience Replay

1. **Breaks correlations:** Random sampling ensures diverse training batches
2. **Data efficiency:** Each experience can be used multiple times
3. **Stable learning:** Smoother gradient updates from varied samples

```python
# Training loop with experience replay
def train_step(dqn, target_net, replay_buffer, optimizer, batch_size=32, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return  # Wait until we have enough experiences

    # Sample random batch
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert to tensors
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Compute current Q-values
    current_q = dqn(states).gather(1, actions.unsqueeze(1))

    # Compute target Q-values (using target network)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        target_q = rewards + gamma * max_next_q * (1 - dones)

    # Update network
    loss = nn.MSELoss()(current_q.squeeze(), target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Target Networks

### The Moving Target Problem

In Q-learning, we update Q-values toward a target:

```
Target = reward + gamma * max(Q(next_state, a))
```

But here's the problem: **we're using the same network to compute both the current Q-value AND the target Q-value.** Every time we update the network, the targets change too!

```
Step 1: Q(s,a) = 0.5,  Target = 0.8  -> Update Q toward 0.8
Step 2: Q(s,a) = 0.65, Target = 0.75 -> Target changed! Update toward 0.75
Step 3: Q(s,a) = 0.7,  Target = 0.72 -> Target changed again!
...

This is like trying to hit a moving bullseye!
```

This causes training instability and can lead to divergence.

### The Solution: Two Networks

Maintain two copies of the network:
1. **Online Network:** Updated every step, used to select actions
2. **Target Network:** Updated periodically, used to compute targets

```
                 +------------------+
    State  ----> | Online Network   | ----> Actions (epsilon-greedy)
                 | (updated often)  |
                 +------------------+
                          |
                          | Copy weights periodically
                          v
                 +------------------+
Next State ----> | Target Network   | ----> Target Q-values
                 | (frozen mostly)  |
                 +------------------+
```

### Implementation

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, target_update_freq=1000):
        self.online_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)

        # Initialize target network with same weights
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.target_update_freq = target_update_freq
        self.steps = 0

    def update_target_network(self):
        """Copy weights from online network to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train_step(self, batch):
        # ... training code ...

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
```

### Soft Updates (Alternative Approach)

Instead of periodic hard copies, gradually blend weights:

```python
def soft_update(self, tau=0.001):
    """Slowly blend target network toward online network."""
    for target_param, online_param in zip(
        self.target_net.parameters(),
        self.online_net.parameters()
    ):
        target_param.data.copy_(
            tau * online_param.data + (1 - tau) * target_param.data
        )
```

This provides smoother, more stable target updates.

---

## The DQN Algorithm Step by Step

Here's the complete DQN algorithm:

```
Algorithm: Deep Q-Learning with Experience Replay

Initialize:
    - Online network Q with random weights theta
    - Target network Q_target with weights theta_target = theta
    - Replay buffer D with capacity N
    - Exploration rate epsilon = 1.0

For each episode:
    Reset environment, get initial state s

    For each step:
        1. SELECT ACTION
           With probability epsilon:
               a = random action
           Otherwise:
               a = argmax_a Q(s, a; theta)

        2. EXECUTE ACTION
           Take action a, observe reward r and next state s'
           Store transition (s, a, r, s', done) in D

        3. SAMPLE AND TRAIN
           Sample random minibatch of transitions from D
           For each transition (s_j, a_j, r_j, s'_j, done_j):
               If done_j:
                   y_j = r_j
               Else:
                   y_j = r_j + gamma * max_a' Q_target(s'_j, a'; theta_target)

           Update theta by minimizing:
               Loss = (y_j - Q(s_j, a_j; theta))^2

        4. UPDATE TARGET NETWORK
           Every C steps: theta_target = theta

        5. DECAY EXPLORATION
           epsilon = max(epsilon_min, epsilon * decay_rate)

        s = s'
```

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.online_net = self._build_network(state_dim, action_dim)
        self.target_net = self._build_network(state_dim, action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        self.steps = 0

    def _build_network(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def choose_action(self, state, legal_actions=None):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            if legal_actions is not None:
                return random.choice(legal_actions)
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.online_net(state_tensor).squeeze()

            if legal_actions is not None:
                # Mask illegal actions with very negative values
                mask = torch.ones(self.action_dim) * float('-inf')
                for a in legal_actions:
                    mask[a] = 0
                q_values = q_values + mask

            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

---

## Key Hyperparameters

### Learning Rate (alpha)

Controls how much the network weights change per update.

```
Too high (e.g., 0.01):   Unstable, may diverge
Too low (e.g., 0.00001): Very slow learning
Good range: 1e-4 to 1e-3
```

**Tip:** Start with 1e-4 and adjust based on loss curves.

### Discount Factor (gamma)

How much the agent values future rewards vs immediate rewards.

```
gamma = 0.0:  Only cares about immediate reward
gamma = 0.5:  Balanced short/long term
gamma = 0.99: Values future rewards highly (typical for games)
gamma = 1.0:  Future rewards equal to immediate (can cause instability)
```

**For games:** Use 0.99 since winning at the end matters.

### Epsilon Decay Schedule

Controls exploration vs exploitation over time.

```
Episode:    1      1000    5000    10000   50000
            |       |       |       |       |
Epsilon:   1.0 --> 0.5 --> 0.2 --> 0.1 --> 0.01
            |       |       |       |       |
Behavior:  Random  Mixed   Mostly  Mostly  Exploit
                           learned learned
```

Common decay strategies:

```python
# Multiplicative decay (most common)
epsilon = max(epsilon_min, epsilon * decay_rate)  # decay_rate = 0.995

# Linear decay
epsilon = max(epsilon_min, epsilon_start - episode * decay_step)

# Exponential decay
epsilon = epsilon_min + (epsilon_start - epsilon_min) * exp(-episode / decay_constant)
```

### Batch Size

Number of experiences sampled per training step.

```
Small (16-32):   More updates, noisier gradients
Medium (64-128): Good balance (recommended)
Large (256+):    Smoother gradients, more memory, fewer updates
```

### Replay Buffer Size

How many experiences to store.

```
Small (1K-10K):     Forgets quickly, may not break correlations well
Medium (50K-100K):  Good balance for most applications
Large (1M+):        More diverse samples, but more memory
```

**Consideration:** Buffer should be large enough to contain experiences from multiple episodes and policies.

### Target Network Update Frequency

How often to sync the target network.

```
Too frequent (every step):      Defeats the purpose, targets still move
Too rare (every 50K steps):     Targets become very stale
Good range: 1000-10000 steps
```

### Hyperparameter Summary Table

| Parameter | Typical Range | Connect Four Suggestion |
|-----------|---------------|------------------------|
| Learning rate | 1e-4 to 1e-3 | 5e-4 |
| Gamma | 0.95 to 0.99 | 0.99 |
| Epsilon start | 1.0 | 1.0 |
| Epsilon end | 0.01 to 0.1 | 0.01 |
| Epsilon decay | 0.99 to 0.9995 | 0.9995 |
| Batch size | 32 to 128 | 64 |
| Buffer size | 50K to 500K | 100K |
| Target update | 1K to 10K steps | 5000 |

---

## Summary

Deep Q-Learning extends tabular Q-learning to handle large state spaces by:

1. **Neural Network Approximation:** Replaces the Q-table with a neural network that can generalize across states

2. **Experience Replay:** Stores and randomly samples past experiences to:
   - Break correlations between consecutive samples
   - Reuse experiences for data efficiency
   - Stabilize training

3. **Target Networks:** Uses a separate, slowly-updated network for computing targets to:
   - Prevent chasing moving targets
   - Stabilize the learning process

These three innovations together made it possible to train agents that achieve human-level performance on complex games directly from raw inputs.

## Further Reading

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Nature paper with full details
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) - Modern DQN improvements
