"""
Deep Q-Learning Agent for Connect Four

This module implements a DQN agent with:
- Neural network for Q-value approximation
- Experience replay buffer
- Target network for training stability
- Epsilon-greedy exploration
"""

from typing import List, Tuple, Optional
from collections import deque
import random
import numpy as np

# Neural network imports (will be added when implementing DQN)
# import torch
# import torch.nn as nn
# import torch.optim as optim


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a random batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for Connect Four.

    Uses a neural network to approximate Q-values, enabling generalization
    across the massive state space (~4.5 trillion positions).
    """

    def __init__(
        self,
        state_shape: Tuple[int, int] = (6, 7),
        n_actions: int = 7,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000
    ):
        """
        Initialize DQN agent.

        Args:
            state_shape: Shape of the board (rows, cols)
            n_actions: Number of possible actions (columns)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per step
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training statistics
        self.training_steps = 0
        self.episodes_trained = 0

        # TODO: Initialize neural networks
        # self.policy_net = self._build_network()
        # self.target_net = self._build_network()
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def choose_action(self, state: np.ndarray, legal_actions: List[int],
                      training: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current board state
            legal_actions: List of valid column indices
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Chosen column index
        """
        if training and random.random() < self.epsilon:
            # Explore: random legal action
            return random.choice(legal_actions)
        else:
            # Exploit: best action according to Q-network
            # TODO: Implement neural network inference
            # For now, return random action
            return random.choice(legal_actions)

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using a batch from replay buffer.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # TODO: Implement training step
        # states, actions, rewards, next_states, dones = zip(*batch)
        # ... compute loss and update network ...

        self.training_steps += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self._update_target_network()

        return 0.0  # Placeholder

    def _update_target_network(self):
        """Copy weights from policy network to target network."""
        # TODO: Implement target network update
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        pass

    def save(self, filepath: str):
        """Save the agent to a file."""
        # TODO: Implement model saving
        pass

    def load(self, filepath: str):
        """Load the agent from a file."""
        # TODO: Implement model loading
        pass

    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            'episodes': self.episodes_trained,
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer)
        }


if __name__ == "__main__":
    # Quick test
    agent = DQNAgent()
    print(f"Agent initialized with epsilon: {agent.epsilon}")
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
