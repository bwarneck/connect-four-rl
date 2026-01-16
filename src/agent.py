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

# Neural network imports
import torch
import torch.nn as nn
import torch.optim as optim


class ConnectFourDQN(nn.Module):
    """
    Deep Q-Network for Connect Four.

    Uses convolutional layers to detect spatial patterns (like 4-in-a-row)
    followed by fully connected layers for Q-value estimation.

    Input: Board state with 3 channels:
        - Channel 0: Current player's pieces (1 where player has piece, 0 otherwise)
        - Channel 1: Opponent's pieces (1 where opponent has piece, 0 otherwise)
        - Channel 2: Empty cells (1 where empty, 0 otherwise)

    Output: Q-values for each of the 7 columns
    """

    def __init__(self, rows: int = 6, cols: int = 7, n_actions: int = 7):
        super(ConnectFourDQN, self).__init__()

        self.rows = rows
        self.cols = cols
        self.n_actions = n_actions

        # Convolutional layers to detect spatial patterns
        # Input: 3 channels x 6 rows x 7 cols
        self.conv_layers = nn.Sequential(
            # First conv layer: 3 input channels -> 64 filters
            nn.Conv2d(3, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Second conv layer: 64 -> 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            # Third conv layer: 128 -> 128 filters
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Calculate the size after conv layers
        # With the padding we use, the spatial dimensions are roughly preserved
        # Let's compute it dynamically
        self._conv_output_size = self._get_conv_output_size(rows, cols)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self._conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_actions)
        )

    def _get_conv_output_size(self, rows: int, cols: int) -> int:
        """Calculate the output size of conv layers."""
        # Create a dummy input to compute output size
        dummy_input = torch.zeros(1, 3, rows, cols)
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        return int(np.prod(output.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 6, 7)

        Returns:
            Q-values tensor of shape (batch_size, 7)
        """
        # Pass through conv layers
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through FC layers
        q_values = self.fc_layers(x)

        return q_values


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
             next_state: np.ndarray, done: bool, current_player: int = 1):
        """Store a transition in the buffer.

        Args:
            state: Board state when action was taken
            action: Column chosen
            reward: Reward received
            next_state: Resulting board state
            done: Whether game ended
            current_player: Player who made this move (1 or 2)
        """
        self.buffer.append((state, action, reward, next_state, done, current_player))

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
        target_update_freq: int = 1000,
        device: Optional[str] = None
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
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
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

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"DQN Agent using device: {self.device}")

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training statistics
        self.training_steps = 0
        self.episodes_trained = 0

        # Initialize neural networks
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self._update_target_network()  # Copy initial weights

        # Target network is not trained directly
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_network(self) -> ConnectFourDQN:
        """Create and return a new DQN network."""
        network = ConnectFourDQN(
            rows=self.state_shape[0],
            cols=self.state_shape[1],
            n_actions=self.n_actions
        )
        return network.to(self.device)

    def _preprocess_state(self, state: np.ndarray, current_player: int = 1) -> torch.Tensor:
        """
        Convert board state to neural network input format.

        Args:
            state: Board state array (6, 7) with values 0 (empty), 1 (player 1), 2 (player 2)
            current_player: The player whose turn it is (1 or 2)

        Returns:
            Tensor of shape (1, 3, 6, 7) with channels for:
                - Current player's pieces
                - Opponent's pieces
                - Empty cells
        """
        opponent = 2 if current_player == 1 else 1

        # Create 3 binary channels
        current_pieces = (state == current_player).astype(np.float32)
        opponent_pieces = (state == opponent).astype(np.float32)
        empty_cells = (state == 0).astype(np.float32)

        # Stack into channels
        processed = np.stack([current_pieces, opponent_pieces, empty_cells], axis=0)

        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(processed).unsqueeze(0).to(self.device)

        return tensor

    def _preprocess_batch(self, states: List[np.ndarray],
                          current_players: Optional[List[int]] = None) -> torch.Tensor:
        """
        Preprocess a batch of states.

        Args:
            states: List of board state arrays
            current_players: List of current player values (default: all 1s)

        Returns:
            Tensor of shape (batch_size, 3, 6, 7)
        """
        if current_players is None:
            current_players = [1] * len(states)

        batch = []
        for state, player in zip(states, current_players):
            opponent = 2 if player == 1 else 1
            current_pieces = (state == player).astype(np.float32)
            opponent_pieces = (state == opponent).astype(np.float32)
            empty_cells = (state == 0).astype(np.float32)
            processed = np.stack([current_pieces, opponent_pieces, empty_cells], axis=0)
            batch.append(processed)

        return torch.FloatTensor(np.array(batch)).to(self.device)

    def choose_action(self, state: np.ndarray, legal_actions: List[int],
                      training: bool = True, current_player: int = 1) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current board state
            legal_actions: List of valid column indices
            training: If True, use epsilon-greedy; if False, use greedy
            current_player: The player making the move (1 or 2)

        Returns:
            Chosen column index
        """
        if training and random.random() < self.epsilon:
            # Explore: random legal action
            return random.choice(legal_actions)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                self.policy_net.eval()

                # Preprocess state
                state_tensor = self._preprocess_state(state, current_player)

                # Get Q-values for all actions
                q_values = self.policy_net(state_tensor).squeeze(0)  # Shape: (7,)

                # Mask illegal actions by setting their Q-values to -infinity
                masked_q_values = torch.full_like(q_values, float('-inf'))
                for action in legal_actions:
                    masked_q_values[action] = q_values[action]

                # Select action with highest Q-value
                best_action = masked_q_values.argmax().item()

                self.policy_net.train()

            return best_action

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool, current_player: int = 1):
        """Store a transition in the replay buffer.

        Args:
            state: Board state when action was taken
            action: Column chosen
            reward: Reward received
            next_state: Resulting board state
            done: Whether game ended
            current_player: Player who made this move (1 or 2)
        """
        self.replay_buffer.push(state, action, reward, next_state, done, current_player)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step using a batch from replay buffer.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch - now includes current_player
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, current_players = zip(*batch)

        # Convert current_players to list for preprocessing
        current_players_list = list(current_players)

        # For next_state, the perspective switches to the opponent
        # (after player X moves, player Y will be choosing the next action)
        next_players_list = [2 if p == 1 else 1 for p in current_players_list]

        # Convert to tensors with correct player perspectives
        state_batch = self._preprocess_batch(list(states), current_players_list)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = self._preprocess_batch(list(next_states), next_players_list)
        done_batch = torch.BoolTensor(dones).to(self.device)

        # Compute current Q-values: Q(s, a)
        self.policy_net.train()
        current_q_values = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            max_next_q_values = next_q_values.max(1)[0]

            # Target: r + gamma * max(Q_target(s', a')) for non-terminal states
            # For terminal states (done=True), target is just r
            target_q_values = reward_batch + (~done_batch).float() * self.gamma * max_next_q_values

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network periodically
        if self.training_steps % self.target_update_freq == 0:
            self._update_target_network()

        return loss.item()

    def _update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str):
        """
        Save the agent to a file.

        Args:
            filepath: Path to save the model (should end with .pth)
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes_trained': self.episodes_trained,
            'state_shape': self.state_shape,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """
        Load the agent from a file.

        Args:
            filepath: Path to the saved model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes_trained = checkpoint['episodes_trained']

        # Restore hyperparameters if they exist
        if 'learning_rate' in checkpoint:
            self.learning_rate = checkpoint['learning_rate']
        if 'gamma' in checkpoint:
            self.gamma = checkpoint['gamma']
        if 'epsilon_end' in checkpoint:
            self.epsilon_end = checkpoint['epsilon_end']
        if 'epsilon_decay' in checkpoint:
            self.epsilon_decay = checkpoint['epsilon_decay']

        print(f"Agent loaded from {filepath}")
        print(f"  Training steps: {self.training_steps}")
        print(f"  Episodes trained: {self.episodes_trained}")
        print(f"  Epsilon: {self.epsilon:.4f}")

    def get_stats(self) -> dict:
        """Return training statistics."""
        return {
            'episodes': self.episodes_trained,
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'device': str(self.device)
        }


if __name__ == "__main__":
    # Quick test
    print("Testing DQN Agent...")

    agent = DQNAgent()
    print(f"Agent initialized with epsilon: {agent.epsilon}")
    print(f"Replay buffer size: {len(agent.replay_buffer)}")
    print(f"Device: {agent.device}")

    # Test state preprocessing
    print("\nTesting state preprocessing...")
    test_state = np.zeros((6, 7), dtype=np.int8)
    test_state[5, 3] = 1  # Player 1 piece at bottom center
    test_state[5, 4] = 2  # Player 2 piece next to it

    processed = agent._preprocess_state(test_state, current_player=1)
    print(f"Preprocessed state shape: {processed.shape}")

    # Test action selection
    print("\nTesting action selection...")
    legal_actions = [0, 1, 2, 3, 4, 5, 6]
    action = agent.choose_action(test_state, legal_actions, training=False)
    print(f"Chosen action (greedy): {action}")

    action = agent.choose_action(test_state, legal_actions, training=True)
    print(f"Chosen action (epsilon-greedy): {action}")

    # Test storing transitions and training
    print("\nTesting replay buffer and training...")
    for i in range(100):
        state = np.random.randint(0, 3, (6, 7), dtype=np.int8)
        action = random.randint(0, 6)
        reward = random.choice([-1.0, 0.0, 1.0])
        next_state = np.random.randint(0, 3, (6, 7), dtype=np.int8)
        done = random.random() < 0.1
        current_player = random.choice([1, 2])
        agent.store_transition(state, action, reward, next_state, done, current_player)

    print(f"Buffer size after adding transitions: {len(agent.replay_buffer)}")

    loss = agent.train_step()
    print(f"Training loss: {loss}")

    # Test save/load
    print("\nTesting save/load...")
    agent.save("test_agent.pth")

    agent2 = DQNAgent()
    agent2.load("test_agent.pth")

    print(f"\nLoaded agent stats: {agent2.get_stats()}")

    # Clean up test file
    import os
    if os.path.exists("test_agent.pth"):
        os.remove("test_agent.pth")
        print("Test file cleaned up.")

    print("\nAll tests passed!")
