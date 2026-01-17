# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A DQN (Deep Q-Network) agent that learns to play Connect Four through self-play. This builds on tabular Q-learning concepts from [tic-tac-toe-rl](https://github.com/bwarneck/tic-tac-toe-rl), introducing neural network function approximation for Connect Four's ~4.5 trillion state space.

## Commands

**Dependencies** (no requirements.txt - install manually):
```bash
pip install numpy torch jupyter
```

**Training** (50,000 episodes, saves to `trained_agent.pth`):
```bash
python src/train.py
```

**Play against trained agent**:
```bash
python src/play.py
```

**Evaluate against random opponent** (1000 games):
```bash
python src/evaluate.py
```

**Training visualization notebook**:
```bash
jupyter notebook notebooks/visualization.ipynb
```

**Run tests** (reward shaping unit tests):
```bash
python -m pytest tests/test_reward_shaping.py -v
```

## Architecture

### Neural Network (`ConnectFourDQN` in src/agent.py)
- **Input**: 3-channel 6x7 tensor (current player pieces, opponent pieces, empty cells)
- **Conv layers**: 3→64→128→128 filters with BatchNorm and ReLU
- **FC layers**: flatten→256→128→7 (Q-values for each column)

### DQN Components (`DQNAgent` in src/agent.py)
- **Policy network**: Makes action decisions
- **Target network**: Provides stable training targets (updated every 1000 steps)
- **Replay buffer**: 100K capacity, random sampling breaks temporal correlations
- **Epsilon-greedy**: ε decays from 1.0→0.01 (decay factor 0.9995)

### Game Environment (`ConnectFour` in src/game.py)
- 6x7 board, values: 0=empty, 1=Player1, 2=Player2
- Key methods: `reset()`, `make_move(column)`, `get_legal_actions()`, `get_state()`
- Win detection: horizontal, vertical, and both diagonals

### Training Loop (src/train.py)
- Self-play: agent plays both sides
- Terminal reward: +1.0 win, -1.0 loss, 0.0 draw (assigned post-game)
- Shaped rewards: intermediate rewards for threats and positioning (see below)
- Default: 50,000 episodes, batch size 64, learning rate 0.001, γ=0.99

### Reward Shaping (src/game.py)
The agent receives intermediate rewards to accelerate learning:

| Reward Type | Value | Description |
|-------------|-------|-------------|
| Create immediate threat | +0.08 | 3-in-row with playable completion cell |
| Block opponent threat | +0.06 | Prevents opponent's immediate win |
| Create future threat | +0.04 | 3-in-row with non-playable completion cell |
| Block future threat | +0.03 | Prevents opponent's future pressure |
| Create 2-in-row setup | +0.02 | 2 pieces with 2 empty in a window |
| Center column play | +0.01 | Strategic advantage |

Helper methods in `ConnectFour`:
- `is_playable(row, col)`: Check if empty cell can receive piece immediately
- `find_threats(player)`: Find all threat patterns (3-in-row, 2-in-row setups)
- `compute_shaped_reward(player, action, prev_board)`: Calculate shaped reward for move

## Key Hyperparameters

| Parameter | Default | Location |
|-----------|---------|----------|
| Episodes | 50,000 | train.py |
| Learning rate | 0.001 | train.py |
| Discount (γ) | 0.99 | train.py |
| Epsilon decay | 0.9995 | train.py |
| Batch size | 64 | train.py |
| Buffer capacity | 100,000 | agent.py |
| Target update freq | 1000 steps | agent.py |
| Gradient clip | 1.0 | agent.py |
