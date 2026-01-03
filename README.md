# connect-four-rl

A reinforcement learning project exploring how an AI agent learns to play Connect Four, building on concepts from tabular Q-learning and introducing function approximation with neural networks.

## Overview

This project implements a Deep Q-Learning (DQN) agent that learns to play Connect Four through self-play. Unlike simpler games like tic-tac-toe where tabular methods suffice, Connect Four's massive state space (~4.5 trillion positions) requires neural networks to generalize across similar board states.

This is a continuation of the [tic-tac-toe-rl](https://github.com/bwarneck/tic-tac-toe-rl) learning workshop, stepping up in complexity to explore more advanced RL techniques.

## Why Connect Four?

Connect Four is an ideal next step after tic-tac-toe because it:
- Maintains familiar structure (deterministic, two-player, zero-sum, perfect information)
- Has a 7x6 board with ~4.5 trillion possible positions (vs ~5,500 for tic-tac-toe)
- Makes tabular Q-learning impractical, motivating function approximation
- Is solved (first player wins with perfect play), allowing verification of agent quality
- Introduces concepts like experience replay and target networks

## Learning Goals

- Implement Deep Q-Networks (DQN) for game playing
- Understand experience replay and its importance for training stability
- Learn about target networks to reduce oscillation
- Explore self-play training dynamics at scale
- Compare performance against known optimal strategies

## Project Status

🚧 Under construction

## License

MIT

## Author

Built as part of a public learning workshop in AI and reinforcement learning.
