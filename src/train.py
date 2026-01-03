"""
Training Script for Connect Four DQN Agent

Trains the agent through self-play, where it plays against itself
to learn optimal strategies.
"""

from typing import Dict, List
from game import ConnectFour
from agent import DQNAgent


def train_self_play(
    agent: DQNAgent,
    num_episodes: int = 50000,
    print_every: int = 1000
) -> Dict[str, List]:
    """
    Train agent through self-play.

    The agent plays both sides, learning from each game's outcome.

    Args:
        agent: DQN agent to train
        num_episodes: Number of games to play
        print_every: Print statistics every N episodes

    Returns:
        Dictionary containing training statistics
    """
    stats = {
        'episode': [],
        'player1_wins': [],
        'player2_wins': [],
        'draws': [],
        'epsilon': [],
        'avg_game_length': []
    }

    # Rolling statistics
    p1_wins = 0
    p2_wins = 0
    draws = 0
    total_moves = 0

    for episode in range(1, num_episodes + 1):
        game = ConnectFour()
        episode_transitions = []

        while not game.done:
            state = game.get_state()
            legal_actions = game.get_legal_actions()
            current_player = game.current_player

            # Choose action
            action = agent.choose_action(state, legal_actions, training=True)

            # Make move
            next_state, reward, done = game.make_move(action)

            # Store transition (we'll update rewards at end of game)
            episode_transitions.append({
                'state': state,
                'action': action,
                'player': current_player,
                'next_state': next_state,
                'done': done
            })

        # Game finished - assign rewards
        total_moves += game.move_count

        for i, transition in enumerate(episode_transitions):
            player = transition['player']

            if game.winner == player:
                reward = 1.0
                if player == ConnectFour.PLAYER_1:
                    p1_wins += 1
                else:
                    p2_wins += 1
            elif game.winner is None:
                reward = 0.0
                if i == len(episode_transitions) - 1:
                    draws += 1
            else:
                reward = -1.0

            # Store in replay buffer
            agent.store_transition(
                transition['state'],
                transition['action'],
                reward,
                transition['next_state'],
                transition['done']
            )

            # Train
            agent.train_step()

        agent.episodes_trained += 1

        # Print progress
        if episode % print_every == 0:
            total_games = p1_wins + p2_wins + draws
            if total_games > 0:
                p1_rate = p1_wins / total_games * 100
                p2_rate = p2_wins / total_games * 100
                draw_rate = draws / total_games * 100
                avg_length = total_moves / total_games
            else:
                p1_rate = p2_rate = draw_rate = avg_length = 0

            print(f"Episode {episode:6d} | "
                  f"P1 wins: {p1_rate:5.1f}% | "
                  f"P2 wins: {p2_rate:5.1f}% | "
                  f"Draws: {draw_rate:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg moves: {avg_length:.1f}")

            # Store stats
            stats['episode'].append(episode)
            stats['player1_wins'].append(p1_rate)
            stats['player2_wins'].append(p2_rate)
            stats['draws'].append(draw_rate)
            stats['epsilon'].append(agent.epsilon)
            stats['avg_game_length'].append(avg_length)

            # Reset rolling stats
            p1_wins = p2_wins = draws = total_moves = 0

    return stats


def main():
    """Main training function."""
    print("=" * 60)
    print("Connect Four DQN Training")
    print("=" * 60)

    # Create agent
    agent = DQNAgent(
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_size=100000,
        batch_size=64
    )

    # Train
    print("\nStarting training...")
    stats = train_self_play(agent, num_episodes=50000, print_every=1000)

    # Save agent
    print("\nSaving agent...")
    agent.save("trained_agent.pth")

    print("\nTraining complete!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total training steps: {agent.training_steps}")


if __name__ == "__main__":
    main()
