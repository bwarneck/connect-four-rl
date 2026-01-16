"""
Watch a Game - Visualize agent playing move by move

Watch the trained DQN agent play a complete game against a random opponent,
with the board and Q-values displayed after each move.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from game import ConnectFour, RandomAgent
from agent import DQNAgent


def print_q_values(agent: DQNAgent, state: np.ndarray, current_player: int,
                   legal_actions: list, chosen_action: int):
    """Display Q-values as a simple bar chart in the terminal."""
    with torch.no_grad():
        state_tensor = agent._preprocess_state(state, current_player)
        q_values = agent.policy_net(state_tensor).cpu().numpy()[0]

    print("\n  Q-values by column:")
    print("  " + "-" * 43)

    # Find max Q-value for scaling
    max_q = max(q_values)
    min_q = min(q_values)
    q_range = max_q - min_q if max_q != min_q else 1

    for col in range(7):
        # Normalize to 0-20 character width
        bar_length = int(20 * (q_values[col] - min_q) / q_range) if q_range > 0 else 10
        bar = "#" * bar_length

        # Mark legal/illegal and chosen
        if col == chosen_action:
            marker = " <- CHOSEN"
        elif col in legal_actions:
            marker = ""
        else:
            marker = " (full)"

        print(f"  Col {col}: {q_values[col]:+.3f} |{bar}{marker}")

    print("  " + "-" * 43)


def watch_game(agent: DQNAgent, delay: float = 1.0, show_q_values: bool = True,
               agent_plays_both: bool = False):
    """
    Watch a complete game with move-by-move visualization.

    Args:
        agent: Trained DQN agent
        delay: Seconds to wait between moves
        show_q_values: Whether to display Q-value analysis
        agent_plays_both: If True, agent plays both sides; otherwise plays vs random
    """
    game = ConnectFour()
    random_agent = RandomAgent()
    move_num = 0

    # Randomly decide if agent plays first or second
    import random
    agent_is_p1 = random.choice([True, False])

    print("\n" + "=" * 50)
    print("        CONNECT FOUR - GAME VISUALIZATION")
    print("=" * 50)

    if agent_plays_both:
        print("\n  Mode: Agent vs Agent (self-play)")
    else:
        agent_role = "Player 1 (Red)" if agent_is_p1 else "Player 2 (Yellow)"
        print(f"\n  Mode: Agent vs Random")
        print(f"  Agent plays as: {agent_role}")

    print(f"  Q-value display: {'ON' if show_q_values else 'OFF'}")
    print(f"  Delay between moves: {delay}s")
    print("\n" + "-" * 50)

    print("\nInitial board:")
    print(game)
    time.sleep(delay)

    while not game.done:
        move_num += 1
        state = game.get_state()
        legal_actions = game.get_legal_actions()
        current_player = game.current_player
        player_name = "Red (P1)" if current_player == 1 else "Yellow (P2)"

        # Determine who makes this move
        if agent_plays_both:
            is_agent_turn = True
        else:
            is_agent_turn = (current_player == ConnectFour.PLAYER_1) == agent_is_p1

        print(f"\n{'=' * 50}")
        print(f"  Move {move_num}: {player_name}'s turn", end="")

        if is_agent_turn:
            print(" [AGENT]")
            action = agent.choose_action(state, legal_actions, training=False,
                                        current_player=current_player)
            if show_q_values:
                print_q_values(agent, state, current_player, legal_actions, action)
        else:
            print(" [RANDOM]")
            action = random_agent.choose_action(state, legal_actions)

        print(f"\n  -> Drops piece in column {action}")

        game.make_move(action)

        print(f"\n{game}")

        if not game.done:
            time.sleep(delay)

    # Game over
    print("\n" + "=" * 50)
    print("                  GAME OVER")
    print("=" * 50)

    if game.winner is None:
        print("\n  Result: DRAW")
    else:
        winner_name = "Red (P1)" if game.winner == 1 else "Yellow (P2)"

        if agent_plays_both:
            print(f"\n  Result: {winner_name} WINS!")
        else:
            agent_won = (game.winner == ConnectFour.PLAYER_1) == agent_is_p1
            if agent_won:
                print(f"\n  Result: AGENT ({winner_name}) WINS!")
            else:
                print(f"\n  Result: RANDOM ({winner_name}) WINS!")

    print(f"  Total moves: {game.move_count}")
    print("=" * 50 + "\n")

    return game.winner


def main():
    parser = argparse.ArgumentParser(
        description="Watch the trained agent play a game of Connect Four"
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=1.0,
        help="Delay in seconds between moves (default: 1.0)"
    )
    parser.add_argument(
        "--no-q-values", "-q", action="store_true",
        help="Hide Q-value display"
    )
    parser.add_argument(
        "--self-play", "-s", action="store_true",
        help="Watch agent play against itself instead of random"
    )
    parser.add_argument(
        "--num-games", "-n", type=int, default=1,
        help="Number of games to watch (default: 1)"
    )

    args = parser.parse_args()

    # Load trained agent
    agent_path = os.path.join(os.path.dirname(__file__), '..', 'trained_agent.pth')
    agent = DQNAgent()

    if os.path.exists(agent_path):
        print(f"Loading trained agent from: {agent_path}")
        agent.load(agent_path)
    else:
        print(f"WARNING: No trained agent found at {agent_path}")
        print("Using untrained agent (will play randomly)")

    # Watch games
    results = {'agent': 0, 'opponent': 0, 'draw': 0}

    for game_num in range(args.num_games):
        if args.num_games > 1:
            print(f"\n{'#' * 50}")
            print(f"  GAME {game_num + 1} of {args.num_games}")
            print(f"{'#' * 50}")

        winner = watch_game(
            agent,
            delay=args.delay,
            show_q_values=not args.no_q_values,
            agent_plays_both=args.self_play
        )

        if args.self_play:
            # In self-play, just track P1/P2 wins
            if winner == 1:
                results['agent'] += 1
            elif winner == 2:
                results['opponent'] += 1
            else:
                results['draw'] += 1

        if args.num_games > 1 and game_num < args.num_games - 1:
            input("Press Enter to continue to next game...")

    if args.num_games > 1:
        print("\n" + "=" * 50)
        print("               SESSION SUMMARY")
        print("=" * 50)
        print(f"  Games played: {args.num_games}")
        if args.self_play:
            print(f"  P1 (Red) wins: {results['agent']}")
            print(f"  P2 (Yellow) wins: {results['opponent']}")
        print(f"  Draws: {results['draw']}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
