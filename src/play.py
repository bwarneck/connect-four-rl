"""
Human vs Agent Interactive Play

Play Connect Four against the trained DQN agent in the terminal.
"""

import os
import sys
from game import ConnectFour
from agent import DQNAgent


def get_human_move(game: ConnectFour) -> int:
    """Get a valid move from the human player."""
    legal_actions = game.get_legal_actions()

    while True:
        try:
            print(f"\nYour turn! Legal columns: {legal_actions}")
            move = input("Enter column (0-6) or 'q' to quit: ").strip()

            if move.lower() == 'q':
                print("Thanks for playing!")
                sys.exit(0)

            column = int(move)

            if column not in legal_actions:
                print(f"Invalid move. Choose from: {legal_actions}")
                continue

            return column

        except ValueError:
            print("Please enter a number 0-6 (or 'q' to quit)")


def play_game(agent: DQNAgent, human_first: bool = True):
    """
    Play a single game against the agent.

    Args:
        agent: Trained DQN agent
        human_first: If True, human plays first (Red)
    """
    game = ConnectFour()

    human_player = ConnectFour.PLAYER_1 if human_first else ConnectFour.PLAYER_2
    human_symbol = "Red (R)" if human_first else "Yellow (Y)"
    agent_symbol = "Yellow (Y)" if human_first else "Red (R)"

    print("\n" + "=" * 40)
    print(f"NEW GAME - You are {human_symbol}")
    print("=" * 40)
    print("\nDrop pieces by entering column number (0-6)")
    print("Connect 4 horizontally, vertically, or diagonally to win!\n")

    while not game.done:
        print("\nCurrent board:")
        print(game)

        if game.current_player == human_player:
            # Human's turn
            action = get_human_move(game)
        else:
            # Agent's turn
            state = game.get_state()
            legal_actions = game.get_legal_actions()
            action = agent.choose_action(state, legal_actions, training=False,
                                         current_player=game.current_player)
            print(f"\nAgent plays column: {action}")

        game.make_move(action)

    # Game over
    print("\n" + "=" * 40)
    print("GAME OVER")
    print("=" * 40)
    print("\nFinal board:")
    print(game)

    if game.winner is None:
        print("\nIt's a draw!")
        return 'draw'
    elif game.winner == human_player:
        print("\nYou win! Congratulations!")
        return 'human'
    else:
        print("\nAgent wins!")
        return 'agent'


def main():
    """Main interactive play loop."""
    print("\n" + "=" * 50)
    print("    CONNECT FOUR: Human vs DQN Agent")
    print("=" * 50)

    # Load trained agent
    agent_path = os.path.join(os.path.dirname(__file__), '..', 'trained_agent.pth')

    agent = DQNAgent()

    if os.path.exists(agent_path):
        print(f"\nLoading trained agent from: {agent_path}")
        agent.load(agent_path)
        stats = agent.get_stats()
        print(f"Agent trained for {stats['episodes']} episodes")
    else:
        print(f"\nNo trained agent found at: {agent_path}")
        print("Using untrained agent (will play randomly)")

    # Game settings
    print("\n" + "-" * 50)
    human_first = input("Do you want to play first? (y/n, default=y): ").strip().lower() != 'n'

    # Score tracking
    scores = {'human': 0, 'agent': 0, 'draw': 0}

    # Play loop
    while True:
        result = play_game(agent, human_first=human_first)
        scores[result] += 1

        print(f"\n--- Score: You {scores['human']} - Agent {scores['agent']} - Draws {scores['draw']} ---")

        again = input("\nPlay again? (y/n, default=y): ").strip().lower()
        if again == 'n':
            break

        switch = input("Switch who goes first? (y/n, default=n): ").strip().lower()
        if switch == 'y':
            human_first = not human_first

    print("\n" + "=" * 50)
    print("Thanks for playing!")
    print(f"Final score: You {scores['human']} - Agent {scores['agent']} - Draws {scores['draw']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
