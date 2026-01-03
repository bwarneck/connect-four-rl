"""
Evaluation Utilities for Connect Four Agent

Evaluate the trained agent's performance against various opponents.
"""

from typing import Dict
from game import ConnectFour, RandomAgent
from agent import DQNAgent


def evaluate_agent(
    agent: DQNAgent,
    opponent: str = 'random',
    num_games: int = 1000
) -> Dict:
    """
    Evaluate the agent against a specified opponent.

    Args:
        agent: The agent to evaluate
        opponent: 'random' for now (minimax too slow for Connect Four)
        num_games: Number of games to play

    Returns:
        Dictionary with detailed statistics
    """
    if opponent == 'random':
        opp_agent = RandomAgent()
    else:
        raise ValueError(f"Unknown opponent: {opponent}")

    results = {
        'as_player1': {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0},
        'as_player2': {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}
    }

    for game_num in range(num_games):
        game = ConnectFour()
        agent_is_p1 = (game_num % 2 == 0)
        key = 'as_player1' if agent_is_p1 else 'as_player2'

        while not game.done:
            state = game.get_state()
            legal_actions = game.get_legal_actions()

            if (game.current_player == ConnectFour.PLAYER_1) == agent_is_p1:
                action = agent.choose_action(state, legal_actions, training=False)
            else:
                action = opp_agent.choose_action(state, legal_actions)

            game.make_move(action)

        results[key]['games'] += 1
        if game.winner is None:
            results[key]['draws'] += 1
        elif (game.winner == ConnectFour.PLAYER_1) == agent_is_p1:
            results[key]['wins'] += 1
        else:
            results[key]['losses'] += 1

    # Compute totals
    total = {
        'wins': results['as_player1']['wins'] + results['as_player2']['wins'],
        'losses': results['as_player1']['losses'] + results['as_player2']['losses'],
        'draws': results['as_player1']['draws'] + results['as_player2']['draws'],
        'games': num_games
    }

    return {
        'as_player1': results['as_player1'],
        'as_player2': results['as_player2'],
        'total': total,
        'opponent': opponent
    }


def print_evaluation_report(agent: DQNAgent):
    """Print a comprehensive evaluation report."""
    print("\n" + "=" * 60)
    print("AGENT EVALUATION REPORT")
    print("=" * 60)

    # Agent stats
    stats = agent.get_stats()
    print(f"\nTraining Statistics:")
    print(f"  Episodes trained: {stats['episodes']}")
    print(f"  Training steps: {stats['training_steps']}")
    print(f"  Current epsilon: {stats['epsilon']:.4f}")
    print(f"  Replay buffer size: {stats['buffer_size']}")

    # Evaluate against random
    print(f"\nEvaluating against Random (1000 games)...")
    random_results = evaluate_agent(agent, 'random', 1000)
    _print_results(random_results)

    print("\n" + "=" * 60)


def _print_results(results: Dict):
    """Helper to print evaluation results."""
    for role in ['as_player1', 'as_player2']:
        r = results[role]
        total = r['games'] if r['games'] > 0 else 1
        role_name = "As Player 1 (Red)" if role == 'as_player1' else "As Player 2 (Yellow)"
        print(f"  {role_name}:")
        print(f"    Wins: {r['wins']/total:.1%} ({r['wins']}/{total})")
        print(f"    Losses: {r['losses']/total:.1%} ({r['losses']}/{total})")
        print(f"    Draws: {r['draws']/total:.1%} ({r['draws']}/{total})")

    t = results['total']
    total_games = t['games'] if t['games'] > 0 else 1
    print(f"  Overall:")
    print(f"    Win rate: {t['wins']/total_games:.1%}")
    print(f"    Loss rate: {t['losses']/total_games:.1%}")
    print(f"    Draw rate: {t['draws']/total_games:.1%}")


if __name__ == "__main__":
    import os

    agent_path = os.path.join(os.path.dirname(__file__), '..', 'trained_agent.pth')

    if os.path.exists(agent_path):
        print(f"Loading agent from: {agent_path}")
        agent = DQNAgent()
        agent.load(agent_path)
        print_evaluation_report(agent)
    else:
        print(f"No trained agent found at: {agent_path}")
        print("Run train.py first to train an agent.")
