"""
Connect Four Game Environment

This module provides the core game logic for Connect Four:
- 7 columns x 6 rows board
- Two players (1 = Red, 2 = Yellow)
- Win by connecting 4 pieces horizontally, vertically, or diagonally
"""

from typing import Optional, List, Tuple
import numpy as np


class ConnectFour:
    """Connect Four game environment."""

    ROWS = 6
    COLS = 7
    EMPTY = 0
    PLAYER_1 = 1  # Red (first player)
    PLAYER_2 = 2  # Yellow (second player)

    def __init__(self):
        """Initialize a new game."""
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.PLAYER_1
        self.done = False
        self.winner = None
        self.move_count = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Return a copy of the current board state."""
        return self.board.copy()

    def get_legal_actions(self) -> List[int]:
        """Return list of valid column indices (columns that aren't full)."""
        return [col for col in range(self.COLS) if self.board[0, col] == self.EMPTY]

    def make_move(self, column: int) -> Tuple[np.ndarray, float, bool]:
        """
        Drop a piece in the specified column.

        Args:
            column: Column index (0-6)

        Returns:
            Tuple of (new_state, reward, done)
        """
        if self.done:
            raise ValueError("Game is already over")

        if column not in self.get_legal_actions():
            raise ValueError(f"Column {column} is not a legal move")

        # Find the lowest empty row in this column
        row = self._get_lowest_empty_row(column)
        self.board[row, column] = self.current_player
        self.move_count += 1

        # Check for win
        if self._check_win(row, column):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        # Check for draw (board full)
        elif self.move_count == self.ROWS * self.COLS:
            self.done = True
            self.winner = None
            reward = 0.0
        else:
            reward = 0.0
            # Switch players
            self.current_player = self.PLAYER_2 if self.current_player == self.PLAYER_1 else self.PLAYER_1

        return self.get_state(), reward, self.done

    def _get_lowest_empty_row(self, column: int) -> int:
        """Find the lowest empty row in a column."""
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row, column] == self.EMPTY:
                return row
        raise ValueError(f"Column {column} is full")

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move at (row, col) resulted in a win."""
        player = self.board[row, col]

        # Directions: horizontal, vertical, diagonal down-right, diagonal down-left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # Check positive direction
            count += self._count_direction(row, col, dr, dc, player)
            # Check negative direction
            count += self._count_direction(row, col, -dr, -dc, player)

            if count >= 4:
                return True

        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        """Count consecutive pieces in a direction."""
        count = 0
        r, c = row + dr, col + dc

        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc

        return count

    def clone(self) -> 'ConnectFour':
        """Create a deep copy of the game."""
        new_game = ConnectFour()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.done = self.done
        new_game.winner = self.winner
        new_game.move_count = self.move_count
        return new_game

    def render(self) -> str:
        """Return a string representation of the board."""
        symbols = {self.EMPTY: '.', self.PLAYER_1: 'R', self.PLAYER_2: 'Y'}
        lines = []

        for row in range(self.ROWS):
            line = ' '.join(symbols[self.board[row, col]] for col in range(self.COLS))
            lines.append(line)

        # Add column numbers
        lines.append(' '.join(str(i) for i in range(self.COLS)))

        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.render()


class RandomAgent:
    """Agent that plays random legal moves."""

    def choose_action(self, state: np.ndarray, legal_actions: List[int]) -> int:
        """Choose a random legal action."""
        import random
        return random.choice(legal_actions)


if __name__ == "__main__":
    # Quick test
    game = ConnectFour()
    print("Initial board:")
    print(game)
    print(f"\nLegal actions: {game.get_legal_actions()}")

    # Play a few random moves
    agent = RandomAgent()
    for _ in range(10):
        if game.done:
            break
        action = agent.choose_action(game.get_state(), game.get_legal_actions())
        game.make_move(action)
        print(f"\nPlayer {3 - game.current_player} played column {action}:")
        print(game)

    if game.winner:
        print(f"\nPlayer {game.winner} wins!")
    elif game.done:
        print("\nDraw!")
