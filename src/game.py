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

    def is_playable(self, row: int, col: int) -> bool:
        """
        Check if an empty cell can receive a piece immediately.

        A cell is playable if it's empty and either:
        - It's on the bottom row, or
        - There's a piece directly below it
        """
        if row < 0 or row >= self.ROWS or col < 0 or col >= self.COLS:
            return False
        if self.board[row, col] != self.EMPTY:
            return False
        if row == self.ROWS - 1:  # Bottom row
            return True
        return self.board[row + 1, col] != self.EMPTY  # Has piece below

    def _get_window_contents(self, start_row: int, start_col: int, dr: int, dc: int) -> List[Tuple[int, int, int]]:
        """
        Get contents of a 4-cell window starting at (start_row, start_col) in direction (dr, dc).

        Returns:
            List of (row, col, cell_value) tuples for the 4 cells, or empty list if window is invalid.
        """
        contents = []
        for i in range(4):
            r = start_row + i * dr
            c = start_col + i * dc
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                contents.append((r, c, self.board[r, c]))
            else:
                return []  # Window extends outside board
        return contents

    def find_threats(self, player: int) -> dict:
        """
        Find all threat patterns for a player.

        Returns:
            Dictionary with:
            - 'three_playable': List of (row, col) positions that would complete an immediate win
            - 'three_future': List of (row, col) positions for 3-in-row with non-playable empty
            - 'two_setups': Count of 2-in-row setups with 2 empty spaces
        """
        threats = {
            'three_playable': [],
            'three_future': [],
            'two_setups': 0
        }

        opponent = self.PLAYER_2 if player == self.PLAYER_1 else self.PLAYER_1

        # Directions: horizontal, vertical, diagonal down-right, diagonal down-left
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        seen_three_playable = set()
        seen_three_future = set()

        for row in range(self.ROWS):
            for col in range(self.COLS):
                for dr, dc in directions:
                    window = self._get_window_contents(row, col, dr, dc)
                    if not window:
                        continue

                    # Count pieces in window
                    player_count = sum(1 for _, _, val in window if val == player)
                    opponent_count = sum(1 for _, _, val in window if val == opponent)
                    empty_cells = [(r, c) for r, c, val in window if val == self.EMPTY]

                    # Skip if opponent has pieces (no threat possible)
                    if opponent_count > 0:
                        continue

                    # 3-in-a-row with 1 empty
                    if player_count == 3 and len(empty_cells) == 1:
                        empty_r, empty_c = empty_cells[0]
                        if self.is_playable(empty_r, empty_c):
                            if (empty_r, empty_c) not in seen_three_playable:
                                threats['three_playable'].append((empty_r, empty_c))
                                seen_three_playable.add((empty_r, empty_c))
                        else:
                            if (empty_r, empty_c) not in seen_three_future:
                                threats['three_future'].append((empty_r, empty_c))
                                seen_three_future.add((empty_r, empty_c))

                    # 2-in-a-row with 2 empty
                    elif player_count == 2 and len(empty_cells) == 2:
                        threats['two_setups'] += 1

        return threats

    def compute_shaped_reward(self, player: int, action: int, prev_board: np.ndarray) -> float:
        """
        Compute shaped reward for a move based on threats and positioning.

        Args:
            player: The player who made the move (PLAYER_1 or PLAYER_2)
            action: The column where the piece was placed
            prev_board: Board state before the move was made

        Returns:
            Shaped reward (small value to supplement terminal reward)
        """
        shaped_reward = 0.0
        opponent = self.PLAYER_2 if player == self.PLAYER_1 else self.PLAYER_1

        # Create temporary game to analyze previous state
        prev_game = ConnectFour()
        prev_game.board = prev_board.copy()

        # Get threats before and after the move
        prev_player_threats = prev_game.find_threats(player)
        prev_opponent_threats = prev_game.find_threats(opponent)

        current_player_threats = self.find_threats(player)
        current_opponent_threats = self.find_threats(opponent)

        # Reward for creating new immediate threats (3-in-row playable)
        new_immediate_threats = len(current_player_threats['three_playable']) - len(prev_player_threats['three_playable'])
        if new_immediate_threats > 0:
            shaped_reward += 0.08 * new_immediate_threats

        # Reward for blocking opponent immediate threats
        blocked_immediate = len(prev_opponent_threats['three_playable']) - len(current_opponent_threats['three_playable'])
        if blocked_immediate > 0:
            shaped_reward += 0.06 * blocked_immediate

        # Reward for creating future threats (3-in-row not immediately playable)
        new_future_threats = len(current_player_threats['three_future']) - len(prev_player_threats['three_future'])
        if new_future_threats > 0:
            shaped_reward += 0.04 * new_future_threats

        # Reward for blocking opponent future threats
        blocked_future = len(prev_opponent_threats['three_future']) - len(current_opponent_threats['three_future'])
        if blocked_future > 0:
            shaped_reward += 0.03 * blocked_future

        # Reward for creating 2-in-row setups
        new_setups = current_player_threats['two_setups'] - prev_player_threats['two_setups']
        if new_setups > 0:
            shaped_reward += 0.02 * new_setups

        # Center column bonus
        if action == 3:  # Center column
            shaped_reward += 0.01

        return shaped_reward

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
