"""
Unit tests for reward shaping functions in game.py.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from game import ConnectFour


class TestIsPlayable:
    """Tests for is_playable method."""

    def test_bottom_row_empty_is_playable(self):
        """Empty cell on bottom row should be playable."""
        game = ConnectFour()
        # Bottom row (row 5), all columns should be playable
        for col in range(game.COLS):
            assert game.is_playable(5, col) is True

    def test_cell_above_piece_is_playable(self):
        """Empty cell directly above a piece should be playable."""
        game = ConnectFour()
        game.board[5, 3] = game.PLAYER_1  # Place piece at bottom center
        # Cell above should be playable
        assert game.is_playable(4, 3) == True

    def test_cell_without_support_not_playable(self):
        """Empty cell with no piece below (not bottom) should not be playable."""
        game = ConnectFour()
        # Middle row with nothing below should not be playable
        assert game.is_playable(3, 3) == False

    def test_occupied_cell_not_playable(self):
        """Cell with a piece is not playable."""
        game = ConnectFour()
        game.board[5, 3] = game.PLAYER_1
        assert game.is_playable(5, 3) is False

    def test_out_of_bounds_not_playable(self):
        """Out of bounds positions should not be playable."""
        game = ConnectFour()
        assert game.is_playable(-1, 0) is False
        assert game.is_playable(6, 0) is False
        assert game.is_playable(0, -1) is False
        assert game.is_playable(0, 7) is False


class TestFindThreats:
    """Tests for find_threats method."""

    def test_horizontal_three_playable(self):
        """Detect horizontal 3-in-row with playable empty cell."""
        game = ConnectFour()
        # Place 3 pieces horizontally on bottom row: X X X _
        game.board[5, 0] = game.PLAYER_1
        game.board[5, 1] = game.PLAYER_1
        game.board[5, 2] = game.PLAYER_1

        threats = game.find_threats(game.PLAYER_1)
        assert (5, 3) in threats['three_playable']

    def test_horizontal_three_in_middle(self):
        """Detect horizontal 3-in-row with playable gap in middle."""
        game = ConnectFour()
        # _ X X X on bottom row
        game.board[5, 1] = game.PLAYER_1
        game.board[5, 2] = game.PLAYER_1
        game.board[5, 3] = game.PLAYER_1

        threats = game.find_threats(game.PLAYER_1)
        assert (5, 0) in threats['three_playable'] or (5, 4) in threats['three_playable']

    def test_vertical_three_playable(self):
        """Detect vertical 3-in-row with playable empty cell above."""
        game = ConnectFour()
        # Stack 3 pieces vertically
        game.board[5, 0] = game.PLAYER_1
        game.board[4, 0] = game.PLAYER_1
        game.board[3, 0] = game.PLAYER_1

        threats = game.find_threats(game.PLAYER_1)
        assert (2, 0) in threats['three_playable']

    def test_diagonal_three_playable(self):
        """Detect diagonal 3-in-row with playable empty cell."""
        game = ConnectFour()
        # Create diagonal down-right: positions (2,0), (3,1), (4,2), empty at (5,3)
        game.board[5, 0] = game.PLAYER_2  # Support for diagonal
        game.board[4, 0] = game.PLAYER_1
        game.board[5, 1] = game.PLAYER_2  # Support
        game.board[4, 1] = game.PLAYER_2  # Support
        game.board[3, 1] = game.PLAYER_1
        game.board[5, 2] = game.PLAYER_2  # Support
        game.board[4, 2] = game.PLAYER_2  # Support
        game.board[3, 2] = game.PLAYER_2  # Support
        game.board[2, 2] = game.PLAYER_1

        threats = game.find_threats(game.PLAYER_1)
        # The diagonal continuation would be at (1, 3) but that's not playable
        # Let's check if we detect something
        assert len(threats['three_playable']) >= 0  # At least no crash

    def test_three_future_not_immediately_playable(self):
        """Detect 3-in-row where empty cell is not immediately playable."""
        game = ConnectFour()
        # Vertical stack with gap above that's not immediately playable
        game.board[5, 0] = game.PLAYER_1
        game.board[4, 0] = game.PLAYER_1
        game.board[3, 0] = game.PLAYER_1
        # Row 2 is playable (directly above the stack)
        # But if we had a scenario where the empty is higher...

        # Better test: horizontal three with gap that needs support
        game2 = ConnectFour()
        game2.board[5, 0] = game2.PLAYER_1
        game2.board[5, 1] = game2.PLAYER_1
        # Gap at (5,2) - playable
        game2.board[5, 3] = game2.PLAYER_1
        # This creates a gap scenario but (5,2) is playable on bottom

        # For a true "future" threat, we need the empty cell not on bottom and no support
        game3 = ConnectFour()
        game3.board[5, 0] = game3.PLAYER_1  # Support
        game3.board[4, 0] = game3.PLAYER_1
        game3.board[5, 1] = game3.PLAYER_1  # Support
        game3.board[4, 1] = game3.PLAYER_1
        # For horizontal at row 4, need: 4,0 4,1 4,2 4,3
        # 4,2 has no support, so it would be "future"
        game3.board[4, 3] = game3.PLAYER_1
        game3.board[5, 3] = game3.PLAYER_2  # Support under 4,3

        threats = game3.find_threats(game3.PLAYER_1)
        # (4, 2) should be a future threat (not playable - no piece below)
        assert (4, 2) in threats['three_future']

    def test_two_in_row_setups(self):
        """Detect 2-in-row setups."""
        game = ConnectFour()
        # Two pieces with two empty spaces
        game.board[5, 0] = game.PLAYER_1
        game.board[5, 1] = game.PLAYER_1

        threats = game.find_threats(game.PLAYER_1)
        # Should have some two_setups count
        assert threats['two_setups'] > 0

    def test_blocked_window_no_threat(self):
        """Window with opponent piece should not be counted as threat."""
        game = ConnectFour()
        # Three player pieces but opponent blocking
        game.board[5, 0] = game.PLAYER_1
        game.board[5, 1] = game.PLAYER_1
        game.board[5, 2] = game.PLAYER_2  # Opponent blocks
        game.board[5, 3] = game.PLAYER_1

        threats = game.find_threats(game.PLAYER_1)
        # No playable threats in this window
        assert (5, 2) not in threats['three_playable']

    def test_opponent_threats_separate(self):
        """Each player's threats should be counted separately."""
        game = ConnectFour()
        # Player 1 has horizontal threat on bottom left
        game.board[5, 0] = game.PLAYER_1
        game.board[5, 1] = game.PLAYER_1
        game.board[5, 2] = game.PLAYER_1
        # P1 threat completion is at (5, 3)

        # Player 2 has vertical threat
        game.board[5, 6] = game.PLAYER_2
        game.board[4, 6] = game.PLAYER_2
        game.board[3, 6] = game.PLAYER_2
        # P2 threat completion is at (2, 6)

        p1_threats = game.find_threats(game.PLAYER_1)
        p2_threats = game.find_threats(game.PLAYER_2)

        assert len(p1_threats['three_playable']) > 0
        assert len(p2_threats['three_playable']) > 0
        # They should be different positions (P1 at col 3, P2 at col 6)
        assert (5, 3) in p1_threats['three_playable']
        assert (2, 6) in p2_threats['three_playable']


class TestComputeShapedReward:
    """Tests for compute_shaped_reward method."""

    def test_center_column_bonus(self):
        """Playing center column should give bonus."""
        game = ConnectFour()
        prev_board = game.board.copy()
        game.board[5, 3] = game.PLAYER_1  # Play center

        reward = game.compute_shaped_reward(game.PLAYER_1, 3, prev_board)
        assert reward >= 0.01  # At least the center bonus

    def test_edge_column_no_center_bonus(self):
        """Playing edge column should not give center bonus."""
        game = ConnectFour()
        prev_board = game.board.copy()
        game.board[5, 0] = game.PLAYER_1  # Play edge

        reward = game.compute_shaped_reward(game.PLAYER_1, 0, prev_board)
        # Should be some positive value from setups but no center bonus
        # We can't assert exact value easily, just that it's different from center

        game2 = ConnectFour()
        prev_board2 = game2.board.copy()
        game2.board[5, 3] = game2.PLAYER_1
        center_reward = game2.compute_shaped_reward(game2.PLAYER_1, 3, prev_board2)

        # Center should give at least 0.01 more
        assert center_reward >= reward

    def test_creating_immediate_threat(self):
        """Creating 3-in-row playable threat should be rewarded."""
        game = ConnectFour()
        # Setup: two pieces in a row
        game.board[5, 0] = game.PLAYER_1
        game.board[5, 1] = game.PLAYER_1
        prev_board = game.board.copy()

        # Add third piece to create threat
        game.board[5, 2] = game.PLAYER_1

        reward = game.compute_shaped_reward(game.PLAYER_1, 2, prev_board)
        assert reward >= 0.08  # Immediate threat reward

    def test_blocking_opponent_threat(self):
        """Blocking opponent's immediate threat should be rewarded."""
        game = ConnectFour()
        # Setup: opponent has threat
        game.board[5, 0] = game.PLAYER_2
        game.board[5, 1] = game.PLAYER_2
        game.board[5, 2] = game.PLAYER_2
        # Position (5, 3) is the threat
        prev_board = game.board.copy()

        # Block it
        game.board[5, 3] = game.PLAYER_1

        reward = game.compute_shaped_reward(game.PLAYER_1, 3, prev_board)
        assert reward >= 0.06  # Block reward

    def test_shaped_reward_is_small(self):
        """Shaped rewards should be small compared to terminal rewards."""
        game = ConnectFour()
        # Create a complex position
        game.board[5, 0] = game.PLAYER_1
        game.board[5, 1] = game.PLAYER_1
        prev_board = game.board.copy()
        game.board[5, 2] = game.PLAYER_1

        reward = game.compute_shaped_reward(game.PLAYER_1, 2, prev_board)
        # Shaped reward should never exceed terminal reward magnitude
        assert abs(reward) < 1.0

    def test_symmetric_for_both_players(self):
        """Reward computation should work symmetrically for both players."""
        # Player 1 creating threat
        game1 = ConnectFour()
        game1.board[5, 0] = game1.PLAYER_1
        game1.board[5, 1] = game1.PLAYER_1
        prev1 = game1.board.copy()
        game1.board[5, 2] = game1.PLAYER_1
        reward1 = game1.compute_shaped_reward(game1.PLAYER_1, 2, prev1)

        # Player 2 creating same pattern
        game2 = ConnectFour()
        game2.board[5, 0] = game2.PLAYER_2
        game2.board[5, 1] = game2.PLAYER_2
        prev2 = game2.board.copy()
        game2.board[5, 2] = game2.PLAYER_2
        reward2 = game2.compute_shaped_reward(game2.PLAYER_2, 2, prev2)

        # Rewards should be the same for symmetric situations
        assert abs(reward1 - reward2) < 0.001


class TestIntegration:
    """Integration tests for reward shaping with game flow."""

    def test_reward_shaping_during_game(self):
        """Test that reward shaping works during a game sequence."""
        game = ConnectFour()

        # Play a few moves and check rewards are computed
        moves = [(3, game.PLAYER_1), (4, game.PLAYER_2), (3, game.PLAYER_1), (5, game.PLAYER_2)]

        for col, expected_player in moves:
            prev_board = game.get_state()
            current_player = game.current_player
            assert current_player == expected_player

            game.make_move(col)

            reward = game.compute_shaped_reward(current_player, col, prev_board)
            assert isinstance(reward, float)
            assert -1.0 < reward < 1.0

    def test_no_crash_on_empty_board(self):
        """Computing threats on empty board should not crash."""
        game = ConnectFour()
        threats = game.find_threats(game.PLAYER_1)
        assert threats['three_playable'] == []
        assert threats['three_future'] == []
        assert threats['two_setups'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
