import unittest
from src.tictactoe.t3_board import T3Board
from .test_constants import (
    EMPTY_BOARD,
    VALID_BOARD,
    VALID_BOARD_M8,
    VALID_X_WINS,
    VALID_O_WINS,
    INVALID_BOARD_STATE,
    STALEMATE_BOARD,
)


class TestT3Board(unittest.TestCase):
    def test_given_valid_state_when_checking_validity_then_return_true(self):
        board = T3Board(VALID_BOARD)

        self.assertTrue(board.is_valid_game_state())
        self.assertTrue(T3Board.validate_state(board.state))

    def test_given_invalid_state_when_checking_validity_then_return_false(self):
        board = INVALID_BOARD_STATE

        self.assertFalse(T3Board.validate_state(board))

    def test_given_x_wins_when_get_winner_then_return_x(self):
        board = T3Board(VALID_X_WINS)

        when_result = board.get_winner()

        self.assertEqual(when_result, "X")

    def test_given_o_wins_when_get_winner_then_return_o(self):
        board = T3Board(VALID_O_WINS)

        when_result = board.get_winner()

        self.assertEqual(when_result, "O")

    def test_given_game_ongoing_when_get_winner_then_return_none(self):
        board = T3Board(VALID_BOARD)

        when_result = board.get_winner()

        self.assertIsNone(when_result)

    def test_given_stalemate_when_get_winner_then_return_space(self):
        board = T3Board(STALEMATE_BOARD)

        when_result = board.get_winner()

        self.assertEqual(when_result, " ")

    def test_given_full_board_when_get_next_possible_moves_then_return_empty_list(self):
        board = T3Board(STALEMATE_BOARD)

        when_result = board.get_next_possible_moves()

        self.assertEqual(when_result, [])

    def test_given_empty_board_when_get_next_possible_moves_then_return_all_positions(
        self,
    ):
        board = T3Board(EMPTY_BOARD)

        when_result = board.get_next_possible_moves()

        self.assertEqual(when_result, list(range(9)))

    def test_given_empty_board_when_get_next_player_then_return_x(self):
        board = T3Board(EMPTY_BOARD)

        when_result = board.get_next_player()

        self.assertEqual(when_result, "X")

    def test_given_board_when_get_next_player_then_return_o(self):
        board = T3Board(VALID_BOARD)

        when_result = board.get_next_player()

        self.assertEqual(when_result, "O")

    def test_given_empty_board_when_checking_is_empty_then_return_true(self):
        t3board = T3Board(EMPTY_BOARD)

        when_result = t3board.is_empty()

        self.assertTrue(when_result)

    def test_given_non_empty_board_when_checking_is_empty_then_return_false(self):
        t3board = T3Board(VALID_BOARD)

        when_result = t3board.is_empty()

        self.assertFalse(when_result)

    def test_given_more_than_one_diff_when_compare_board_states_then_raises_value_error(
        self,
    ):
        old_board = EMPTY_BOARD
        new_board = VALID_BOARD

        with self.assertRaises(ValueError):
            T3Board.compare_board_states(old_board, new_board)

    def test_given_one_diff_move_when_compare_board_states_then_returns_index(self):
        old_board = VALID_BOARD
        new_board = VALID_BOARD_M8

        index = T3Board.compare_board_states(old_board, new_board)

        self.assertEqual(index, 8)

    def test_given_no_diff_moves_when_find_difference_position_then_returns_none(self):
        old_board = EMPTY_BOARD
        new_board = EMPTY_BOARD

        index = T3Board.compare_board_states(old_board, new_board)

        self.assertIsNone(index)


if __name__ == "__main__":
    unittest.main()
