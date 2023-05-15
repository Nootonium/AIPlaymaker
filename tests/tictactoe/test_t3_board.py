import unittest
from src.tictactoe.t3_board import T3Board
from .test_constants import *


class TestT3Board(unittest.TestCase):
    def setUp(self):
        self.board = T3Board("         ")

    def test_given_nested_list_when_valid_then_detect_format(self):
        input_board = VALID_NESTED_LIST
        self.assertEqual(
            T3Board.detect_format(input_board), T3Board.Formats.NESTED_LIST
        )

    def test_given_nested_list_when_invalid_then_raise_error(self):
        input_board = INVALID_NESTED_LIST_1
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)

        input_board = INVALID_NESTED_LIST_2
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)

    def test_given_flat_list_when_valid_then_detect_format(self):
        input_board = VALID_FLAT_LIST
        self.assertEqual(T3Board.detect_format(input_board), T3Board.Formats.FLAT_LIST)

    def test_given_flat_list_when_invalid_then_raise_error(self):
        input_board = INVALID_FLAT_LIST_1
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)
        input_board = INVALID_FLAT_LIST_2
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)

    def test_given_string_when_valid_then_detect_format(self):
        input_board = VALID_STRING
        self.assertEqual(T3Board.detect_format(input_board), T3Board.Formats.STRING)

    def test_given_string_when_invalid_then_raise_error(self):
        input_board = INVALID_STRING_1
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)
        input_board = INVALID_STRING_2
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)

    def test_given_none_when_detect_format_then_raise_error(self):
        input_board = None
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)

    def test_given_not_string_or_list_when_detect_format_then_raise_error(self):
        input_board = 12345
        with self.assertRaises(ValueError):
            T3Board.detect_format(input_board)

    def test_given_nested_list_when_convert_to_internal_format_then_return_string(self):
        given_input_board = VALID_NESTED_LIST
        given_format = T3Board.Formats.NESTED_LIST
        expected_output = VALID_STRING

        when_result = T3Board.convert_to_internal_format(
            given_input_board, given_format
        )

        self.assertEqual(when_result, expected_output)

    def test_given_flat_list_when_convert_to_internal_format_then_return_string(self):
        given_input_board = VALID_FLAT_LIST
        given_format = T3Board.Formats.FLAT_LIST
        expected_output = VALID_STRING

        when_result = T3Board.convert_to_internal_format(
            given_input_board, given_format
        )

        self.assertEqual(when_result, expected_output)

    def test_given_string_when_convert_to_internal_format_then_return_same_string(self):
        given_input_board = VALID_STRING
        given_format = T3Board.Formats.STRING
        expected_output = VALID_STRING

        when_result = T3Board.convert_to_internal_format(
            given_input_board, given_format
        )

        self.assertEqual(when_result, expected_output)

    def test_given_invalid_format_when_convert_to_internal_format_then_return_none(
        self,
    ):
        given_input_board = VALID_STRING
        given_format = "INVALID_FORMAT"
        expected_output = None

        when_result = T3Board.convert_to_internal_format(
            given_input_board, given_format
        )

        self.assertEqual(when_result, expected_output)

    def test_given_valid_state_when_checking_validity_then_return_true(self):
        given_state = VALID_STRING
        board = T3Board(given_state)

        self.assertTrue(board.is_valid_game_state())
        self.assertTrue(T3Board.validate_state(given_state))

    def test_given_invalid_state_when_checking_validity_then_return_false(self):
        given_state = INVALID_STRING_STATE
        board = T3Board(given_state)

        self.assertFalse(board.is_valid_game_state())
        self.assertFalse(T3Board.validate_state(given_state))

    def test_given_x_wins_when_get_winner_then_return_x(self):
        given_state = "XXXOO    "
        board = T3Board(given_state)

        when_result = board.get_winner()

        self.assertEqual(when_result, "X")

    def test_given_o_wins_when_get_winner_then_return_o(self):
        given_state = "OOOXX    "
        board = T3Board(given_state)

        when_result = board.get_winner()

        self.assertEqual(when_result, "O")

    def test_given_not_finished_when_get_winner_then_return_space(self):
        given_state = VALID_STRING
        board = T3Board(given_state)

        when_result = board.get_winner()

        self.assertIsNone(when_result)

    def test_given_stalemate_when_get_winner_then_return_none(self):
        given_state = "XOXXOXOXO"
        board = T3Board(given_state)

        when_result = board.get_winner()
        self.assertEqual(when_result, " ")

    def test_given_full_board_when_get_next_possible_moves_then_return_empty_list(self):
        given_state = "XOXOXOXOX"  # Full board
        board = T3Board(given_state)

        when_result = board.get_next_possible_moves()

        self.assertEqual(when_result, [])

    def test_given_empty_board_when_get_next_possible_moves_then_return_all_positions(
        self,
    ):
        given_state = "         "  # Empty board
        board = T3Board(given_state)

        when_result = board.get_next_possible_moves()

        self.assertEqual(when_result, list(range(9)))

    def test_given_empty_board_when_get_next_player_then_return_x(self):
        given_state = "         "  # Empty board
        board = T3Board(given_state)

        when_result = board.get_next_player()

        self.assertEqual(when_result, "X")

    def test_given_board_with_x_when_get_next_player_then_return_o(self):
        given_state = "X        "  # Board with X
        board = T3Board(given_state)

        when_result = board.get_next_player()

        self.assertEqual(when_result, "O")

    def test_validate_board(self):
        pass  # Replace pass with your test code

    def test_get_format(self):
        pass  # Replace pass with your test code

    def test_convert_to_output_format(self):
        pass  # Replace pass with your test code

    def test_convert_from_internal_format(self):
        pass  # Replace pass with your test code

    def test_is_empty(self):
        pass  # Replace pass with your test code


if __name__ == "__main__":
    unittest.main()
