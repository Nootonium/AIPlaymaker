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

    def test_given_game_ongoing_when_get_winner_then_return_space(self):
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

    def test_given_nested_list_when_valid_then_validate_board(self):
        given = [["XO "], ["XO "], ["   "]]
        when = T3Board.validate_board(given)
        self.assertTrue(when)

    def test_given_nested_list_when_invalid_then_validate_board(self):
        given = [["XOP"], ["XO "], ["   "]]
        when = T3Board.validate_board(given)
        self.assertFalse(when)

    def test_given_flat_list_when_valid_then_validate_board(self):
        given = ["X", "O", " ", " ", "X", "O", "O", " ", "X"]
        when = T3Board.validate_board(given)
        self.assertTrue(when)

    def test_given_flat_list_when_invalid_then_validate_board(self):
        given = ["X", "O", "P", " ", "X", "O", "O", " ", "X"]
        when = T3Board.validate_board(given)
        self.assertFalse(when)

    def test_given_string_when_valid_then_validate_board(self):
        given = "XO XO O X"
        when = T3Board.validate_board(given)
        self.assertTrue(when)

    def test_given_string_when_invalid_then_validate_board(self):
        given = "XO XP O X"
        when = T3Board.validate_board(given)
        self.assertFalse(when)

    def test_given_nested_list_when_get_format_then_return_nested_list(self):
        given = T3Board([["XO "], ["XO "], ["   "]])
        when = given.get_format()
        self.assertEqual(when, T3Board.Formats.NESTED_LIST)

    def test_given_flat_list_when_get_format_then_return_flat_list(self):
        given = T3Board(["X", "O", " ", " ", "X", "O", "O", " ", "X"])
        when = given.get_format()
        self.assertEqual(when, T3Board.Formats.FLAT_LIST)

    def test_given_string_when_get_format_then_return_string(self):
        given = T3Board("XO XO O X")
        when = given.get_format()
        self.assertEqual(when, T3Board.Formats.STRING)

    def test_given_nested_list_when_converting_then_return_nested_list(self):
        given_state = "XO XO    "
        given_format = T3Board.Formats.NESTED_LIST
        expected_result = [["X", "O", " "], ["X", "O", " "], [" ", " ", " "]]

        when_result = T3Board.convert_from_internal_format(given_state, given_format)

        self.assertEqual(when_result, expected_result)

    def test_given_flat_list_when_converting_then_return_flat_list(self):
        given_state = "XO XO    "
        given_format = T3Board.Formats.FLAT_LIST
        expected_result = list(given_state)

        when_result = T3Board.convert_from_internal_format(given_state, given_format)

        self.assertEqual(when_result, expected_result)

    def test_given_string_when_converting_then_return_string(self):
        given_state = "XO XO    "
        given_format = T3Board.Formats.STRING
        expected_result = given_state

        when_result = T3Board.convert_from_internal_format(given_state, given_format)

        self.assertEqual(when_result, expected_result)

    def test_given_invalid_format_when_converting_then_raise_error(self):
        given_state = "XO XO    "
        given_format = "Invalid Format"

        with self.assertRaises(ValueError):
            T3Board.convert_from_internal_format(given_state, given_format)

    def test_given_empty_board_when_checking_is_empty_then_return_true(self):
        t3board = T3Board(" " * 9)

        when_result = t3board.is_empty()

        self.assertTrue(when_result)

    def test_given_non_empty_board_when_checking_is_empty_then_return_false(self):
        t3board = T3Board("XO XO    ")

        when_result = t3board.is_empty()

        self.assertFalse(when_result)


if __name__ == "__main__":
    unittest.main()
