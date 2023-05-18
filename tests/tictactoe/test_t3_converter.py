import pytest
from src.tictactoe.t3_converter import T3Converter
from src.tictactoe.t3_constants import BoardFormats
from .test_constants import VALID_BOARD


def test_given_flat_list_when_valid_then_detect_format():
    input_board = T3Converter.convert_from_internal_format(
        VALID_BOARD, BoardFormats.FLAT_LIST
    )
    print(T3Converter.detect_format(input_board))

    assert T3Converter.detect_format(input_board) == BoardFormats.FLAT_LIST


def test_given_string_when_valid_then_detect_format():
    input_board = VALID_BOARD

    assert T3Converter.detect_format(input_board) == BoardFormats.STRING


def test_given_none_when_detect_format_then_raise_error():
    input_board = None

    with pytest.raises(ValueError):
        T3Converter.detect_format(input_board)


def test_given_not_string_or_list_when_detect_format_then_raise_error():
    input_board = 12345

    with pytest.raises(ValueError):
        T3Converter.detect_format(input_board)


"""
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

def test_given_flat_list_when_convert_to_internal_format_then_return_string(self):
    given_input_board = VALID_FLAT_LIST
    given_format = T3Board.Formats.FLAT_LIST
    expected_output = VALID_STRING

    when_result = T3Board.convert_to_internal_format(
        given_input_board, given_format
    )

    self.assertEqual(when_result, expected_output)

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
        """
