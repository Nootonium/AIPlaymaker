import pytest
from src.tictactoe.t3_converter import T3Converter
from src.tictactoe.t3_constants import BoardFormats
from .test_constants import VALID_BOARD, INVALID_BOARD


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

    assert T3Converter.detect_format(input_board) == BoardFormats.INVALID


def test_given_not_string_or_list_when_detect_format_then_raise_error():
    input_board = 12345

    assert T3Converter.detect_format(input_board) == BoardFormats.INVALID


def test_given_flat_list_when_convert_to_internal_format_then_return_string():
    given_flat_board = T3Converter.convert_from_internal_format(
        VALID_BOARD, BoardFormats.FLAT_LIST
    )

    when_result = T3Converter.convert_to_internal_format(
        given_flat_board, BoardFormats.FLAT_LIST
    )

    assert when_result == VALID_BOARD


def test_given_string_when_convert_to_internal_format_then_return_same_string():
    given_string_board = VALID_BOARD

    when_result = T3Converter.convert_to_internal_format(
        given_string_board, BoardFormats.STRING
    )

    assert when_result == VALID_BOARD


def test_given_invalid_format_when_convert_to_internal_format_then_return_none():
    with pytest.raises(ValueError):
        T3Converter.convert_to_internal_format(VALID_BOARD, "INVALID_FORMAT")


def test_given_string_when_convert_from_internal_format_to_flat_list_then_return_list():
    given_internal_board = VALID_BOARD

    when_result = T3Converter.convert_from_internal_format(
        given_internal_board, BoardFormats.FLAT_LIST
    )

    assert when_result == list(VALID_BOARD)


def test_given_string_when_convert_from_internal_format_to_string_then_return_string():
    given_internal_board = VALID_BOARD

    when_result = T3Converter.convert_from_internal_format(
        given_internal_board, BoardFormats.STRING
    )

    assert when_result == VALID_BOARD


def test_given_invalid_format_when_convert_from_internal_format_then_raise_error():
    with pytest.raises(ValueError):
        T3Converter.convert_from_internal_format(VALID_BOARD, "INVALID_FORMAT")


def test_given_flat_list_format_when_validating_board_then_returns_true():
    # Given
    board = list(VALID_BOARD)  # assuming VALID_MOVES is a list of 9 elements

    # When
    is_valid, board_format = T3Converter.validate_board(board)

    # Then
    assert is_valid is True
    assert board_format == BoardFormats.FLAT_LIST


def test_given_string_format_when_validating_board_then_returns_true():
    # Given
    board = "".join(VALID_BOARD)  # assuming VALID_MOVES is a list of 9 elements

    # When
    is_valid, board_format = T3Converter.validate_board(board)

    # Then
    assert is_valid is True
    assert board_format == BoardFormats.STRING


def test_given_invalid_format_when_validating_board_then_returns_false():
    # Given
    board = {INVALID_BOARD}

    # When
    is_valid, board_format = T3Converter.validate_board(board)

    # Then
    assert is_valid is False
    assert board_format == BoardFormats.INVALID


def test_given_valid_flat_list_when_validating_then_returns_true():
    # Given
    board = list(VALID_BOARD)

    # When
    result = T3Converter._validate_flat_list(board)

    # Then
    assert result is True


def test_given_invalid_flat_list_when_validating_then_returns_false():
    # Given
    board = list(INVALID_BOARD)

    # When
    result = T3Converter._validate_flat_list(board)

    # Then
    assert result is False
