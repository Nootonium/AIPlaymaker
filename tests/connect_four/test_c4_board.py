import pytest
from src.connect_four.c4_board import C4Board


@pytest.mark.parametrize("dimensions", [(value, value) for value in range(4, 20)])
def test_given_dimensions_when_creating_empty_board_then_attributes_match(
    dimensions,
):
    rows, cols = dimensions
    board = C4Board(dimensions, " " * (rows * cols))
    assert board.state == " " * (rows * cols)
    assert board.dimensions == dimensions


def test_next_player_when_board_is_empty():
    # Given an empty board
    board = C4Board((6, 7), " " * 42)
    # When I ask for the next player
    next_player = board.get_next_player()
    # Then it should be Player 1
    assert next_player == "1"


def test_next_player_when_last_move_by_player1():
    # Given a board where the last move was made by Player 1
    board = C4Board((6, 7), "1" + " " * 41)
    # When I ask for the next player
    next_player = board.get_next_player()
    # Then it should be Player 2
    assert next_player == "2"


def test_given_empty_board_when_get_next_possible_moves_then_return_all_columns():
    # Given an empty board
    board = C4Board((6, 7), " " * 42)
    # When we get the next possible moves
    moves = board.get_next_possible_moves()
    # Then it should return all columns
    assert moves == list(range(7))


def test_given_board_with_full_column_when_get_next_possible_moves_then_return_other_columns():
    # Given a board with a full column
    state = " " * 42
    for i in range(6):  # fill the first column
        state = state[: 7 * i] + "1" + state[7 * i + 1 :]
    print(state.replace(" ", "0"))
    board = C4Board((6, 7), state)
    # When we get the next possible moves
    moves = board.get_next_possible_moves()
    # Then it should return all columns except the full one
    assert moves == list(range(1, 7))


def test_given_board_with_all_columns_full_when_get_next_possible_moves_then_return_no_columns():
    # Given a board with all columns full
    state = "1" * 21 + "2" * 21
    board = C4Board((6, 7), state)
    # When we get the next possible moves
    moves = board.get_next_possible_moves()
    # Then it should return no columns
    assert moves == []


def test_winner():
    board = C4Board((6, 7), "1111" + " " * 38)
    assert board.get_winner() == "1"


if __name__ == "__main__":
    pytest.main()
