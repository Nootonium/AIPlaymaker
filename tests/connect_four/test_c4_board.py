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


def test_given_state_length_not_matching_dimensions_when_initializing_then_raise_error():
    # Given a state where the length does not match the dimensions
    dimensions = (6, 7)
    state = " " * 40  # Not 42 (6 * 7)
    # When we try to initialize a C4Board
    # Then it should raise a ValueError
    with pytest.raises(ValueError, match="Invalid board"):
        C4Board(dimensions, state)


def test_given_state_with_invalid_token_when_initializing_then_raise_error():
    # Given a state with an invalid token
    dimensions = (6, 7)
    state = " " * 41 + "3"  # '3' is not a valid token
    # When we try to initialize a C4Board
    # Then it should raise a ValueError
    with pytest.raises(ValueError, match="Invalid board"):
        C4Board(dimensions, state)


def test_given_state_with_floating_piece_when_initializing_then_raise_error():
    # Given a state with a floating piece
    dimensions = (6, 7)
    state = (
        " " * 7 + "1" + " " * 6 + " " * 7 * 4  # Top row  # Second row  # Remaining rows
    )
    # When we try to initialize a C4Board
    # Then it should raise a ValueError
    with pytest.raises(ValueError, match="Invalid board"):
        C4Board(dimensions, state)


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


def test_given_valid_move_when_with_move_then_return_updated_board():
    # Given a valid move
    dimensions = (6, 7)
    state = " " * 42
    board = C4Board(dimensions, state)
    column = 0
    # When we make the move with the with_move method
    new_board = board.with_move(column)
    # Then it should return an updated board
    assert new_board.state == "1" + " " * 41


def test_given_out_of_range_move_when_with_move_then_raise_error():
    # Given an out of range move
    dimensions = (6, 7)
    state = " " * 42
    board = C4Board(dimensions, state)
    column = 7  # Out of range
    # When we try to make the move with the with_move method
    # Then it should raise a ValueError
    with pytest.raises(ValueError, match="Column 7 is out of range"):
        board.with_move(column)


def test_given_full_column_when_with_move_then_raise_error():
    # Given a full column
    dimensions = (6, 7)
    state = " " * 42
    for i in range(6):  # fill the first column
        state = state[: 7 * i] + "1" + state[7 * i + 1 :]
    board = C4Board(dimensions, state)
    column = 0  # Full column
    # When we try to make the move with the with_move method
    # Then it should raise a ValueError
    with pytest.raises(ValueError, match="Column 0 is full"):
        board.with_move(column)


def test_given_empty_board_when_is_empty_then_return_true():
    # Given an empty board
    board = C4Board((6, 7), " " * 42)

    # When we check if it's empty
    result = board.is_empty()

    # Then it should return True
    assert result is True


def test_given_non_empty_board_when_is_empty_then_return_false():
    # Given a non-empty board
    state = "1" + " " * 41
    board = C4Board((6, 7), state)

    # When we check if it's empty
    result = board.is_empty()

    # Then it should return False
    assert result is False


def test_given_two_differences_when_find_move_position_then_raise_value_error():
    # Given a board with an initial state
    initial_state = " " * 42
    board = C4Board((6, 7), initial_state)

    # When we try to find the move position with a new state that has two differences
    new_state = "12" + " " * 40  # two moves have been made
    with pytest.raises(ValueError) as exc_info:
        board.find_move_position(new_state)

    # Then it should raise a ValueError
    assert (
        str(exc_info.value)
        == "Invalid board state: more than one move was made in a single turn"
    )


def test_given_one_difference_when_find_move_position_then_return_diff_position():
    # Given a board with an initial state
    initial_state = " " * 42
    board = C4Board((6, 7), initial_state)

    # When we try to find the move position with a new state that has one difference
    new_state = "1" + " " * 41  # one move has been made
    row, col = board.find_move_position(new_state)

    # Then it should return the position of the move
    assert row == 0
    assert col == 0


def test_given_no_difference_when_find_move_position_then_raise_error():
    # Given a board with an initial state
    initial_state = " " * 42
    board = C4Board((6, 7), initial_state)

    # When we try to find the move position with the same state (no difference)
    # Then it should raise a ValueError
    with pytest.raises(ValueError) as e:
        board.find_move_position(initial_state)

    assert str(e.value) == "Invalid board state: no move was made."


def test_given_diagonal_win_when_get_winner_then_return_winner():
    # Given a board with a diagonal win
    state = "1112221" + " 1212  " + " 2121  " + " 1112  " + " " * 7 * 2
    board = C4Board((6, 7), state)

    # When we get the winner
    winner = board.get_winner()

    # Then it should return the player who made a diagonal win
    assert winner == "1"


def test_given_straight_win_when_get_winner_then_return_winner():
    # Given a board with a straight win
    state = "1" * 4 + "2" * 3 + " " * 7 * 5
    board = C4Board((6, 7), state)

    # When we get the winner
    winner = board.get_winner()

    # Then it should return the player who made a straight win
    assert winner == "1"


def test_given_no_win_when_get_winner_then_return_none():
    # Given a board with no win (maximum length of consecutive pieces is 3)
    state = "1" * 3 + "2" * 3 + " " * 36
    board = C4Board((6, 7), state)

    # When we get the winner
    winner = board.get_winner()

    # Then it should return None
    assert winner is None


def test_given_stalemate_when_get_winner_then_return_space():
    # Given a board with a stalemate (all spaces are filled but no one won)
    state = "1212121" + "1212121" + "1212121" + "2121212" + "2121212" + "2121212"
    board = C4Board((6, 7), state)

    # When we get the winner
    winner = board.get_winner()

    # Then it should return " " indicating a stalemate
    assert winner == " "


if __name__ == "__main__":
    pytest.main()
