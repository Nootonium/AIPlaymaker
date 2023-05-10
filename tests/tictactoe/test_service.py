from src.tictactoe.service import Service


def test_get_next_move():
    service = Service()
    input_board = "         "  # assuming a 3x3 empty board represented as a string
    next_move = service.get_next_move(input_board)

    assert next_move["move"] in range(9)


def test_get_next_moves():
    service = Service()
    input_board = "         "  # assuming a 3x3 empty board represented as a string
    next_moves = service.get_next_moves(input_board)
    print(next_moves)
