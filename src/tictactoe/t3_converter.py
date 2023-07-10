from typing import Tuple
from .t3_constants import VALID_MOVES, BoardFormats

# import torch


class T3Converter:
    @staticmethod
    def detect_format(input_board: str | list) -> BoardFormats:
        if isinstance(input_board, list) and len(input_board) == 9:
            return BoardFormats.FLAT_LIST

        if isinstance(input_board, str) and len(input_board) == 9:
            return BoardFormats.STRING

        return BoardFormats.INVALID

    @staticmethod
    def convert_to_internal_format(
        input_board: str | list[str], board_format: BoardFormats
    ) -> str:
        match board_format:
            case BoardFormats.FLAT_LIST:
                return "".join(input_board)
            case BoardFormats.STRING:
                return str(input_board)
            case _:
                raise ValueError("Invalid board format")

    @staticmethod
    def convert_from_internal_format(
        state: str, board_format: BoardFormats
    ) -> str | list[str]:
        match board_format:
            case BoardFormats.FLAT_LIST:
                return list(state)
            case BoardFormats.STRING:
                return state
            case _:
                raise ValueError("Invalid board format")

    @staticmethod
    def validate_board(input_board) -> Tuple[bool, BoardFormats]:
        board_format = T3Converter.detect_format(input_board)

        match board_format:
            case BoardFormats.FLAT_LIST:
                return (T3Converter._validate_flat_list(input_board), board_format)
            case BoardFormats.STRING:
                return (T3Converter._validate_string(input_board), board_format)
            case _:
                return (False, board_format)

    @staticmethod
    def _validate_flat_list(board: list) -> bool:
        if not isinstance(board, list) or len(board) != 9:
            return False
        return T3Converter._validate_cells(board)

    @staticmethod
    def _validate_string(board: str) -> bool:
        if not isinstance(board, str) or len(board) != 9:
            return False
        return T3Converter._validate_cells(list(board))

    @staticmethod
    def _validate_cells(cells: list) -> bool:
        for cell in cells:
            if not isinstance(cell, str) or len(cell) != 1 or cell not in VALID_MOVES:
                return False
        return True

    """
    @staticmethod
    def convert_to_tensor(board_string: str) -> torch.Tensor:
        assert len(board_string) == 9, "Board string must be of length 9"

        # Create a tensor of zeros with the appropriate shape
        board_tensor = torch.zeros(1, 3, 3, 3)

        # Iterate over the characters in the board string
        for i, cell in enumerate(board_string):
            # Calculate the corresponding row and column in the 3x3 grid
            row = i // 3
            col = i % 3

            # Set the appropriate channel to 1 based on the cell contents
            if cell == " ":
                board_tensor[0, 0, row, col] = 1
            elif cell == "X":
                board_tensor[0, 1, row, col] = 1
            elif cell == "O":
                board_tensor[0, 2, row, col] = 1
            else:
                raise ValueError(f"Invalid character '{cell}' in board string")

        return board_tensor
    """
