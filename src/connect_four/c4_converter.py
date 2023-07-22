from typing import Tuple
from .c4_constants import VALID_MOVES, BoardFormats


class C4Converter:
    @staticmethod
    def detect_format(input_board: str | list) -> BoardFormats:
        if isinstance(input_board, list):
            return BoardFormats.FLAT_LIST

        if isinstance(input_board, str):
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
    def validate_board(input_board, dimension) -> Tuple[bool, BoardFormats]:
        board_format = C4Converter.detect_format(input_board)
        rows, columns = dimension
        expected_length = rows * columns

        match board_format:
            case BoardFormats.FLAT_LIST:
                return (
                    C4Converter._validate_flat_list(input_board, expected_length),
                    board_format,
                )
            case BoardFormats.STRING:
                return (
                    C4Converter._validate_string(input_board, expected_length),
                    board_format,
                )
            case _:
                return (False, board_format)

    @staticmethod
    def _validate_flat_list(board: list, expected_length: int) -> bool:
        if not isinstance(board, list) or len(board) != expected_length:
            return False
        return C4Converter._validate_cells(board)

    @staticmethod
    def _validate_string(board: str, expected_length: int) -> bool:
        if not isinstance(board, str) or len(board) != expected_length:
            return False
        return C4Converter._validate_cells(list(board))

    @staticmethod
    def _validate_cells(cells: list) -> bool:
        for cell in cells:
            if not isinstance(cell, str) or len(cell) != 1 or cell not in VALID_MOVES:
                return False
        return True
