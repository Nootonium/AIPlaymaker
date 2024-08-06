from typing import Tuple
from .t3_constants import VALID_MOVES, BoardFormats
import numpy as np


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


def encode_board(board_state: str) -> np.ndarray:
    mapping = {"X": [1, 0, 0], "O": [0, 1, 0], " ": [0, 0, 1]}
    return np.array([mapping[char] for char in board_state])


def encode_moves(moves):
    encoded_moves = np.zeros(9)
    for move in moves:
        encoded_moves[move] = 1
    return encoded_moves
