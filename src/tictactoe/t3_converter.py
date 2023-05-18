from .t3_constants import VALID_MOVES, BoardFormats


class T3Converter:
    @staticmethod
    def detect_format(input_board: str | list) -> BoardFormats:
        if isinstance(input_board, list) and len(input_board) == 9:
            return BoardFormats.FLAT_LIST

        if isinstance(input_board, str) and len(input_board) == 9:
            return BoardFormats.STRING

        raise ValueError("Invalid input board format")

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
    def validate_board(input_board) -> bool:
        board_format = T3Converter.detect_format(input_board)

        match board_format:
            case BoardFormats.FLAT_LIST:
                return T3Converter._validate_flat_list(input_board)
            case BoardFormats.STRING:
                return T3Converter._validate_string(input_board)
            case _:
                return False

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
