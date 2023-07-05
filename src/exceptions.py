class InvalidBoardException(Exception):
    """Exception raised when the board format is invalid."""

    def __init__(self, message="Invalid board format.") -> None:
        super().__init__(message)


class GameFinishedException(Exception):
    """Exception raised when the game is finished."""

    def __init__(self, message="The game is already finished.") -> None:
        super().__init__(message)
