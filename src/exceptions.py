class InvalidBoardException(Exception):
    """Exception raised when the board format is invalid."""

    def __init__(self, message="Invalid board format.") -> None:
        super().__init__(message)
